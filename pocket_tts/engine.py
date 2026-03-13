import onnxruntime as ort
import numpy as np
import re
from typing import Generator, cast
import os

from .config import Config
from .tokenizer import Tokenizer, split_sentences
from .audio import load_audio, float_to_s16le


def natural_sort_key(s):
    """Sorts 'state_10' after 'state_2'."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]

class StatefulRunner:
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_feed = {}

        # Pre-calculate indices and names
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        self.state_inputs = sorted([n.name for n in inputs if n.name.startswith("state_")], key=natural_sort_key)
        self.state_outputs = sorted([n.name for n in outputs if n.name.startswith("out_state_")], key=natural_sort_key)

        # Map state output index to input name for fast updates
        output_names = [o.name for o in outputs]
        self.state_mapping = [
            (output_names.index(out_n), in_n)
            for out_n, in_n in zip(self.state_outputs, self.state_inputs)
        ]

        self.non_state_inputs = [n.name for n in inputs if not n.name.startswith("state_")]
        self.reset_state()

    def reset_state(self):
        self.input_feed.clear()
        for node in self.session.get_inputs():
            shape = [d if (isinstance(d, int) and d > 0) else 0 for d in node.shape]
            dtype = np.float32 if 'float' in node.type else (bool if 'bool' in node.type else np.int64)
            self.input_feed[node.name] = np.zeros(shape, dtype=dtype)

    def run(self, inputs: dict) -> list[np.ndarray]:
        # In-place update (no new dict created)
        for k, v in inputs.items():
            self.input_feed[k] = v

        outputs = self.session.run(None, self.input_feed)

        # Update states in-place for next frame
        for out_idx, in_name in self.state_mapping:
            self.input_feed[in_name] = outputs[out_idx]

        return outputs

class LatentGen:
    def __init__(self, main_runner: StatefulRunner, txt_sess: ort.InferenceSession,
                 flow_sess: ort.InferenceSession, voice_emb: np.ndarray,
                 tokens: np.ndarray, config: Config):
        self.main = main_runner
        self.flow = flow_sess
        self.config = config

        # Pre-calculate constants for the flow matching loop
        self.dt = np.float32(1.0 / config.lsd_steps)
        self.st_values = [
            (np.float32(j * self.dt), np.float32((j + 1) * self.dt))
            for j in range(config.lsd_steps)
        ]
        self.temp = np.float32(np.sqrt(config.temperature))

        # Pre-allocate noise buffer to avoid calling np.random in the hot loop
        # 2000 frames covers ~16 seconds of audio, enough for most sentences.
        self.noise_ptr = 0
        self.noise_buffer = np.random.randn(2000, 32).astype(np.float32)
        if config.noise_clamp > 0:
            np.clip(self.noise_buffer, -config.noise_clamp, config.noise_clamp, out=self.noise_buffer)
        self.noise_buffer *= self.temp

        # Pre-allocate static empty conditioner for the next() AR steps
        self.empty_cond = np.zeros((1, 0, 1024), dtype=np.float32)

        # Internal state tracking
        self.done = False
        self.eos_detected = False
        self.extra_frames_count = 0

        # Initialize cl (current latent) with NaN as the Start-of-Sequence marker
        self.cl = np.full((1, 1, 32), np.nan, dtype=np.float32)

        # 1. Reset the Transformer KV cache/states
        self.main.reset_state()

        # 2. Generate the text conditioning embedding
        txt_in_name = txt_sess.get_inputs()[0].name
        temb = txt_sess.run(None, {txt_in_name: tokens.astype(np.int64)})[0]

        # 3. Conditioning Passes (The "Priming" phase)
        # We pass an empty sequence for the latents and the real context for conditioning
        empty_seq = np.zeros((1, 0, 32), dtype=np.float32)

        # Pass 1: Voice Conditioning
        self.main.run({
            self.main.non_state_inputs[0]: empty_seq,
            self.main.non_state_inputs[1]: voice_emb.astype(np.float32)
        })

        # Pass 2: Text Conditioning
        self.main.run({
            self.main.non_state_inputs[0]: empty_seq,
            self.main.non_state_inputs[1]: temb.astype(np.float32)
        })

    def next(self) -> np.ndarray | None:
        if self.done: return None

        # 1. Main LM Step - Use pre-allocated empty_cond
        outputs = self.main.run({
            self.main.non_state_inputs[0]: self.cl,
            self.main.non_state_inputs[1]: self.empty_cond
        })

        cond = outputs[0]
        if not self.eos_detected and outputs[1].flatten()[0] > self.config.eos_threshold:
            self.eos_detected = True

        if self.eos_detected:
            self.extra_frames_count += 1
            if self.extra_frames_count > self.config.eos_extra_frames:
                self.done = True
                return None

        # 2. Flow Matching - Use pre-sampled noise
        fx = self.noise_buffer[self.noise_ptr : self.noise_ptr + 1]
        self.noise_ptr = (self.noise_ptr + 1) % 2000

        flow_inputs = [i.name for i in self.flow.get_inputs()]
        for s, t in self.st_values:
            # We must use dict here because ORT requires it, but we can reuse the keys
            f_out = self.flow.run(None, {
                flow_inputs[0]: cond,
                flow_inputs[1]: np.array([[s]], dtype=np.float32),
                flow_inputs[2]: np.array([[t]], dtype=np.float32),
                flow_inputs[3]: fx
            })[0]
            # Use += to update in-place
            fx = fx + (f_out * self.dt)

        self.cl = fx.reshape(1, 1, 32)
        return self.cl

class PocketTTS:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.tokenizer = Tokenizer(config.tokenizer_path)

        opts = ort.SessionOptions()
        sfx = "_int8" if config.precision == "int8" else ""
        mdir = config.models_dir

        self.enc = ort.InferenceSession(os.path.join(mdir, "mimi_encoder.onnx"), opts)
        self.txt = ort.InferenceSession(
            os.path.join(mdir, "text_conditioner.onnx"), opts
        )
        self.main_sess = ort.InferenceSession(
            os.path.join(mdir, f"flow_lm_main{sfx}.onnx"), opts
        )
        self.flow = ort.InferenceSession(
            os.path.join(mdir, f"flow_lm_flow{sfx}.onnx"), opts
        )
        self.dec = ort.InferenceSession(
            os.path.join(mdir, f"mimi_decoder{sfx}.onnx"), opts
        )

        self.main_runner = StatefulRunner(self.main_sess)
        self.dec_runner = StatefulRunner(self.dec)

    def encode_voice(self, wav_path: str) -> np.ndarray:
        audio = load_audio(wav_path, self.config.sample_rate)
        audio_tensor = audio[np.newaxis, np.newaxis, :].astype(np.float32)
        enc_in = self.enc.get_inputs()[0].name
        voice_emb = self.enc.run(None, {enc_in: audio_tensor})[0]
        while len(voice_emb.shape) > 3:
            voice_emb = voice_emb[0]
        if len(voice_emb.shape) < 3:
            voice_emb = np.expand_dims(voice_emb, axis=0)
        return voice_emb.astype(np.float32)

    def stream_raw_pcm(
        self, text: str, voice_emb: np.ndarray
    ) -> Generator[bytes, None, None]:
        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]

        # Reset the Mimi decoder states once at the start of the stream
        self.dec_runner.reset_state()
        dec_in_name = self.dec_runner.non_state_inputs[0]

        for sentence in sentences:
            tokens = self.tokenizer.encode(sentence)
            tokens_tensor = np.array(tokens, dtype=np.int64).reshape(1, -1)

            # The main LM is reset inside LatentGen for each sentence
            gen = LatentGen(
                self.main_runner,
                self.txt,
                self.flow,
                voice_emb,
                tokens_tensor,
                self.config,
            )

            latents = []
            first_chunk = True

            while True:
                latent = gen.next()
                if latent is None:
                    break
                latents.append(latent)

                limit = (
                    self.config.first_chunk_frames
                    if first_chunk
                    else self.config.max_chunk_frames
                )
                if len(latents) >= limit:
                    batch = np.concatenate(latents, axis=1)
                    # Pass the accumulated latents through the stateful Mimi decoder
                    audio_float = self.dec_runner.run({dec_in_name: batch})[0].flatten()
                    yield float_to_s16le(audio_float).tobytes()
                    latents = []
                    first_chunk = False

            if latents:
                batch = np.concatenate(latents, axis=1)
                audio_float = self.dec_runner.run({dec_in_name: batch})[0].flatten()
                yield float_to_s16le(audio_float).tobytes()

    def generate(self, text: str, voice_emb: np.ndarray) -> np.ndarray:
        all_chunks = []
        for pcm_bytes in self.stream_raw_pcm(text, voice_emb):
            all_chunks.append(np.frombuffer(pcm_bytes, dtype=np.int16))
        return (
            np.concatenate(all_chunks) if all_chunks else np.array([], dtype=np.int16)
        )

    @staticmethod
    def export_voice(voice_emb: np.ndarray, path):
        """Saves the voice embedding to a .npy file."""
        np.save(path, voice_emb)

    @staticmethod
    def import_voice(path) -> np.ndarray:
        """Loads a voice embedding from a .npy file."""
        return np.load(path)
