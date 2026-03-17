import onnxruntime as ort
import numpy as np
import re
from functools import lru_cache
from pathlib import Path
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

class IOStatefulRunner:
    """High-performance runner using IOBinding for fixed-shape Transformer states."""
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.io_binding = self.session.io_binding()
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.state_inputs = sorted([n for n in self.input_names if n.startswith("state_")], key=natural_sort_key)
        self.state_outputs = sorted([n for n in self.output_names if n.startswith("out_state_")], key=natural_sort_key)
        self.state_pairs = list(zip(self.state_inputs, self.state_outputs))
        self.non_state_inputs = [n for n in self.input_names if not n.startswith("state_")]
        self.reset_state()

    def reset_state(self):
        for node in self.session.get_inputs():
            if node.name.startswith("state_"):
                shape = [d if (isinstance(d, int) and d > 0) else 0 for d in node.shape]
                dtype = np.float32 if 'float' in node.type else (bool if 'bool' in node.type else np.int64)
                self.io_binding.bind_cpu_input(node.name, np.zeros(shape, dtype=dtype))
        for out_name in self.output_names:
            self.io_binding.bind_output(out_name)

    def run(self, inputs: dict) -> list[np.ndarray]:
        for k, v in inputs.items():
            self.io_binding.bind_cpu_input(k, v)
        self.session.run_with_iobinding(self.io_binding)
        outputs = self.io_binding.get_outputs()
        out_dict = {name: val for name, val in zip(self.output_names, outputs)}
        for in_name, out_name in self.state_pairs:
            self.io_binding.bind_ortvalue_input(in_name, out_dict[out_name])
        # Only return non-state outputs (latent and eos logit)
        return [out_dict[n].numpy() for n in self.output_names if not n.startswith("out_state_")]

class SimpleStatefulRunner:
    """Reliable runner for models with dynamic shapes (like the Mimi Decoder)."""
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.state = {}
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.state_inputs = sorted([n for n in self.input_names if n.startswith("state_")], key=natural_sort_key)
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.state_outputs = sorted([n for n in self.output_names if n.startswith("out_state_")], key=natural_sort_key)
        self.non_state_inputs = [n for n in self.input_names if not n.startswith("state_")]
        self.reset_state()

    def reset_state(self):
        self.state.clear()
        for node in self.session.get_inputs():
            if node.name.startswith("state_"):
                shape = [d if (isinstance(d, int) and d > 0) else 0 for d in node.shape]
                dtype = np.float32 if 'float' in node.type else (bool if 'bool' in node.type else np.int64)
                self.state[node.name] = np.zeros(shape, dtype=dtype)

    def run(self, inputs: dict) -> list[np.ndarray]:
        run_inputs = {**inputs, **self.state}
        outputs = self.session.run(None, run_inputs)
        out_dict = dict(zip(self.output_names, outputs))
        for out_name, in_name in zip(self.state_outputs, self.state_inputs):
            self.state[in_name] = out_dict[out_name]
        return [out_dict[n] for n in self.output_names if not n.startswith("out_state_")]

class LatentGen:
    def __init__(self, main_runner: IOStatefulRunner, txt_sess: ort.InferenceSession,
                 flow_sess: ort.InferenceSession, voice_emb: np.ndarray,
                 tokens: np.ndarray, config: Config):
        self.main = main_runner
        self.flow = flow_sess
        self.config = config

        # Pre-calculate constants
        self.dt = np.float32(1.0 / config.lsd_steps)
        self.st_values = [(np.float32(j * self.dt), np.float32((j + 1) * self.dt)) for j in range(config.lsd_steps)]
        self.temp = np.float32(np.sqrt(config.temperature))

        # Dynamic Name Mapping (Fixes the 'c' vs 'cond' error)
        flow_inputs = self.flow.get_inputs()
        self.flow_names = [i.name for i in flow_inputs]

        # Pre-allocate loop tensors (Saves ~10us per frame)
        self.s_zero = np.array([[0.0]], dtype=np.float32)
        self.t_one = np.array([[1.0]], dtype=np.float32)
        self.empty_cond = np.zeros((1, 0, 1024), dtype=np.float32)

        # Pre-allocate noise
        self.noise_ptr = 0
        self.noise_buffer = np.random.randn(2000, 32).astype(np.float32)
        if config.noise_clamp > 0:
            np.clip(self.noise_buffer, -config.noise_clamp, config.noise_clamp, out=self.noise_buffer)
        self.noise_buffer *= self.temp

        self.done = False
        self.eos_detected = False
        self.extra_frames_count = 0
        self.cl = np.full((1, 1, 32), np.nan, dtype=np.float32)

        # Prime the main model
        self.main.reset_state()
        temb = txt_sess.run(None, {txt_sess.get_inputs()[0].name: tokens.astype(np.int64)})[0]
        empty_seq = np.zeros((1, 0, 32), dtype=np.float32)

        self.main.run({self.main.non_state_inputs[0]: empty_seq, self.main.non_state_inputs[1]: voice_emb.astype(np.float32)})
        self.main.run({self.main.non_state_inputs[0]: empty_seq, self.main.non_state_inputs[1]: temb.astype(np.float32)})

    def next(self) -> np.ndarray | None:
        if self.done: return None

        # Main LM Step
        main_out = self.main.run({
            self.main.non_state_inputs[0]: self.cl,
            self.main.non_state_inputs[1]: self.empty_cond
        })

        cond = main_out[0]
        if not self.eos_detected and main_out[1].item() > self.config.eos_threshold:
            self.eos_detected = True

        if self.eos_detected:
            self.extra_frames_count += 1
            if self.extra_frames_count > self.config.eos_extra_frames:
                self.done = True
                return None

        # Flow Step - Final optimization for 8x
        fx = self.noise_buffer[self.noise_ptr : self.noise_ptr + 1]
        self.noise_ptr = (self.noise_ptr + 1) % 2000

        # Call flow model once (lsd_steps=1)
        # Using a list for inputs is slightly faster than a dict in some ORT versions
        f_out = self.flow.run(None, {
            self.flow_names[0]: cond,
            self.flow_names[1]: self.s_zero,
            self.flow_names[2]: self.t_one,
            self.flow_names[3]: fx
        })[0]

        self.cl = (fx + f_out).reshape(1, 1, 32)
        return self.cl

class PocketTTS:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.tokenizer = Tokenizer(config.tokenizer_path)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        sfx = "_int8" if config.precision == "int8" else ""
        mdir = config.models_dir

        self.enc = ort.InferenceSession(os.path.join(mdir, "mimi_encoder.onnx"), opts)
        self.txt = ort.InferenceSession(
            os.path.join(mdir, "text_conditioner.onnx"), opts
        )
        self.main_sess = ort.InferenceSession(
            os.path.join(mdir, f"flow_lm_main{sfx}.onnx"), opts,
        )
        self.flow = ort.InferenceSession(
            os.path.join(mdir, f"flow_lm_flow{sfx}.onnx"), opts
        )
        self.dec = ort.InferenceSession(
            os.path.join(mdir, f"mimi_decoder{sfx}.onnx"), opts
        )

        # Use IOBinding for the heavy Transformer
        self.main_runner = IOStatefulRunner(self.main_sess)

        # Use Simple runner for the dynamic Decoder
        self.dec_runner = SimpleStatefulRunner(self.dec)

    def encode_voice(self, wav_path: str | Path) -> np.ndarray:
        audio = load_audio(wav_path, self.config.sample_rate)
        audio_tensor = audio[np.newaxis, np.newaxis, :].astype(np.float32)
        enc_in = self.enc.get_inputs()[0].name
        voice_emb = self.enc.run(None, {enc_in: audio_tensor})[0]
        while len(voice_emb.shape) > 3:
            voice_emb = voice_emb[0]
        if len(voice_emb.shape) < 3:
            voice_emb = np.expand_dims(voice_emb, axis=0)
        return voice_emb.astype(np.float32)

    def stream(
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
        for pcm_bytes in self.stream(text, voice_emb):
            all_chunks.append(np.frombuffer(pcm_bytes, dtype=np.int16))
        return (
            np.concatenate(all_chunks) if all_chunks else np.array([], dtype=np.int16)
        )

    @staticmethod
    def export_voice(voice_emb: np.ndarray, path):
        """Saves the voice embedding to a .npy file."""
        np.save(path, voice_emb)

    @staticmethod
    @lru_cache(maxsize=2)
    def import_voice(path) -> np.ndarray:
        """Loads a voice embedding from a .npy file."""
        return np.load(path)
