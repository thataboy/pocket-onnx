import time
import scipy.io.wavfile as wavfile
from pocket_tts.config import Config
from pocket_tts.engine import PocketTTS

def main():
    # Setup configuration. Make sure models and tokenizer exist in the specified paths
    config = Config(
        models_dir="models",
        tokenizer_path="models/tokenizer.model",
        precision="int8"
    )
    
    # Initialize Engine
    print("Loading ONNX models...")
    tts = PocketTTS(config)
    
    text = "She tore her gaze away from her ruined footwear, still very much grieving the loss, and set herself to the task of finding a dry place to stand."
    voice_sample_path = "../pocket-tts/voices/narrator.wav"
    output_wav_path = "output.wav"
    output_raw_path = "output.pcm"

    # --- 1) Voice Cloning (Load model state from WAV sample) ---
    # print(f"Encoding voice from {voice_sample_path}...")
    voice_emb = tts.encode_voice(voice_sample_path)
    PocketTTS.export_voice(voice_emb, "narrator")
    # voice_emb = PocketTTS.import_voice("narrator.npy")
    
    # --- 2) Stream raw PCM s16le (Suitable for streaming/sockets) ---
    # print(f"Streaming raw PCM data to {output_raw_path}...")
    # with open(output_raw_path, "wb") as f_raw:
    #     # stream_raw_pcm yields bytes in int16 format
    #     for chunk_bytes in tts.stream_raw_pcm(text, voice_emb):
    #         f_raw.write(chunk_bytes)
    #         # You can send chunk_bytes over a WebSocket or HTTP chunked response here

    # --- 3) Generate complete WAV file (s16le 24000hz) ---
    print(f"Generating complete WAV file to {output_wav_path}...")
    for i in range(1, 5):
        t0 = time.perf_counter()
        audio = tts.generate(text, voice_emb)
        elapsed = time.perf_counter() - t0
        wavfile.write(f"out{i}.wav", config.sample_rate, audio)
        duration = len(audio) / 24000
        rtfx = duration / elapsed
        print(f"dur={duration:.2f}s time={elapsed:.3f}s  spd={rtfx:.2f}x")
    print("Done!")

if __name__ == "__main__":
    main()