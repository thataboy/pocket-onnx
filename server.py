import asyncio
import io
import logging
import numpy as np
import os
import scipy.io.wavfile as wavfile
import threading
import time
import uvicorn

from os.path import getmtime
from pathlib import Path
from queue import Empty, Queue

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from html import escape
from pydantic import BaseModel
from urllib.parse import quote

from italk.italk import router as italk_router  # iTalk logic module
from pocket_tts.config import Config
from pocket_tts.engine import PocketTTS

logger = logging.getLogger(__name__)

VOICES_PATH = "./voices"


class StaticFilesEx(StaticFiles):
    async def get_response(self, path: str, scope):
        full_path, stat_result = self.lookup_path(path)

        if stat_result and os.path.isdir(full_path):
            # Serve index.html if exists
            for idx in ["index.html", "index.htm", "index.xhtml"]:
                index_path = os.path.join(full_path, idx)
                if os.path.exists(index_path):
                    return await super().get_response(
                        os.path.join(path, idx), scope
                    )

            entries = os.listdir(full_path)

            # Sort: directories first, then files
            dirs = []
            files = []

            for name in entries:
                if name.startswith("."):
                    continue
                abs_entry = os.path.join(full_path, name)
                if os.path.isdir(abs_entry):
                    dirs.append(name)
                else:
                    files.append(name)

            dirs.sort()
            files.sort()

            items = []

            # Parent link stays relative
            if path not in ("", "/", "."):
                items.append('<li><a href="../">.. (parent directory)</a></li>')

            # Directories
            for name in dirs:
                href = quote(name) + "/"
                items.append(f'<li><a href="{href}">{escape(name)}/</a></li>')

            # Files
            for name in files:
                href = quote(name)
                items.append(f'<li><a href="{href}">{escape(name)}</a></li>')

            html = f"""
            <html>
                <head>
                    <title>Index of /{escape(path)}</title>
                    <style>li {{ margin: 1em 1em; }}</style>
                </head>
                <body>
                    <h2>Index of /{escape(path)}</h2>
                    <ul>
                        {''.join(items)}
                    </ul>
                </body>
            </html>
            """

            return HTMLResponse(content=html)

        return await super().get_response(path, scope)

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)


@web_app.get("/voices")
def list_voices():
    """Return list of available voice style names."""
    return {"voices": sorted(list(voices.keys()))}


@web_app.get("/voices/refresh")
def refresh_voices():
    process_voices()
    return {"voices": sorted(list(voices.keys()))}


class SynthesizeRequest(BaseModel):
    text: str = ""
    voice: str = ""


@web_app.post("/synthesize")
@web_app.post("/v1/audio/speech")
async def synthesize(req: SynthesizeRequest, request: Request):
    """Generate complete text in one go with graceful abort on disconnect."""
    if not req.text.strip():
        raise HTTPException(status_code=406, detail="Text cannot be empty")

    if req.voice not in voices:
        req.voice = next(iter(voices.keys())) if len(voices) > 0 else ""
        if not req.voice:
            raise HTTPException(status_code=407, detail="No voice found")

    print(f"{req.voice}➡️{req.text}⬅️")
    t0 = time.perf_counter()

    voice_emb = tts_model.import_voice(voices[req.voice])

    # # 1. Setup thread-safe communication
    # queue = Queue()
    # cancel_event = threading.Event()

    # def producer():
    #     """Worker thread to run the blocking TTS generation."""
    #     try:
    #         chunks_iter = tts_model.stream(req.text, voice_emb)
    #         for chunk in chunks_iter:
    #             if cancel_event.is_set():
    #                 return
    #             queue.put(chunk)
    #     except Exception as e:
    #         logger.exception("Synthesis failed in thread: %s", e)
    #     finally:
    #         queue.put(None)  # Signal EOF

    # # 2. Start generation in background
    # thread = threading.Thread(target=producer, daemon=True)
    # thread.start()

    # all_chunks = []

    # # 3. Async collection loop
    # try:
    #     while True:
    #         # Check if user closed the connection/browser tab
    #         if await request.is_disconnected():
    #             print("Client disconnected. Aborting synthesis...")
    #             cancel_event.set()
    #             break

    #         try:
    #             # Use non-blocking get to keep the event loop responsive
    #             chunk = queue.get_nowait()
    #         except Empty:
    #             # Yield control to the event loop for a moment
    #             await asyncio.sleep(0.01)
    #             continue

    #         if chunk is None:  # Worker finished
    #             break
    #         all_chunks.append(np.frombuffer(chunk, dtype=np.int16))
    # finally:
    #     # Ensure thread cleanup
    #     cancel_event.set()
    #     thread.join(timeout=0.2)

    # # 4. If we aborted, return a partial or empty response
    # if not all_chunks:
    #     return Response(status_code=204)

    # # 5. Build the final WAV response
    # audio = np.concatenate(all_chunks)

    buffer = io.BytesIO()
    audio = tts_model.generate(req.text, voice_emb)
    wavfile.write(buffer, config.sample_rate, audio)
    duration = len(audio) / config.sample_rate
    elapsed = time.perf_counter() - t0
    spd = duration / elapsed
    print(f"[{elapsed:.3f}s] len={len(req.text)} dur={duration:.2f}s  {spd:.3f}x")

    return Response(content=buffer.getvalue(), media_type="audio/wav")

def process_voices():
    print("Loading voices...")
    voices.clear()
    for path in Path(VOICES_PATH).iterdir():
        if not path.is_file() or path.suffix not in [".npy", ".wav"]:
            continue
        voice = path.stem
        if voice in voices:
            continue
        wav = path.with_suffix(".wav")
        npy = path.with_suffix(".npy")
        if not npy.exists() or wav.exists() and getmtime(wav) > getmtime(npy):
            print(f"Extracting voice {voice}")
            voice_emb = tts_model.encode_voice(wav)
            PocketTTS.export_voice(voice_emb, npy)
        voices[voice] = npy
    print(f"{len(voices)} voices loaded")


@web_app.post("/stream")
def stream(req: SynthesizeRequest, request: Request):
    return _stream(req.text, req.voice, request)


@web_app.post("/tts")
def text_to_speech(
    request: Request,
    text: str = Form(...),
    voice: str | None = Form(None),
    voice_url: str | None = Form(None),
):
    return _stream(text, voice or voice_url, request)


async def generate_pcm_stream(
    text_to_generate: str, voice_emb: np.ndarray, request: Request | None = None
):
    queue: Queue[bytes | None] = Queue()
    cancel_event = threading.Event()

    def producer():
        try:
            # Iterate directly over the model's audio chunks
            chunks_iter = tts_model.stream(text_to_generate, voice_emb)

            for chunk in chunks_iter:
                if cancel_event.is_set():
                    return

                queue.put(chunk)

        except Exception as e:
            logger.exception("PCM generation failed: %s", e)
        finally:
            queue.put(None)  # EOF marker

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    try:
        while True:
            if request is not None and await request.is_disconnected():
                cancel_event.set()
                break

            try:
                data = queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.01)
                continue

            if data is None:
                break
            yield data
    finally:
        cancel_event.set()
        thread.join(timeout=0.2)

def _stream(text: str, voice: str | None, request: Request | None = None):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice not in voices:
        voice = next(iter(voices.keys())) if voices else ""
        if not voice:
            raise HTTPException(status_code=404, detail="No voice found")

    voice_emb = tts_model.import_voice(voices[voice])
    print(f"\n⏩{voice}➡️{text[:200]}{'...' if len(text)>200 else ''}⬅️")

    return StreamingResponse(
        generate_pcm_stream(text, voice_emb),
        media_type=f"audio/pcm; rate={config.sample_rate}",
    )

# --- App Routing ---

# Mount the iTalk backend logic at /italk prefix
web_app.include_router(italk_router, prefix="/italk")


@web_app.get("/")
@web_app.get("/index.html")
async def root():
    """Default: Serve iTalk index"""
    return FileResponse(Path(__file__).parent / "italk" / "index.html")


@web_app.get("/demo")
async def demo_page():
    """Serve demo page"""
    return FileResponse(Path(__file__).parent / "static" / "demo.html")


web_app.mount("/books", StaticFilesEx(directory="/Volumes/T7/books", html=True), name="books")
web_app.mount("/", StaticFilesEx(directory="./italk", html=True), name="italk")

@web_app.on_event("startup")
def startup():
    global voices, tts_model, config
    config = Config(
        models_dir="models",
        tokenizer_path="models/tokenizer.model",
        precision="int8"
    )
    # Initialize Engine
    print("Loading ONNX models...")
    tts_model = PocketTTS(config)
    voices = {}
    process_voices()
    if voices:
        print("Warming up...")
        voice_emb = PocketTTS.import_voice(next(iter(voices.values())))
        tts_model.generate("Hello, world.", voice_emb)
    else:
        print("No voices found!")
        exit(1)


if __name__ == "__main__":
    uvicorn.run(
        "server:web_app", host="0.0.0.0", port=9500, reload=False,
    )
