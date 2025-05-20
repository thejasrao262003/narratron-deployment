import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import PIL

import modal

# Local imports
from utils import save_audio_to_file, get_predefined_voices
from engine import load_model, generate_speech
from config import (
    get_predefined_voices_path,
    get_gen_default_cfg_scale,
    get_gen_default_temperature,
    get_gen_default_top_p,
    get_gen_default_cfg_filter_top_k,
    get_gen_default_split_text,
    get_gen_default_chunk_size,
)
from final_workflow.complete_workflow import run_workflow  # <--- Added

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("inference")

# --- Modal Setup ---
app = modal.App("dia-tts-service")

model_volume = modal.Volume.from_name("dia-tts-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.11")

    # Phase 1: Core + runtime essentials
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "numpy<2.0.0",
        "soundfile",
        "pydantic",
        "python-dotenv~=1.1.0",
        "Jinja2",
        "python-multipart",
        "requests",
        "PyYAML",
        "tqdm"
    )

    # Phase 2: ML/audio/video/media dependencies
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "openai-whisper",
        "descript-audio-codec",
        "huggingface_hub",
        "safetensors",
        "praat-parselmouth",
        "pydub~=0.25.1",
        "librosa",
        "moviepy==1.0.3",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "proglog==0.1.12",
        "Pillow==9.5.0"
    )

    # Phase 3: Google, AWS, scraping, and web
    .pip_install(
        "google-generativeai",
        "boto3==1.38.18",
        "s3transfer==0.12.0",
        "botocore==1.38.18",
        "jmespath==1.0.1",
        "httpx==0.28.1",
        "duckduckgo-search==8.0.1",
        "bing-image-downloader==1.1.2",
        "google-images-download==2.8.0",
        "google-images-downloader==1.0.16",
        "selenium==4.31.0",
        "wsproto==1.2.0",
        "trio==0.30.0",
        "pymongo",
        "trio-websocket==0.12.2",
        "outcome==1.3.0.post0",
        "PySocks==1.7.1",
        "lxml==5.4.0",
        "primp==0.15.0",
        "hf-transfer",
        "protobuf==4.25.3",
        "decorator==4.4.2",
        "sortedcontainers==2.4.0",
        force_build=True
    )
    .run_commands(
        "pip install --force-reinstall Pillow==9.5.0"
    ).run_commands(
        # Install build tools and dependencies
        "apt-get update && apt-get install -y "
        "build-essential wget xz-utils ffmpeg libmagick++-dev libfontconfig1 libxrender1 "
        "libsm6 libxext6 libx11-dev ghostscript fonts-liberation sox bc gsfonts",

        # Download and install ImageMagick
        "mkdir -p /tmp/distr && cd /tmp/distr && "
        "wget https://download.imagemagick.org/ImageMagick/download/releases/ImageMagick-7.0.11-2.tar.xz && "
        "tar xvf ImageMagick-7.0.11-2.tar.xz && "
        "cd ImageMagick-7.0.11-2 && "
        "./configure --enable-shared=yes --disable-static --without-perl && "
        "make && make install && ldconfig /usr/local/lib && "
        "cd /tmp && rm -rf distr"
    )
    .apt_install("ffmpeg", "libsndfile1")
    .add_local_dir(".", "/root/dia-tts-server")
)
print("✅ Pillow version:", PIL.__version__)

# --- FastAPI App ---
fastapi_app = FastAPI()
OUTPUT_PATH = "/tmp/outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Data Model ---
class SimpleTTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    seed: int = 42
    voice: str = "default"

@fastapi_app.on_event("startup")
async def startup_event():
    logger.info("\U0001f501 Loading Dia TTS model...")
    success = load_model()
    if not success:
        logger.error("\u274c Model failed to load.")
        raise RuntimeError("Model load failed.")
    logger.info("\u2705 Model loaded successfully.")

@fastapi_app.get("/")
async def health_check():
    return {"status": "running"}

@fastapi_app.post("/simple-tts")
async def simple_tts(request: SimpleTTSRequest):
    logger.info(f"\U0001f3a4 TTS request: text_len={len(request.text)}, speed={request.speed}, seed={request.seed}, voice='{request.voice}'")

    voice_mode = "single_s1"
    reference_file_path = None
    transcript_text = None

    if request.voice and request.voice != "default":
        voices_dir = get_predefined_voices_path()
        predefined_voices = get_predefined_voices()
        matched_voice = next(
            (v for v in predefined_voices if request.voice.lower() in [v["display_name"].lower(), v["filename"].lower()]),
            None
        )

        if matched_voice:
            reference_file_path = os.path.join(voices_dir, matched_voice["filename"])
            transcript_file_path = os.path.splitext(reference_file_path)[0] + ".txt"

            if not os.path.isfile(transcript_file_path):
                logger.warning(f"⚠️ Transcript file missing for {matched_voice['filename']}")
                raise HTTPException(status_code=400, detail="Transcript file missing for selected voice.")

            with open(transcript_file_path, "r", encoding="utf-8") as f:
                transcript_text = f.read().strip()

            voice_mode = "clone"
            logger.info(f"🗣️ Using predefined voice with transcript: {matched_voice['filename']}")
        else:
            logger.warning(f"⚠️ Voice '{request.voice}' not found. Using default.")

    try:
        audio_result = generate_speech(
            text_to_process=request.text,
            voice_mode=voice_mode,
            clone_reference_filename=reference_file_path,
            transcript=transcript_text,
            speed_factor=request.speed,
            seed=request.seed,
            max_tokens=None,
            cfg_scale=get_gen_default_cfg_scale(),
            temperature=get_gen_default_temperature(),
            top_p=get_gen_default_top_p(),
            cfg_filter_top_k=get_gen_default_cfg_filter_top_k(),
            split_text=False,
            chunk_size=get_gen_default_chunk_size(),
            enable_silence_trimming=True,
            enable_internal_silence_fix=True,
            enable_unvoiced_removal=True,
        )

        if not audio_result:
            logger.error("❌ TTS generation returned None.")
            raise HTTPException(status_code=500, detail="TTS generation failed.")

        audio_array, sample_rate = audio_result
        logger.info(f"✅ Generated audio — shape={audio_array.shape}, rate={sample_rate}")

        filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        file_path = os.path.join(OUTPUT_PATH, filename)
        save_audio_to_file(audio_array, sample_rate, file_path)
        logger.info(f"💾 Audio saved: {file_path}")

        return FileResponse(file_path, media_type="audio/wav", filename=filename)

    except Exception as e:
        logger.exception("❗ Error during TTS generation")
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/generate-short")
async def generate_short():
    print("✅ Pillow version:", PIL.__version__)
    try:
        logger.info("⚙️ Running complete horror shorts workflow...")
        run_workflow()
        return {"status": "success", "message": "Short generation complete."}
    except Exception as e:
        logger.exception("❗ Error running complete_workflow")
        raise HTTPException(status_code=500, detail=str(e))

# --- Web Entry Point ---
@app.function(
    image=image,
    gpu="L40S",
    timeout=600,
    volumes={"/model_cache": model_volume}
)
@modal.asgi_app()
def serve():
    return fastapi_app