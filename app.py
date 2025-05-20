from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils import save_audio_to_file, get_predefined_voices
from datetime import datetime
import os
import logging

from engine import generate_speech, EXPECTED_SAMPLE_RATE, load_model
from config import (
    get_predefined_voices_path,
    get_gen_default_cfg_scale,
    get_gen_default_temperature,
    get_gen_default_top_p,
    get_gen_default_cfg_filter_top_k,
    get_gen_default_split_text,
    get_gen_default_chunk_size,
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("inference")

# --- App Setup ---
app = FastAPI()
OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)
logger.info(f"Output path ensured at: {OUTPUT_PATH}")

@app.on_event("startup")
async def startup_event():
    logger.info("🔄 Loading Dia TTS model...")
    success = load_model()
    if not success:
        logger.error("❌ Model failed to load.")
        raise RuntimeError("Model load failed.")
    logger.info("✅ Model loaded successfully.")

# --- Request Model ---
class SimpleTTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    seed: int = 42
    voice: str = "default"

# --- Health Check ---
@app.get("/")
async def health_check():
    return {"status": "running"}

# --- TTS Endpoint ---
@app.post("/simple-tts")
async def simple_tts(request: SimpleTTSRequest):
    logger.info(f"🎤 TTS request: text_len={len(request.text)}, speed={request.speed}, seed={request.seed}, voice='{request.voice}'")

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
            split_text=get_gen_default_split_text(),
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

        return JSONResponse({"status": "success", "file_path": f"/outputs/{filename}"})

    except Exception as e:
        logger.exception("❗ Error during TTS generation")
        raise HTTPException(status_code=500, detail=str(e))

# --- Entry Point ---
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting FastAPI on http://0.0.0.0:8000")
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=False)
