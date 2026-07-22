import os
import time
import logging
import asyncio
import subprocess
import shutil

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from gradio_client import Client, handle_file
from gtts import gTTS

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional, but strongly recommended for higher quota
HF_SPACE_ID = os.getenv("HF_SPACE_ID", "multimodalart/stable-video-diffusion")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set")

# Create the Space client once at startup (reused across requests)
gradio_client = Client(HF_SPACE_ID, hf_token=HF_TOKEN) if HF_TOKEN else Client(HF_SPACE_ID)

# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I'm MotionPixel \U0001F4F8\u2192\U0001F3AC (free / self-hosted model edition)\n\n"
        "Send me a photo WITH A CAPTION. The caption becomes the narration.\n\n"
        "Note: this runs on a free, shared community GPU, so it can be slow "
        "(1-3 minutes, sometimes longer if busy) and clips are short and simple "
        "motion only (no complex scene changes). Keep captions short!"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send a photo with a short caption. The caption becomes the spoken narration.\n\n"
        "Tips:\n"
        "- Keep captions to one short sentence (clips are only a few seconds long)\n"
        "- Simple, clear photos work best (one subject, not too busy)\n"
        "- This uses a free shared GPU, so please be patient and avoid spamming requests"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    prompt = message.caption

    if not prompt:
        await message.reply_text(
            "Please resend the photo WITH a caption — it becomes the narration."
        )
        return

    status_message = await message.reply_text(
        "Got it! Generating your video on a free shared GPU, this can take 1-3+ minutes..."
    )

    try:
        photo = message.photo[-1]  # highest resolution version
        tg_file = await photo.get_file()
        image_bytes = bytes(await tg_file.download_as_bytearray())

        video_path = await generate_talking_video(image_bytes, prompt)

        with open(video_path, "rb") as video_file:
            await message.reply_video(video=video_file, caption=f"Prompt: {prompt}")

        os.remove(video_path)
        await status_message.delete()

    except Exception as e:
        logger.exception("Video generation failed")
        await status_message.edit_text(
            f"Sorry, something went wrong generating your video:\n{str(e)[:300]}\n\n"
            "This often means the free shared GPU is busy right now — try again in a bit."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Please send a photo with a caption, not just text. Use /help for tips."
    )


# ---------------------------------------------------------------------------
# Video generation (Hugging Face Space / SVD) + narration (gTTS) + merge (ffmpeg)
# ---------------------------------------------------------------------------

async def generate_talking_video(image_bytes: bytes, prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate_talking_video_blocking, image_bytes, prompt)


def _generate_talking_video_blocking(image_bytes: bytes, prompt: str) -> str:
    ts = int(time.time())

    # 1. Save the source image locally
    input_image_path = f"/tmp/input_{ts}.jpg"
    with open(input_image_path, "wb") as f:
        f.write(image_bytes)

    # 2. Call the public Hugging Face Space to generate the silent motion video.
    #    NOTE: parameter names/order can change if the Space's author updates it.
    #    If this call errors with an "unexpected argument" style message, check
    #    the Space's current API signature at:
    #    https://huggingface.co/spaces/multimodalart/stable-video-diffusion?view=api
    #    and adjust the arguments below to match.
    result = gradio_client.predict(
        handle_file(input_image_path),  # input image
        0,          # seed
        True,       # randomize_seed
        127,        # motion_bucket_id (higher = more motion)
        6,          # fps_id
        api_name="/video",
    )

    # result is typically a tuple like (video_dict_or_path, seed_used)
    raw_video_path = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(raw_video_path, dict) and "video" in raw_video_path:
        raw_video_path = raw_video_path["video"]

    local_raw_path = f"/tmp/raw_{ts}.mp4"
    shutil.copy(raw_video_path, local_raw_path)

    # 3. Generate narration audio from the prompt text
    narration_path = f"/tmp/narration_{ts}.mp3"
    tts = gTTS(text=prompt, lang="en")
    tts.save(narration_path)

    # 4. Merge narration onto the video with ffmpeg
    final_path = f"/tmp/final_{ts}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", local_raw_path,
            "-i", narration_path,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            final_path,
        ],
        check=True,
        capture_output=True,
    )

    for p in (input_image_path, local_raw_path, narration_path):
        if os.path.exists(p):
            os.remove(p)

    return final_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot starting (polling mode)...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
