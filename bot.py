import os
import time
import logging
import asyncio
import subprocess

import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import fal_client
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
FAL_KEY = os.getenv("FAL_KEY")
FAL_MODEL = os.getenv("FAL_MODEL", "fal-ai/kling-video/v1.6/standard/image-to-video")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set")
if not FAL_KEY:
    raise RuntimeError("FAL_KEY environment variable is not set")

# fal_client reads the FAL_KEY environment variable automatically,
# but we double check it's set above so the bot fails fast with a clear error.

# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I'm MotionPixel \U0001F4F8\u2192\U0001F3AC\n\n"
        "Send me a photo WITH A CAPTION. The caption will be used as:\n"
        "1) A short motion description for the video, and\n"
        "2) The narration the video will 'say' out loud.\n\n"
        "Keep captions short and clear, e.g. 'Water is essential for human health "
        "and helps every cell in the body function properly.'\n\n"
        "Generation usually takes 1-2 minutes, so please be patient after sending."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send a photo with a caption. The caption becomes the spoken narration AND "
        "the motion prompt for the video.\n\n"
        "Tips:\n"
        "- Keep it to 1-2 short sentences (clips are only a few seconds long)\n"
        "- Clear, simple language works best for narration\n"
        "- Avoid text/logos in the source image, results are less reliable"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    prompt = message.caption

    if not prompt:
        await message.reply_text(
            "Please resend the photo WITH a caption — that caption becomes both "
            "the narration and the motion prompt."
        )
        return

    status_message = await message.reply_text(
        "Got it! Generating your video, this usually takes 1-2 minutes..."
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
            f"Sorry, something went wrong generating your video:\n{str(e)[:300]}"
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Please send a photo with a caption, not just text. Use /help for tips."
    )


# ---------------------------------------------------------------------------
# Video generation (fal.ai / Kling) + narration (gTTS) + merge (ffmpeg)
# ---------------------------------------------------------------------------

async def generate_talking_video(image_bytes: bytes, prompt: str) -> str:
    """
    Full pipeline:
    1. Upload the image to fal.ai and generate a short silent video (motion).
    2. Generate narration audio from the prompt text (gTTS).
    3. Merge the audio onto the video with ffmpeg.
    Runs the blocking work in a background thread so the bot stays responsive.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate_talking_video_blocking, image_bytes, prompt)


def _generate_talking_video_blocking(image_bytes: bytes, prompt: str) -> str:
    ts = int(time.time())

    # 1. Save and upload the source image
    input_image_path = f"/tmp/input_{ts}.jpg"
    with open(input_image_path, "wb") as f:
        f.write(image_bytes)

    image_url = fal_client.upload_file(input_image_path)

    # 2. Generate the silent motion video via fal.ai (Kling)
    result = fal_client.subscribe(
        FAL_MODEL,
        arguments={
            "prompt": prompt,
            "image_url": image_url,
        },
        with_logs=False,
    )

    video_remote_url = result["video"]["url"]

    raw_video_path = f"/tmp/raw_{ts}.mp4"
    _download_file(video_remote_url, raw_video_path)

    # 3. Generate narration audio from the same prompt text
    narration_path = f"/tmp/narration_{ts}.mp3"
    tts = gTTS(text=prompt, lang="en")
    tts.save(narration_path)

    # 4. Merge narration onto the video with ffmpeg
    #    -shortest trims to whichever is shorter (video is usually ~5s,
    #    so keep captions short or the narration will get cut off)
    final_path = f"/tmp/final_{ts}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", raw_video_path,
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

    # Clean up intermediate files
    for p in (input_image_path, raw_video_path, narration_path):
        if os.path.exists(p):
            os.remove(p)

    return final_path


def _download_file(url: str, dest_path: str):
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


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
