import os
import time
import logging
import asyncio

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VEO_MODEL = os.getenv("VEO_MODEL", "veo-3.1-fast-generate-preview")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I'm MotionPixel \U0001F4F8\u2192\U0001F3AC\n\n"
        "Send me a photo WITH A CAPTION describing how you want it animated.\n"
        "Example: send a photo of a lake, caption it 'gentle waves and birds flying over the water'.\n\n"
        "Generation usually takes 1-2 minutes, so please be patient after sending."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Just send a photo with a caption (your animation prompt) and I'll turn it into a short video.\n\n"
        "Tips for good prompts:\n"
        "- Describe motion, not just the scene (e.g. 'waves rolling in slowly')\n"
        "- Keep it to one clear action\n"
        "- Avoid text/logos in the source image, results are less reliable"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    prompt = message.caption

    if not prompt:
        await message.reply_text(
            "Please resend the photo WITH a caption describing the motion you want "
            "(the caption is used as your prompt)."
        )
        return

    status_message = await message.reply_text(
        "Got it! Generating your video, this usually takes 1-2 minutes..."
    )

    try:
        photo = message.photo[-1]  # highest resolution version
        tg_file = await photo.get_file()
        image_bytes = bytes(await tg_file.download_as_bytearray())

        video_path = await generate_video_from_image(image_bytes, prompt)

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
        "Please send a photo with a caption describing the motion you want, not just text. "
        "Use /help for tips."
    )


# ---------------------------------------------------------------------------
# Gemini / Veo video generation
# ---------------------------------------------------------------------------

async def generate_video_from_image(image_bytes: bytes, prompt: str) -> str:
    """
    Calls the Gemini API (Veo) to generate a video from an image + text prompt.
    The Gemini SDK calls are blocking, so they run in a background thread
    to avoid freezing the bot while polling for the result.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate_video_blocking, image_bytes, prompt)


def _generate_video_blocking(image_bytes: bytes, prompt: str) -> str:
    image = types.Image(image_bytes=image_bytes, mime_type="image/jpeg")

    operation = gemini_client.models.generate_videos(
        model=VEO_MODEL,
        prompt=prompt,
        image=image,
        config=types.GenerateVideosConfig(
            number_of_videos=1,
        ),
    )

    # Poll until the generation job finishes
    while not operation.done:
        time.sleep(10)
        operation = gemini_client.operations.get(operation)

    if operation.response is None or not operation.response.generated_videos:
        raise RuntimeError("No video was returned by the API")

    generated_video = operation.response.generated_videos[0]
    output_path = f"/tmp/video_{int(time.time())}.mp4"
    generated_video.video.save(output_path)
    return output_path


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
