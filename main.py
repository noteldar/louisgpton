from __future__ import annotations
from dotenv import load_dotenv
import os
import requests

load_dotenv()
import logging
import base64
from pathlib import Path
from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions

from livekit import api
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)

from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, google
from livekit.plugins.rime import TTS
from google import genai
from google.genai import types

from typing import Annotated, Union, List
import io
import aiohttp

# Email sending imports
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib  # For SMTP fallback if needed

# Add SendGrid for easier email sending
import os
import base64

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content, HtmlContent

    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False
    print("SendGrid not installed. Will use fallback method for emails.")


gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
logger = logging.getLogger("myagent")
logger.setLevel(logging.INFO)

google_llm = google.LLM(
    # model="gemini-2.0-flash-thinking-exp-01-21",
    model="gemini-2.0-flash",
    tool_choice="required",
)

rime_tts = TTS(
    model="mistv2",
    speaker="blossom",
    speed_alpha=0.9,
    reduce_latency=True,
    pause_between_brackets=True,
    phonemize_between_brackets=True,
)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    async def get_video_track(room: rtc.Room):
        """Find and return the first available remote video track in the room."""
        for participant_id, participant in room.remote_participants.items():
            for track_id, track_publication in participant.track_publications.items():
                if track_publication.track and isinstance(
                    track_publication.track, rtc.RemoteVideoTrack
                ):
                    logger.info(
                        f"Found video track {track_publication.track.sid} "
                        f"from participant {participant_id}"
                    )
                    return track_publication.track
        raise ValueError("No remote video track found in the room")

    async def get_latest_image(room: rtc.Room):
        """Capture and return a single frame from the video track."""
        video_stream = None
        try:
            video_track = await get_video_track(room)
            video_stream = rtc.VideoStream(video_track)
            async for event in video_stream:
                logger.info(f"Captured latest video frame: {type(event.frame.data)}")
                return event.frame
        except Exception as e:
            logger.error(f"Failed to get latest image: {e}")
            return None
        finally:
            if video_stream:
                await video_stream.aclose()

    async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """
        Callback that runs right before the LLM generates a response.
        Captures the current video frame and adds it to the conversation context.
        """
        logger.info("before_llm_cb")
        latest_image = await get_latest_image(ctx.room)
        if not latest_image:
            logger.warning("No image captured from video feed")
            return

        # Add the image to the conversation context
        image_content = [llm.ChatImage(image=latest_image)]
        chat_ctx.messages.append(llm.ChatMessage(role="user", content=image_content))
        logger.debug("Added latest frame to conversation context")

    def add_pdf_examples(chat_ctx):
        pdf_path = Path("data/LV.pdf")

        if not pdf_path.exists():
            logger.warning(f"PDF example file not found at {pdf_path}")
            return

        # Read the PDF file as binary data
        with open(pdf_path, "rb") as file:
            pdf_bytes = file.read()

        # Create a message with the PDF content
        logger.info(f"Adding PDF examples from {pdf_path} to chat context")
        chat_ctx.messages.append(
            llm.ChatMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "Here are examples of commenting on clothing looks:",
                    },
                    {
                        "type": "inline_data",
                        "mime_type": "application/pdf",
                        "data": base64.b64encode(pdf_bytes).decode("utf-8"),
                    },
                ],
            )
        )

        # Add a system message explaining what to do with the examples
        chat_ctx.messages.append(
            llm.ChatMessage(
                role="system",
                content="Use the above examples to guide your responses when commenting on clothing items. Try to match the style and tone of the examples. The comments are in Russian but please use English. Be detailed and comment on the fit, style and also be supportive and uplifting",
            )
        )

    # Initialize chat context with system prompt
    initial_chat_ctx = llm.ChatContext()
    initial_chat_ctx.messages.append(
        llm.ChatMessage(
            content="""
	You are helping your good friend shop for clothes.
	You are given their look from the dressing room and need to comment on their appearance in the style of the examples provided.
			""",
            role="system",
        )
    )

    # Add PDF examples to the chat context
    add_pdf_examples(initial_chat_ctx)

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    # first define a class that inherits from llm.FunctionContext
    class AssistantFnc(llm.FunctionContext):
        def _format_email_with_similar_items(self, similar_items, item_type=None):
            """Format similar items into an HTML email."""
            if not similar_items:
                return ""

            # Group items by type
            items_by_type = {}
            for item in similar_items:
                item_type = item.get("item_type", "Other")
                if item_type not in items_by_type:
                    items_by_type[item_type] = []
                items_by_type[item_type].append(item)

            # Create HTML content
            html = """
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; }
                    .item { margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 15px; }
                    .item-image { max-width: 200px; margin-right: 20px; float: left; }
                    h2 { color: #333; }
                    .price { font-weight: bold; color: #e53935; }
                    .clearfix { clear: both; }
                </style>
            </head>
            <body>
                <h1>Similar Fashion Items Found</h1>
                <p>Here are some items similar to what you were trying on:</p>
            """

            for type_name, items in items_by_type.items():
                html += f"<h2>{type_name}</h2>\n"

                for item in items:
                    html += "<div class='item'>\n"
                    if item.get("image_url"):
                        html += f"<img src='{item['image_url']}' class='item-image'>\n"

                    html += f"<h3>{item['name']}</h3>\n"
                    html += f"<p>Brand: {item['brand']}</p>\n"
                    html += f"<p class='price'>Price: {item['price']}</p>\n"
                    html += f"<p><a href='{item['product_url']}'>View Product</a></p>\n"
                    html += "<div class='clearfix'></div>\n"
                    html += "</div>\n"

            html += """
                <p>Happy shopping!</p>
            </body>
            </html>
            """

            return html

        def _send_email(self, to_email, subject, html_content):
            """Send an email with HTML content."""
            logger.info(f"Sending email to {to_email} with subject: {subject}")

            # Try to send using SendGrid if available
            if SENDGRID_AVAILABLE and os.getenv("SENDGRID_API_KEY"):
                try:
                    # Use SendGrid's API
                    sg = sendgrid.SendGridAPIClient(
                        api_key=os.getenv("SENDGRID_API_KEY")
                    )
                    from_email = Email("eldar@atarino.io")  # Your verified sender
                    to_email = To(to_email)
                    html_content = HtmlContent(html_content)
                    mail = Mail(from_email, to_email, subject, html_content)

                    # Send the email
                    response = sg.client.mail.send.post(request_body=mail.get())

                    # Log result
                    logger.info(f"Email sent via SendGrid: {response.status_code}")
                    if response.status_code >= 200 and response.status_code < 300:
                        return True
                    else:
                        logger.error(f"SendGrid error: {response.status_code}")
                        raise Exception(
                            f"SendGrid returned status code {response.status_code}"
                        )

                except Exception as e:
                    logger.error(f"Failed to send email via SendGrid: {e}")
                    # Fall through to alternative methods

            # Fall back to using a mock/logging method
            logger.info("Using mock email method since SendGrid is not configured")
            logger.info(f"Would send email to: {to_email}")
            logger.info(f"Email subject: {subject}")
            logger.info(f"Email HTML preview: {html_content[:200]}...")

            # Indicate success via the mock method
            return True

        # the llm.ai_callable decorator marks this function as a tool available to the LLM
        # by default, it'll use the docstring as the function's description
        @llm.ai_callable()
        async def feedback_on_clothing_or_accessories(
            self,
            item_type: Annotated[
                str,
                llm.TypeInfo(
                    description="Type of clothing item to evaluate (e.g., 'top', 'pants', 'dress')"
                ),
            ] = None,
        ):
            """Find similar clothing items based on the latest image captured. Returns suggestions for similar items that the user might like."""
            logger.info(f"Evaluating {item_type if item_type else 'detected item'}")
            # Get the latest image from the video feed
            latest_image = await get_latest_image(ctx.room)
            if not latest_image:
                logger.info(
                    "I couldn't capture an image to analyze. Please make sure your camera is working properly."
                )
                return "I couldn't capture an image to analyze. Please make sure your camera is working properly."

            # Encode the image to JPEG format
            image_bytes = encode(
                latest_image,
                EncodeOptions(
                    format="JPEG",
                    resize_options=ResizeOptions(
                        width=512, height=512, strategy="scale_aspect_fit"
                    ),
                ),
            )

            # Search for similar clothing items using Lykdat API
            url = "https://cloudapi.lykdat.com/v1/global/search"
            payload = {
                "api_key": os.getenv("LYKDAT_API_KEY"),
                # "catalog_name": "global",
            }

            # Use the latest captured image
            files = [("image", ("image.jpg", io.BytesIO(image_bytes), "image/jpeg"))]

            try:
                response = requests.post(url, data=payload, files=files)
                similar_items_data = response.json()

                # Extract result groups and similar products from the response
                result_groups = similar_items_data.get("data", {}).get(
                    "result_groups", []
                )
                similar_items = []

                # Process each detected clothing item and its similar products
                for group in result_groups:
                    detected_item = group.get("detected_item", {})
                    detected_type = detected_item.get("name", "Item")

                    # If a specific item type was requested, only include matching items
                    if item_type and item_type.lower() != detected_type.lower():
                        continue

                    # Get top similar products for this detected item
                    products = group.get("similar_products", [])[
                        :5
                    ]  # Limit to top 5 items

                    for product in products:
                        similar_items.append(
                            {
                                "item_type": detected_type,
                                "name": product.get("name", "Unknown"),
                                "brand": product.get("brand_name", "Unknown brand"),
                                "price": product.get("price", "N/A"),
                                "image_url": product.get("matching_image", ""),
                                "product_url": product.get("url", ""),
                                "score": product.get("score", 0),
                            }
                        )

                logger.info(f"Found {len(similar_items)} similar clothing items")

                # Format the results as a response
                if similar_items:
                    # Send email with the similar items
                    try:
                        # Format the email content
                        email_content = self._format_email_with_similar_items(
                            similar_items, item_type
                        )
                        self._send_email(
                            "eldar@atarino.io",
                            "Similar Clothing Items Found",
                            email_content,
                        )
                        result = f"I found {len(similar_items)} similar items you might like. I've sent the details to your email."
                        return result
                    except Exception as email_error:
                        logger.error(f"Failed to send email: {email_error}")
                        result = "I found similar items but couldn't send the email. Here are some recommendations:\n\n"
                        for i, item in enumerate(similar_items[:5]):
                            result += f"{i+1}. {item['name']} by {item['brand']} - {item['price']}\n   {item['product_url']}\n\n"
                        return result
                else:
                    return "I couldn't find any similar clothing items that match."
            except Exception as e:
                logger.error(f"Failed to get similar items: {e}")
                return f"Sorry, I encountered an error while searching for similar items: {str(e)}"

    fnc_ctx = AssistantFnc()

    participant = await ctx.wait_for_participant()
    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        # flexibility to use any models
        stt=deepgram.STT(model="nova-2-general"),
        llm=google_llm,
        tts=rime_tts,
        # intial ChatContext with system prompt
        chat_ctx=initial_chat_ctx,
        before_llm_cb=before_llm_cb,
        # whether the agent can be interrupted
        allow_interruptions=True,
        # sensitivity of when to interrupt
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        # minimal silence duration to consider end of turn
        min_endpointing_delay=0.5,
        fnc_ctx=fnc_ctx,
    )

    logger.info(f"Agent connected to room: {ctx.room.name}")
    logger.info(f"Local participant identity: {ctx.room.local_participant.identity}")
    assistant.start(ctx.room)
    await assistant.say("What's up beautiful?")
    logger.info("starting agent")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
