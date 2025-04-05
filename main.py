from __future__ import annotations
from dotenv import load_dotenv
import os

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
    speaker="cove",
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

        latest_user_message = [m for m in chat_ctx.messages if m.role == "user"][-1]
        logger.info(f"latest user message: {latest_user_message.content}")
        if latest_image:
            image_content = [llm.ChatImage(image=latest_image)]
            chat_ctx.messages.append(
                llm.ChatMessage(role="user", content=image_content)
            )
            logger.debug("Added latest frame to conversation context")

        # Function to read PDF examples and add them to the chat context

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
    )

    logger.info(f"Agent connected to room: {ctx.room.name}")
    logger.info(f"Local participant identity: {ctx.room.local_participant.identity}")
    assistant.start(ctx.room)
    await assistant.say("What's up beautiful?")
    logger.info("starting agent")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
