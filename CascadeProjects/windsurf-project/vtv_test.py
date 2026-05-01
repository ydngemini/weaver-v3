"""Minimal Live API test — mirrors official Google cookbook exactly, using arecord/aplay."""

import asyncio
import os

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
CHUNK = 1024

CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful voice assistant. Greet the user when they speak.",
    "proactivity": {"proactive_audio": True},
}


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY")

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    arecord = await asyncio.create_subprocess_exec(
        "arecord", "-D", "default", "-q",
        "-f", "S16_LE", "-c", "1", "-r", "16000", "-t", "raw", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )

    aplay = await asyncio.create_subprocess_exec(
        "aplay", "-f", "S16_LE", "-c", "1", "-r", "24000", "-q", "-",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )

    audio_out_queue = asyncio.Queue()
    send_queue = asyncio.Queue(maxsize=5)

    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:

        async def listen_mic():
            while True:
                data = await arecord.stdout.read(CHUNK)
                if not data:
                    break
                await send_queue.put({"data": data, "mime_type": "audio/pcm"})

        async def send_audio():
            while True:
                msg = await send_queue.get()
                await session.send_realtime_input(audio=msg)

        async def receive_audio():
            while True:
                turn = session.receive()
                async for response in turn:
                    if data := response.data:
                        audio_out_queue.put_nowait(data)
                        print("🔊", end="", flush=True)
                        continue
                    if text := response.text:
                        print(f"\n[TEXT]: {text}", end="", flush=True)
                # turn complete — flush playback queue on interruption
                while not audio_out_queue.empty():
                    audio_out_queue.get_nowait()
                print("\n[TURN COMPLETE]", flush=True)

        async def play_audio():
            while True:
                data = await audio_out_queue.get()
                aplay.stdin.write(data)
                await aplay.stdin.drain()

        print("[READY] Speak into the mic…")
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(listen_mic())
                tg.create_task(send_audio())
                tg.create_task(receive_audio())
                tg.create_task(play_audio())
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        os.system("pkill -f arecord")
        os.system("pkill -f aplay")
