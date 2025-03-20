import asyncio
import websockets

async def send_command():
    uri = "ws://localhost:8000/events"
    async with websockets.connect(uri) as websocket:
        await websocket.send("right")
        print("Sent command: right")
        await asyncio.sleep(2)

asyncio.run(send_command())