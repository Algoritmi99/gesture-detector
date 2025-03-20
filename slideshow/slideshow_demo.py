from sanic import Sanic
from sanic.response import html
import pathlib

app = Sanic("slideshow_server")

slideshow_root_path = pathlib.Path(__file__).parent.joinpath("slideshow")

app.static("/static", slideshow_root_path)

@app.route("/")
async def index(request):
    return html(open(slideshow_root_path.joinpath("slideshow.html"), "r").read())

connected_clients = set()

@app.websocket("/events")
async def event_listener(_request, ws):
    print("WebSocket connection opened")
    connected_clients.add(ws)

    try:
        while True:
            message = await ws.recv()
            print(f"Received command from client: {message}")

            for client in connected_clients:
                if client != ws:
                    await client.send(message)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connected_clients.remove(ws)

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
