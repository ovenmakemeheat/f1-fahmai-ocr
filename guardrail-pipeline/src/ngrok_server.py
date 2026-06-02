from __future__ import annotations

import os
import socket
import threading
import time

import uvicorn
from dotenv import load_dotenv
from pyngrok import conf, ngrok


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def wait_for_server(server: uvicorn.Server, host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: OSError | None = None

    while time.monotonic() < deadline:
        if server.should_exit:
            raise RuntimeError("FastAPI server stopped before startup completed.")

        try:
            with socket.create_connection((host, port), timeout=1.0):
                if server.started:
                    return
        except OSError as exc:
            last_error = exc

        time.sleep(0.25)

    raise RuntimeError(
        f"Timed out waiting for FastAPI startup at {host}:{port} after "
        f"{timeout_seconds:.0f}s. Last socket error: {last_error}"
    )


def main() -> None:
    load_dotenv()

    host = os.getenv("HOST", "127.0.0.1")
    port = get_int_env("PORT", 8000)
    region = os.getenv("NGROK_REGION", "ap")
    domain = os.getenv("NGROK_DOMAIN") or None
    authtoken = os.getenv("NGROK_AUTHTOKEN")
    startup_timeout = float(os.getenv("STARTUP_TIMEOUT_SECONDS", "180"))

    if not authtoken:
        raise RuntimeError("NGROK_AUTHTOKEN is required. Set it in .env or the environment.")

    conf.get_default().auth_token = authtoken
    conf.get_default().region = region

    config = uvicorn.Config("src.app:app", host=host, port=port, reload=False)
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)

    print(f"Starting local FastAPI server at http://{host}:{port} ...")
    server_thread.start()
    wait_for_server(server, host, port, startup_timeout)
    print("FastAPI startup complete.")

    local_url = f"http://{host}:{port}"
    tunnel_kwargs: dict[str, str] = {"proto": "http", "addr": local_url}
    if domain:
        tunnel_kwargs["domain"] = domain

    print("Starting ngrok tunnel ...")
    tunnel = ngrok.connect(**tunnel_kwargs)
    print(f"ngrok tunnel: {tunnel.public_url}")
    print(f"local server: {local_url}")
    print(f"API docs: {tunnel.public_url}/docs")

    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True
    finally:
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        server.should_exit = True
        server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
