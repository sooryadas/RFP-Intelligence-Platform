import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    processes = []

    try:
        api = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=PROJECT_DIR,
        )
        processes.append(("API", api))

        ui = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(PROJECT_DIR / "ui" / "app.py"),
                "--server.port",
                "8501",
            ],
            cwd=PROJECT_DIR,
        )
        processes.append(("UI", ui))

        print("RFP Intelligence Platform is starting.")
        print("API: http://127.0.0.1:8000")
        print("UI:  http://localhost:8501")
        print("Press Ctrl+C to stop both services.")

        while True:
            for name, process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"{name} stopped unexpectedly with exit code {exit_code}."
                    )
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping the RFP Intelligence Platform...")

    finally:
        for _, process in reversed(processes):
            stop_process(process)


if __name__ == "__main__":
    main()