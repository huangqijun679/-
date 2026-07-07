import subprocess, sys

PORT = 8003

subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], capture_output=True)
print(f"Starting server at http://localhost:{PORT}")
sys.exit(subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(PORT), "--reload"]).returncode)
