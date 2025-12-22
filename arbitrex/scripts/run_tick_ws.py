import uvicorn
from arbitrex.stream import ws_server


if __name__ == "__main__":
    # Run on localhost:8000
    uvicorn.run("arbitrex.stream.ws_server:app", host="0.0.0.0", port=8000, reload=False)
