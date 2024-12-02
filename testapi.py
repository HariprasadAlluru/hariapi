from flask import Flask

app = Flask(__name__)

@app.get("/")
def read_root():
    return {"message": "Hello, World! Welcome to FastAPI on AWS."}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
