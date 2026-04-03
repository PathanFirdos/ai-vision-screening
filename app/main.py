from fastapi import FastAPI

app = FastAPI(title="AI Vision Screening API")

@app.get("/")
def home():
    return {"message": "AI Vision Screening Backend Running"}