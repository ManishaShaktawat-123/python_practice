from fastapi import FastAPI 

app = FastAPI(title="ai integration api - day4")

@app.get("/")
def home():
    return {"status": "online", "message": "welcome to the ai integration api - day4"}

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "fastapi server"}