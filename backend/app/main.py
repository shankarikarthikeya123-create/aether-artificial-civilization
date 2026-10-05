from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

app = FastAPI(
    title="The Digital Mind",
    description="AETHER — An Artificial Civilization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "project": "The Digital Mind",
        "civilization": "AETHER",
        "status": "online",
        "message": "The artificial civilization has been initialized.",
    }