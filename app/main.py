from fastapi import FastAPI
from app.api.routes import router
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = RotatingFileHandler(
        "app.log",
        maxBytes=5_000_000,
        backupCount=2
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)


app = FastAPI()

app.include_router(router)

