import logging
import os

# import cv2
# from coral_bleaching_segmentation import SegmentAnythingClassifier, Segment
# import matplotlib.pyplot as plt
from fastapi import FastAPI

import coral_bleaching_endpoint
from coral_bleaching_endpoint import images, points, segments, surveys, work

TRUE_STRINGS = ["true", "TRUE", "True", "yes", "YES", "Yes", "1"]
CORAL_BLEACHING_DATA_BUCKET = os.getenv(
    "CORAL_BLEACHING_BUCKET", "coral-bleaching-data-bucket"
)
DEBUG = os.getenv("DEBUG", "TRUE") == "TRUE"
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO

logging.basicConfig(level=LOG_LEVEL)
logging.getLogger().setLevel(LOG_LEVEL)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("s3transfer").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


app = FastAPI()
app.include_router(images.router)
app.include_router(points.router)
app.include_router(segments.router)
app.include_router(surveys.router)
app.include_router(work.router)

@app.get("/healthcheck")
def healthcheck():
    return {"status": True}
