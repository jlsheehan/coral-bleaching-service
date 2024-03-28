import logging
import os
import shutil
import uuid
from datetime import datetime
from io import BytesIO
from json import JSONEncoder
from typing import Annotated, List, Optional

import PIL.Image
import cv2
# from coral_bleaching_segmentation import SegmentAnythingClassifier, Segment
import matplotlib.pyplot as plt
import numpy
import numpy as np
from coral_bleaching_common import (
    Image,
    Segment, Survey,
)
from coral_bleaching_db import (
    get_segment_repo,
    SegmentRepository,
    get_image_repo,
    ImageRepository, get_survey_repo, SurveyRepository,
)
from fastapi import FastAPI, UploadFile, Depends, status, Form, File
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from mangum import Mangum

from coral_bleaching_image import add_mask

API_KEYS = [
    "af2e03df7efe4bb198d08b75f277f377",
    "c17ee96ee97e49669561c0db7599a281",
    "ffc6541942f44e89a742f315c19eb51e",
]
TRUE_STRINGS = ["true", "TRUE", "True", "yes", "YES", "Yes", "1"]
CORAL_BLEACHING_DATA_BUCKET = os.getenv(
    "CORAL_BLEACHING_BUCKET", "coral-bleaching-data"
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
api_key_header = APIKeyHeader(name="x-api-key")


def valid_api_key(api_key: str = Depends(api_key_header)):
    if api_key in API_KEYS:
        return True
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Segment):
            return obj.dict()
        else:
            return JSONEncoder.default(self, obj)


# class SegmentAnythingListResponse(JSONResponse):
#
#     def render(self, segments: List[Segment]) -> bytes:
#         return json.dumps(
#             [s.dict(exclude={"coords"}) for s in segments],
#             cls=NumpyArrayEncoder,
#         ).encode("utf-8")
#
#
# class SegmentAnythingSegmentResponse(JSONResponse):
#     def render(self, segment: Segment) -> bytes:
#         return json.dumps(segment.dict(), cls=NumpyArrayEncoder).encode("utf-8")
#

# @app.on_event("startup")
# async def startup_event():
#     if not os.path.exists(os.path.join(SAM_MODEL_FILE)):
#         global sam
#         sam_model_key = f"models/sam/{SAM_MODEL_FILE}"
#         logger.info(
#             "Downloading model from %s:%s", CORAL_BLEACHING_DATA_BUCKET, sam_model_key
#         )
#         s3: S3Client = boto3.client("s3", region_name="ap-southeast-2")
#         s3.download_file(
#             Bucket=CORAL_BLEACHING_DATA_BUCKET,
#             Key=sam_model_key,
#             Filename=SAM_MODEL_FILE,
#         )
#     else:
#         logger.debug("SAM model already present at: %s", SAM_MODEL_FILE)
#     sam = SegmentAnythingClassifier.load_sam_model(SAM_MODEL_FILE)


# @app.on_event("shutdown")
# async def shutdown_event():
#     sam.close()


@app.post(
    "/images",
)
def create_image(
        image_file: Annotated[UploadFile, File()],
        survey_id: Annotated[str, Form()],
        valid_api_key: str = Depends(valid_api_key),
        image_repo: ImageRepository = Depends(get_image_repo),
) -> Image:
    logger.debug("Got file: %s", image_file.filename)
    with image_repo.storage() as storage:
        image_temp_file = os.path.join(storage.temp_dir, image_file.filename)
        with open(image_temp_file, "wb") as t:
            t.write(image_file.file.read())
        image: Image = from_image_file(image_temp_file, survey_id=survey_id)
        image_repo.save(image)
        storage.export_to_storage(image, image_temp_file)
        return image


def from_image_file(image_file_path, survey_id, image_id=None):
    if os.path.exists(image_file_path):
        image_name = os.path.basename(image_file_path)
        with PIL.Image.open(image_file_path) as image:
            width, height = image.size
            return Image(
                id=(image_id or str(uuid.uuid4())),
                image_name=image_name,
                survey_id=survey_id,
                width=width,
                height=height,
            )

@app.get(
    "/images/{image_id}",
)
def get_image(
        image_id: str,
        valid_api_key: str = Depends(valid_api_key),
        image_repo: ImageRepository = Depends(get_image_repo),
) -> Image:
    image: Image = image_repo.find(image_id)
    return image


@app.get(
    "/images/{image_id}/jpeg",
)
def get_image_file(
        image_id: str,
        valid_api_key: str = Depends(valid_api_key),
        image_repo: ImageRepository = Depends(get_image_repo),
) -> Image:
    image: Image = image_repo.find(image_id)
    image_repo.download_file(image, image.file_name)
    with image_repo.storage() as storage:
        image_bytes: BytesIO = storage.import_bytes_from_storage(image)
        return Response(image_bytes.getvalue(), media_type="image/jpg")


# @app.post("/work")
# def create_work_item(
#         work_request: WorkItemRequest,
#         valid_api_key: str = Depends(valid_api_key),
#         image_repo: ImageRepository = Depends(get_image_repo),
#         segment_repo: SegmentRepository = Depends(get_segment_repo),
#         segment_classifier: SegmentAnythingClassifier = Depends(get_sam),
# ):
#     logger.debug("Got work request: %s", work_request)
#     work_item = WorkItem.from_request(work_request)
#     if work_item.work_type == WorkItemType.SEGMENT_IMAGE:
#         work_item.start_time = datetime.now()
#         work_item.status = WorkItemStatus.RUNNING
#         image: Image = image_repo.find(work_item.work_data["image_id"])
#         with image_repo.storage() as image_store:
#             image_data = image_store.import_bytes_from_storage(image)
#             bytes_as_np_array = np.frombuffer(image_data.read(), dtype=np.uint8)
#             cv2_image = cv2.imdecode(bytes_as_np_array, cv2.IMREAD_COLOR)
#
#             cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
#         with segment_repo.storage() as segment_store:
#             logger.debug("Created image")
#             segments, coords_list = segment_classifier.segment_image(
#                 cv2_image, image.id
#             )
#             for segment, coords in zip(segments, coords_list):
#                 segment_repo.save(segment)
#                 segment_store.export_to_storage(segment, coords)
#         work_item.work_data["segment_ids"] = [s.id for s in segments]
#         work_item.status = WorkItemStatus.COMPLETED
#         work_item.end_time = datetime.now()
#     else:
#         work_item.status = WorkItemStatus.FAILED
#     return work_item


# @app.get("/work/{work_item_id}")
# def get_job(work_item_id: str, valid_api_key: str = Depends(valid_api_key)):
#     if work_item_id in work_repo:
#         return work_repo[work_item_id]
#     else:
#         return Response(status_code=status.HTTP_404_NOT_FOUND)


@app.get("/segments/{segment_id}/maskold")
def get_segment_mask(
        segment_id: str,
        include_image: Optional[str] = "false",
        valid_api_key: str = Depends(valid_api_key),
        segment_repository: SegmentRepository = Depends(get_segment_repo),
        image_repository: ImageRepository = Depends(get_image_repo),
):
    segment: Segment = segment_repository.find(segment_id)
    image: Image = image_repository.find(segment.image_id)
    with image_repository.storage() as image_store:
        image_file = image_store.import_from_storage(image)
        # image_bytes = image_store.import_bytes_from_storage(image)
        # cv2_image = cv2.imdecode(np.frombuffer(image_bytes.getvalue()))
        cv2_image = cv2.imread(image_file)
    with segment_repository.storage() as segment_store:
        coords = segment_store.import_from_storage(segment)
        mask_array = numpy.zeros((image.width, image.height), dtype=int)
        mask_array[coords] = 1

    # if im is None:
    #     print("No image provided, only showing mask")
    # else:
    #     plt.figure()
    #     plt.imshow(im)
    plt.figure()
    plt.imshow(cv2_image)
    logger.debug("AAAAAAAAAAAAAAAAAAAAAAA")
    # sorted_mask_array = sorted(mask_array, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    logger.debug("BBBBBBBBBBBBBBBBBBBBBBBBB")
    img = np.ones((image.width, image.height, 4))
    img[:, :, 3] = 0
    color_mask = np.concatenate([np.random.random(3), [0.35]])
    logger.debug("CCCCCC, %s", datetime.now())
    img[mask_array] = color_mask
    logger.debug("DDDDDDDDDDDDDD, %s", datetime.now())
    ax.imshow(img)
    # plt.show()
    plt.axis("off")
    # plt.show()
    png_bytes = BytesIO()
    plt.savefig(png_bytes, format="png")
    png_bytes.seek(0)
    # with open("blah.png", "rb") as blah:
    #     png_bytes.write(blah.read())
    logger.debug("============================")
    return Response(png_bytes.getvalue(), media_type="image/png")


@app.get("/segments/{segment_id}/mask")
def get_segment_mask2(
        segment_id: str,
        include_image: Optional[str] = "false",
        valid_api_key: str = Depends(valid_api_key),
        segment_repository: SegmentRepository = Depends(get_segment_repo),
        image_repository: ImageRepository = Depends(get_image_repo),
):
    segment: Segment = segment_repository.find(segment_id)
    image: Image = image_repository.find(segment.image_id)
    with image_repository.storage() as image_store:
        image_file = image_store.import_from_storage(image)
        shutil.copy(image_file, "C:\\Users\\jlsheeha\\Pictures\\original.jpg")
        with PIL.Image.open(image_file) as pil_image:
            logger.debug(f"Image size: {pil_image}")
            with segment_repository.storage() as segment_store:
                coords = segment_store.import_from_storage(segment)
                masked_pil_image = add_mask(pil_image, coords)
                masked_pil_image_bytes = BytesIO()
                masked_pil_image.save(masked_pil_image_bytes, format="PNG")
                return Response(masked_pil_image_bytes.getvalue(), media_type="image/png")


@app.get("/images/{image_id}/mask")
def get_segment_mask2(
        image_id: str,
        include_image: Optional[str] = "false",
        valid_api_key: str = Depends(valid_api_key),
        segment_repository: SegmentRepository = Depends(get_segment_repo),
        image_repository: ImageRepository = Depends(get_image_repo),
):
    image: Image = image_repository.find(image_id)
    segments: List[Segment] = segment_repository.find_image_segments(image_id)
    with image_repository.storage() as image_store:
        image_file = image_store.import_from_storage(image)
        shutil.copy(image_file, "C:\\Users\\jlsheeha\\Pictures\\original.jpg")
        with PIL.Image.open(image_file) as pil_image:
            logger.debug(f"Image size: {pil_image}")
            masked_pil_image = pil_image.copy()
            for segment in segments:
                with segment_repository.storage() as segment_store:
                    coords = segment_store.import_from_storage(segment)
                    masked_pil_image = add_mask(masked_pil_image, coords)
            masked_pil_image_bytes = BytesIO()
            masked_pil_image.save(masked_pil_image_bytes, format="PNG")
            return Response(masked_pil_image_bytes.getvalue(), media_type="image/png")

            # mask_array = numpy.zeros((image.width, image.height), dtype=int)
            # mask_array[coords] = 1
            # mask_image = PIL.Image.fromarray(mask_array, mode='LA')
            # mask_image.save("C:\\Users\\jlsheeha\\Pictures\\mask.png")
            # mask_image = mask_image.convert('RGBA')
            # logger.debug(f"Mask Image size: {mask_image}")
            # blank = pil_image.point(lambda _: 0)
            # composite_image = PIL.Image.composite(pil_image, blank, mask_image)
            # # composite_image = PIL.Image.blend(pil_image, mask_image, 0.5)
            # composite_bytes = BytesIO()
            # composite_image.save(composite_bytes, format="PNG")
            # # composite_image = PIL.Image.composite(pil_image, mask_image, None)
            # return Response(composite_bytes.getvalue(), media_type="image/png")

    # mask_image = np.ones((image.width, image.height, 4))
    # mask_image[:, :, 3] = 0
    # color_mask = np.concatenate([np.random.random(3), [0.35]])
    # logger.debug("CCCCCC")
    # mask_image[mask_array] = color_mask
    #
    # masked_image = cv2.addWeighted(cv2_image, 0.5, mask_image, 0.5, 0)
    # _, png_image = cv2.imencode(".png", masked_image)
    # return Response(png_image, media_type="image/png")


# @app.post(
#     "/segment",
# )
# def segment(
#     image_file: Annotated[UploadFile, File()],
#     survey_id: Annotated[str, Form()],
#     valid_api_key: str = Depends(valid_api_key),
#     segment_repo: SegmentRepository = Depends(get_segment_repo),
#     image_repo: ImageRepository = Depends(get_image_repo),
#     segment_classifier: SegmentAnythingClassifier = Depends(get_sam),
# ) -> List[Segment]:
#     logger.debug("Got file: %s", image_file.filename)
#     with image_repo.storage() as image_store:
#         image_temp_file = os.path.join(image_store.temp_dir, image_file.filename)
#         with open(image_temp_file, "wb") as t:
#             t.write(image_file.file.read())
#         image: Image = Image.from_image_file(image_temp_file, survey_id=survey_id)
#         image_repo.save(image)
#         image_store.export_to_storage(image, image_temp_file)
#
#         image = cv2.imread(temp_file)
#
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         logger.debug("Created image")
#         segments, coords_list = sam.segment_image(image, image_file.filename)
#         segment_repo.save(segments)
#
#         return SegmentAnythingListResponse(segments)


@app.get("/segments/{segment_id}")
def get_segment(
        segment_id: str,
        valid_api_key: str = Depends(valid_api_key),
        segment_repo: SegmentRepository = Depends(get_segment_repo),
):
    logger.debug("Getting segment: %s", segment_id)
    segment = segment_repo.find(segment_id)
    return segment


@app.get("/images/{image_id}/segments")
def get_segments(
        image_id: str,
        valid_api_key: str = Depends(valid_api_key),
        segment_repo: SegmentRepository = Depends(get_segment_repo),
):
    logger.debug("Getting segments for image: %s", image_id)
    segments: List[Segment] = segment_repo.find_image_segments(image_id)
    return segments

handler = Mangum(app)

# @app.get("/images/{image_name}/sample-points")
# def sample_points(
#         image_name: str,
#         points_per_segment: Optional[int] = 10,
#         valid_api_key: str = Depends(valid_api_key),
#         segment_repo: SegmentRepository = Depends(get_segment_repo),
# ):
#     segments = segment_repo.find_image_segments(image_name)
#     points: List[Point] = sample_segments(
#         segments, points_per_segment=points_per_segment
#     )
#     return points


@app.get("/images/{image_id}/segments")
def sample_points(
        image_id: str,
        valid_api_key: str = Depends(valid_api_key),
        segment_repo: SegmentRepository = Depends(get_segment_repo),
):
    logger.debug("Searching for segments for image: %s", image_id)
    segments = segment_repo.find_image_segments(image_id)
    return segments


@app.get("/surveys/{survey_id}")
def get_segment(
        survey_id: str,
        valid_api_key: str = Depends(valid_api_key),
        survey_repo: SurveyRepository = Depends(get_survey_repo),
):
    logger.debug("Getting survey: %s", survey_id)
    survey: Survey = survey_repo.find(survey_id)
    return survey


@app.post("/surveys")
def create_survey(
        survey_request: dict,
        valid_api_key: str = Depends(valid_api_key),
        survey_repo: SurveyRepository = Depends(get_survey_repo),
):
    survey_data = survey_request.copy()
    if "survey_date" not in survey_data:
        survey_data["survey_date"] = datetime.now()
    if "survey_name" not in survey_data:
        survey_data["survey_name"] = f"Survey {survey_data['project_name']} on {survey_data['survey_date']}"
    survey: Survey = Survey(id = str(uuid.uuid4()), **survey_data)
    survey_repo.save(survey)
    return survey


@app.get("/healthcheck")
def healthcheck():
    return {"status": True}
