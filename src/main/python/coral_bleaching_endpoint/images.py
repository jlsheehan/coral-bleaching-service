import logging
import os
# import shutil
import uuid
from io import BytesIO
from typing import Annotated, List, Optional

from PIL import Image as PILImage
from coral_bleaching_common import Image, Point, Segment
from coral_bleaching_db import ImageRepository, get_image_repo, PointRepository, get_point_repo, SegmentRepository, \
    get_segment_repo
from fastapi import APIRouter, UploadFile, Depends, Form, File, Response

from coral_bleaching_image import add_mask

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/images",
)
def create_image(
        image_file: Annotated[UploadFile, File()],
        survey_id: Annotated[str, Form()],
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
        with PILImage.open(image_file_path) as image:
            width, height = image.size
            return Image(
                id=(image_id or str(uuid.uuid4())),
                image_name=image_name,
                survey_id=survey_id,
                width=width,
                height=height,
            )

@router.get(
    "/images/{image_id}",
)
def get_image(
        image_id: str,
        image_repo: ImageRepository = Depends(get_image_repo),
) -> Image:
    image: Image = image_repo.find(image_id)
    return image


@router.get(
    "/images/{image_id}/jpeg",
)
def get_image_jpeg(
        image_id: str,
        image_repo: ImageRepository = Depends(get_image_repo),
) -> Response:
    image: Image = image_repo.find(image_id)
    with image_repo.storage() as storage:
        image_bytes: BytesIO = storage.import_bytes_from_storage(image)
        return Response(image_bytes.getvalue(), media_type="image/jpeg")


@router.get(
    "/images/{image_id}/points",
)
def get_image_points(
        image_id: str,
        point_repository: PointRepository = Depends(get_point_repo),
) -> List[Point]:
    points: List[Point] = point_repository.find_image_points(image_id)
    return points


@router.get("/images/{image_id}/mask")
def get_segment_mask2(
        image_id: str,
        include_image: Optional[str] = "false",
        segment_repository: SegmentRepository = Depends(get_segment_repo),
        image_repository: ImageRepository = Depends(get_image_repo),
):
    image: Image = image_repository.find(image_id)
    segments: List[Segment] = segment_repository.find_image_segments(image_id)
    with image_repository.storage() as image_store:
        image_file = image_store.import_from_storage(image)
        # shutil.copy(image_file, "C:\\Users\\jlsheeha\\Pictures\\original.jpg")
        with PILImage.open(image_file) as pil_image:
            logger.debug(f"Image size: {pil_image}")
            masked_pil_image = pil_image.copy()
            for segment in segments:
                with segment_repository.storage() as segment_store:
                    coords = segment_store.import_from_storage(segment)
                    masked_pil_image = add_mask(masked_pil_image, coords)
            masked_pil_image_bytes = BytesIO()
            masked_pil_image.thumbnail((1000, 1000))
            masked_pil_image.save(masked_pil_image_bytes, format="PNG")
            return Response(masked_pil_image_bytes.getvalue(), media_type="image/png")


@router.get("/images/{image_id}/segments")
def get_image_segments(
        image_id: str,
        segment_repo: SegmentRepository = Depends(get_segment_repo),
):
    logger.debug("Getting segments for image: %s", image_id)
    segments: List[Segment] = segment_repo.find_image_segments(image_id)
    return segments
