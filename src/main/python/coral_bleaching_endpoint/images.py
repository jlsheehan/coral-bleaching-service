import logging
import uuid
from io import BytesIO
from typing import List, Optional

from PIL import Image as PILImage
from coral_bleaching_common import Image, Point, Segment
from coral_bleaching_db import ImageRepository, get_image_repo, PointRepository, get_point_repo, SegmentRepository, \
    get_segment_repo
from coral_bleaching_image import add_mask
from fastapi import APIRouter, Depends, Response, Request
from fastapi.responses import RedirectResponse, JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/images", response_model=Image)
def create_image(image: Image, image_repo: ImageRepository = Depends(get_image_repo)) -> Image:
    logger.debug("Received image %s", image)
    new_image = image.copy()
    if new_image.id is None:
        new_image.id = str(uuid.uuid4())
    image_repo.save(new_image)
    return new_image


@router.get(
    "/images/{image_id}",
)
def get_image(
        image_id: str,
        image_repo: ImageRepository = Depends(get_image_repo),
) -> Image:
    image: Image = image_repo.find(image_id)
    return image


@router.post(
    "/images/{image_id}/jpeg",
)
def upload_image_jpeg(
        image_id: str,
        image_repo: ImageRepository = Depends(get_image_repo),
) -> dict:
    image: Image = image_repo.find(image_id)
    logger.debug("Uploading image: %s", image.id)
    redirect_url: dict = image_repo.upload_url(image)
    logger.debug(redirect_url)
    return redirect_url


@router.get(
    "/images/{image_id}/jpeg",
)
def get_image_jpeg(
        image_id: str,
        image_repo: ImageRepository = Depends(get_image_repo),
) -> Response:
    image: Image = image_repo.find(image_id)
    return RedirectResponse(image_repo.download_url(image))


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
