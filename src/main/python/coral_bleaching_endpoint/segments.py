import logging
from io import BytesIO
from typing import Optional

from PIL import Image as PILImage
from coral_bleaching_common import Image, Segment
from coral_bleaching_db import ImageRepository, get_image_repo, SegmentRepository, \
    get_segment_repo
from fastapi import APIRouter, Depends, Response
from fastapi.responses import RedirectResponse

from coral_bleaching_image import add_mask

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/segments/{segment_id}/mask")
def get_segment_mask2(
        segment_id: str,
        include_image: Optional[str] = "false",
        segment_repository: SegmentRepository = Depends(get_segment_repo),
        image_repository: ImageRepository = Depends(get_image_repo),
):
    segment: Segment = segment_repository.find(segment_id)
    image: Image = image_repository.find(segment.image_id)
    with image_repository.storage() as image_store:
        image_file = image_store.import_from_storage(image)
        # shutil.copy(image_file, "C:\\Users\\jlsheeha\\Pictures\\original.jpg")
        with PILImage.open(image_file) as pil_image:
            logger.debug(f"Image size: {pil_image}")
            with segment_repository.storage() as segment_store:
                coords = segment_store.import_from_storage(segment)
                masked_pil_image = add_mask(pil_image, coords)
                masked_pil_image.thumbnail((1000, 1000))
                masked_pil_image_bytes = BytesIO()
                masked_pil_image.save(masked_pil_image_bytes, format="PNG")
                return Response(masked_pil_image_bytes.getvalue(), media_type="image/png")


@router.get("/segments/{segment_id}")
def get_segment(
        segment_id: str,
        segment_repo: SegmentRepository = Depends(get_segment_repo),
):
    logger.debug("Getting segment: %s", segment_id)
    segment = segment_repo.find(segment_id)
    return segment


@router.get("/segments/{segment_id}/coords")
def get_segment_coords(
        segment_id: str,
        segment_repo: SegmentRepository = Depends(get_segment_repo),
) -> Response:
    logger.debug("Getting segment: %s", segment_id)
    segment: Segment = segment_repo.find(segment_id)
    return RedirectResponse(segment_repo.download_url(segment))
