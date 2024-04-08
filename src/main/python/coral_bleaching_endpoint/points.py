import logging
# import shutil
from io import BytesIO

from coral_bleaching_common import Image, Point
from coral_bleaching_db import ImageRepository, get_image_repo, PointRepository, get_point_repo
from fastapi import APIRouter, Depends, Response

from coral_bleaching_image import load_image_and_crop

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/points/{point_id}",
)
def get_survey(
        point_id: str,
        point_repo: PointRepository = Depends(get_point_repo),
) -> Point:
    point: Point = point_repo.find(point_id)
    return point


@router.get("/points/{point_id}/patch")
def get_patch(
        point_id: str,
        point_repository: PointRepository = Depends(get_point_repo),
        image_repository: ImageRepository = Depends(get_image_repo),
):
    point: Point = point_repository.find(point_id)
    image: Image = image_repository.find(point.image_id)
    with image_repository.storage() as image_store:
        image_file = image_store.import_from_storage(image)
        # shutil.copy(image_file, f"C:\\Users\\jlsheeha\\Pictures\\{image.image_name}.jpg")
        # with PIL.Image.open(image_file) as pil_image:
        #     logger.debug(f"Image size: {pil_image}")
        patch = load_image_and_crop(image_file, point.coordinate[0], point.coordinate[1])
        patch_bytes = BytesIO()
        patch.save(patch_bytes, format="JPEG")
        return Response(patch_bytes.getvalue(), media_type="image/jpeg")
