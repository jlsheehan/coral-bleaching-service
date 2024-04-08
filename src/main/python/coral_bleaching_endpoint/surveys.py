import logging
import uuid
from datetime import datetime
from typing import List

from coral_bleaching_common import Image, Survey
from coral_bleaching_db import ImageRepository, get_image_repo, SurveyRepository, get_survey_repo
from fastapi import APIRouter, Depends

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/surveys/{survey_id}/images",
)
def get_survey_images(
        survey_id: str,
        image_repo: ImageRepository = Depends(get_image_repo),
) -> List[Image]:
    images: List[Image] = image_repo.find_survey_images(survey_id)
    return images


@router.get(
    "/projects/{project_name}/surveys",
)
def get_project_surveys(
        project_name: str,
        survey_repository: SurveyRepository = Depends(get_survey_repo),
) -> List[Survey]:
    surveys: List[Survey] = survey_repository.find_project_surveys(project_name)
    return surveys


@router.get(
    "/surveys/{survey_id}",
)
def get_survey(
        survey_id: str,
        survey_repo: SurveyRepository = Depends(get_survey_repo),
) -> Survey:
    survey: Survey = survey_repo.find(survey_id)
    return survey

@router.get(
    "/surveys/{survey_id}/images",
)
def get_images_for_survey(
        survey_id: str,
        image_repo: ImageRepository = Depends(get_image_repo),
) -> List[Image]:
    images: List[Image] = image_repo.find_survey_images(survey_id)
    return images


@router.get("/surveys/{survey_id}")
def get_segment(
        survey_id: str,
        survey_repo: SurveyRepository = Depends(get_survey_repo),
):
    logger.debug("Getting survey: %s", survey_id)
    survey: Survey = survey_repo.find(survey_id)
    return survey


@router.post("/surveys")
def create_survey(
        survey_request: dict,
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
