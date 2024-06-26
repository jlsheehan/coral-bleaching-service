import json
import logging
import os

import boto3
from coral_bleaching_common import WorkItem, WorkItemRequest, WorkItemType
from fastapi import APIRouter
from mypy_boto3_sqs import SQSClient

logger = logging.getLogger(__name__)
router = APIRouter()

sqs: SQSClient = boto3.client("sqs", region_name="ap-southeast-2")
CLASSIFICATION_QUEUE_URL = os.getenv("CLASSIFICATION_QUEUE")
SEGMENTATION_QUEUE_URL = os.getenv("SEGMENTATION_QUEUE")

@router.post(
    "/work",
)
def submit_work(
        work_item_request: WorkItemRequest,
) -> WorkItem:
    work_item: WorkItem = WorkItem.from_request(work_item_request)
    if work_item.work_type in (WorkItemType.PROCESS_SURVEY, WorkItemType.PROCESS_IMAGE):
        queue_url = SEGMENTATION_QUEUE_URL
    else:
        queue_url = CLASSIFICATION_QUEUE_URL
    sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(work_item.dict()))
    return work_item
