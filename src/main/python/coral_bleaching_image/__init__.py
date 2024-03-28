import logging
from io import BytesIO
from random import random, randint

from PIL.Image import Image as PILImage

# import PIL

logger = logging.getLogger(__name__)


def add_mask(pil_image: PILImage, mask_coords):
    logger.debug("IM is %s", pil_image.im)
    if pil_image.mode != 'RGBA':
        pil_image_copy = pil_image.convert("RGBA")
    else:
        pil_image_copy = pil_image.copy()
    colour_inc = randint(0, 255)
    colour_field = randint(0, 2)
    for x, y in mask_coords:
        pixel = list(pil_image.getpixel((y, x)))
        pixel[colour_field] = pixel[colour_field] + colour_inc
        pil_image_copy.putpixel((y, x), tuple(pixel))
    return pil_image_copy


def add_masks(pil_image: PILImage, mask_coords_list):
    pil_image_copy = pil_image.convert("RGBA")
    for mask_coords in mask_coords_list:
        pil_image_copy = add_mask(pil_image_copy, mask_coords)
    return pil_image_copy
