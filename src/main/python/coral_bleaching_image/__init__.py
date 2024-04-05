import logging
from io import BytesIO
from random import random, randint
import numpy

from PIL import Image as PILImage

# import PIL

logger = logging.getLogger(__name__)


class BadImageException(Exception):
    pass


class ImageNotFoundException(Exception):
    pass


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


def pre_process_image(x, y, image_path):
    patch = load_image_and_crop(image_path, x, y, 256, 256, 8)
    patch_array = img_to_array(patch)

    expanded_patch_array = numpy.expand_dims(patch_array, axis=0)
    del patch
    del patch_array
    if expanded_patch_array.shape == (1, 256, 256, 3):
        return expanded_patch_array
    else:
        logger.error("Shape is %s", expanded_patch_array.shape)
        raise BadImageException(image_path)


def pre_process_point(point_dict):
    return pre_process_image(
        point_dict["x"], point_dict["y"], point_dict["image_path"]
    )


def load_image_and_crop(
        image_path, point_x, point_y, crop_width=256, crop_height=256, cut_divisor=8
):
    try:
        img = PILImage.open(image_path)
    except BadImageException as e:
        logging.warning("Error loading image from bucket")
        raise ImageNotFoundException(image_path) from e
    width, height = img.size
    cut_width = int(height / cut_divisor)
    cut_height = int(height / cut_divisor)

    patch = cut_patch(
        img, cut_width, cut_height, point_x, point_y
    )
    resized_patch = patch.resize((crop_width, crop_height), PILImage.NEAREST)
    del img
    del patch
    return resized_patch


def img_to_array(
        pil_image: PILImage, data_format="channels_last", dtype="float32"
):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format: %s" % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = numpy.asarray(pil_image, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError("Unsupported image shape: %s" % (x.shape,))
    return x


def cut_patch(pil_image: PILImage, patch_width, patch_height, x, y):
    dimensions = get_rect_dimensions_pixels(
        patch_width, patch_height, x, y
    )
    new_image = pil_image.crop(dimensions)
    return new_image


def get_rect_dimensions_pixels(patchwidth, patchheight, pointx, pointy):
    return [
        int((pointx) - (patchwidth / 2)),
        int((pointy) - (patchheight / 2)),
        int((pointx) + (patchwidth / 2)),
        int((pointy) + (patchheight / 2)),
    ]
