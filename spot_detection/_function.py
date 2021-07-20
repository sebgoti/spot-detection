"""
This module is an example of a barebones function plugin for napari

It implements the ``napari_experimental_provide_function`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

import numpy as np
from napari_plugin_engine import napari_hook_implementation
from scipy import ndimage as ndi
from skimage.feature import blob_log

if TYPE_CHECKING:
    import napari


# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
# we can return a single function
# or a tuple of (function, magicgui_options)
# or a list of multiple functions with or without options, as shown here:
    return detect_spots


def gaussian_high_pass(image: np.ndarray, sigma: float = 2):
    """Apply a gaussian high pass filter to an image.

    Parameters
    ----------
    image : np.ndarray
        The image to be filtered.
    sigma : float
        The sigma (width) of the gaussian filter to be applied.
        The default value is 2.

    Returns
    -------
    high_passed_im : np.ndarray
        The image with the high pass filter applied
    """
    low_pass = ndi.gaussian_filter(image, sigma)
    high_passed_im = image - low_pass

    return high_passed_im


def detect_spots(
    image: "napari.types.ImageData",
    high_pass_sigma: float = 2,
    spot_threshold: float = 0.01,
    blob_sigma: float = 2
) -> "napari.types.LayerDataTuple":
    """Apply a gaussian high pass filter to an image.

    Parameters
    ----------
    image : np.ndarray
        The image in which to detect the spots.
    high_pass_sigma : float
        The sigma (width) of the gaussian filter to be applied.
        The default value is 2.
    spot_threshold : float
        The threshold to be passed to the blob detector.
        The default value is 0.01.
    blob_sigma: float
        The expected sigma (width) of the spots. This parameter
        is passed to the "max_sigma" parameter of the blob
        detector.

    Returns
    -------
    points_coords : np.ndarray
        An NxD array with the coordinate for each detected spot.
        N is the number of spots and D is the number of dimensions.
    sizes : np.ndarray
        An array of size N, where N is the number of detected spots
        with the diameter of each spot.

    """
    # filter the image with the gaussian_high_pass filter
    filtered_spots = gaussian_high_pass(image, high_pass_sigma)

    # detect the spots on the filtered image
    blobs_log = blob_log(filtered_spots,
                         max_sigma=blob_sigma,
                         num_sigma=1,
                         threshold=spot_threshold
                         )

    # convert the output of the blob detector to the
    # desired points_coords and sizes arrays
    # (see the docstring for details)
    points_coords, sizes = blobs_log[:,:-1], 3 * blobs_log[:, -1]

    layer_data = (
        points_coords,
        {
            "face_color": "magenta",
            "size": sizes,
            "symbol": 'ring'
            },
        "Points"
            )
    return layer_data
