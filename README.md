# iiif_image_load
A python library for loading iiif images (IIIF Image Api Manifests).

## How to Install

To install, just run `pip install iiif_image_load`.

## About

This library makes it possible to submit the URL to a IIIF Image API manifest (along with optional cropping parameters) and receive a standard OpenCV Numpy (3D BGR) array containing the image.

The downloading process will access image tiles in accordance with the IIIF Image server's specifications. This makes download of image at full resolution possible on all IIIF compliant image servers and will potentially speed up the download process (and relieve stress on the IIIF image server).

## Usage

To acquire an image with this library, the following examples should suffice:

```python
from iiif_image_load import iiif_image_from_url
import cv2

img = iiif_image_from_url('https://ids.lib.harvard.edu/ids/iiif/47174896')

cv2.imshow('Demo Paul Gauguin', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

It is also possible to specify a crop box for the image:

```python
from iiif_image_load import iiif_image_from_url
from matplotlib.pyplot import imshow

img = iiif_image_from_url('https://ids.lib.harvard.edu/ids/iiif/47174896', 10, 20, 500, 500)

imshow(img[...,::-1])
```