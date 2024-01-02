import cv2
import numpy as np


def find_length_scale_bar_web(image, meter):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set the threshold to detect all white areas in the image
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # Detect all contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # We expect the scale bar to be near the bottom of the image
    lower_third = binary_image[-20:, :]
    # Detect all contours in the lower third of the image
    scale_bar_contours, _ = cv2.findContours(lower_third, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour that corresponds to the scale bar based on expected characteristics
    scale_bar_contours = sorted(scale_bar_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    _, _, w, h = cv2.boundingRect(scale_bar_contours[-1])
    if w > h * 4:  # This is a simple heuristic and may need adjustment
        return meter / w
    else:
        raise Exception('Error : Color bar not found')


def calc_mask_area(mask, label, pixel_meter_ratio):
    # Count the number of pixels that match the specified label
    pixel_count = np.sum(mask == label)

    # Calculate the real-world area
    area = pixel_count * pixel_meter_ratio ** 2

    return area
