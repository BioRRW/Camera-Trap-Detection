import argparse
import os
import json
import numpy as np
import cv2
from pathlib import Path


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirpath_input_images', type=str, required=True,
        help='Absolute directory path to images to detect the horizon on.'
    )
    parser.add_argument(
        '--dirpath_output_images', type=str, required=True,
        help='Absolute directory path to save output images in.'
    )
    parser.add_argument(
        '--fpath_output_coords', type=str, required=True,
        help='Absolute file path to save output coordinates in JSON format.'
    )

    args = parser.parse_args()
    return args


def detect_horizon_line(image_grayscaled):
    """Detect the horizon's starting and ending points in the given image.
    ...
    """
    
    msg = ('`image_grayscaled` should be a grayscale, 2-dimensional image '
             'of shape (height, width).')
    assert image_grayscaled.ndim == 2, msg
    image_blurred = cv2.GaussianBlur(image_grayscaled, ksize=(3, 3), sigmaX=0)

    _, image_thresholded = cv2.threshold(
        image_blurred, thresh=0, maxval=1,
        type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )
    image_thresholded = image_thresholded - 1
    image_closed = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE,
                                     kernel=np.ones((9, 9), np.uint8))
    
    horizon_x1 = 0
    horizon_x2 = image_grayscaled.shape[1] - 1
    
    # Check if a horizon line can be detected
    if np.all(image_closed == 1) or np.all(image_closed == 0):
        return None, None, None, None
    
    # Get the row indices where the thresholded image has a value of 0
    y_coords_x1 = np.where(image_closed[:, horizon_x1] == 0)[0]
    y_coords_x2 = np.where(image_closed[:, horizon_x2] == 0)[0]
    
    # Check if the arrays are empty
    if y_coords_x1.size == 0 or y_coords_x2.size == 0:
        return None, None, None, None
    
    horizon_y1 = max(y_coords_x1)
    horizon_y2 = max(y_coords_x2)

    return horizon_x1, horizon_x2, horizon_y1, horizon_y2


def main():
    """Main logic"""

    args = parse_args()

    dirpath_input_images = Path(args.dirpath_input_images)
    dirpath_output_images = Path(args.dirpath_output_images)
    fpath_output_coords = Path(args.fpath_output_coords)

    msg = ('`dirpath_input_images` and `dirpath_output_images` cannot point to'
           'the same directory.')
    assert dirpath_input_images != dirpath_output_images, msg
    os.makedirs(dirpath_output_images, exist_ok=True)
    os.makedirs(fpath_output_coords.parent, exist_ok=True)


    # Initialize a dictionary to store horizon coordinates
    horizon_coords_data = {}

    # Thresholds to skip horizon detection
    NIGHT_BRIGHTNESS_THRESHOLD = 60  # Adjust this value as needed
    SNOW_WHITE_PIXELS_THRESHOLD = 0.35 # 25% of pixels are "white"
    LOW_HORIZON_THRESHOLD_PERCENTAGE = 0.9

    # Recursively find all image files
    image_file_paths = []
    for dirpath, _, filenames in os.walk(dirpath_input_images):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_file_paths.append(Path(dirpath) / filename)

    for fpath_input in image_file_paths:
        # Create output path, preserving subdirectory structure
        relative_path = fpath_input.relative_to(dirpath_input_images)
        fpath_output_image = dirpath_output_images / relative_path

        # Create output subdirectory if it doesn't exist
        os.makedirs(fpath_output_image.parent, exist_ok=True)

        image_original = cv2.imread(str(fpath_input))

        if image_original is None:
            print(f"Warning: Failed to load image {fpath_input}. Skipping.")
            continue

        image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)

        # Calculate the average brightness and white pixel percentage
        average_brightness = np.mean(image_grayscale)
        white_pixels = np.sum(hsv[:, :, 2] > 200) # White pixels have a high V value
        white_pixel_percentage = white_pixels / (hsv.shape[0] * hsv.shape[1])

        # Conditional logic to skip horizon detection
        is_night_image = average_brightness < NIGHT_BRIGHTNESS_THRESHOLD
        if is_night_image:
            # Skip for dark images
            print(f"Skipping horizon detection for dark image: {fpath_input}")
            cv2.imwrite(str(fpath_output_image), image_original)
            horizon_coords_data[str(fpath_input)] = {
                'horizon_x1': None,
                'horizon_x2': None,
                'horizon_y1': None,
                'horizon_y2': None,
                'average_brightness': float(average_brightness),
                'is_night': True
            }
        elif white_pixel_percentage > SNOW_WHITE_PIXELS_THRESHOLD:
            # Skip for snow images
            cv2.imwrite(str(fpath_output_image), image_original)
            print(f"Skipping horizon detection for snow image: {fpath_input}")
            horizon_coords_data[str(fpath_input)] = {
                'horizon_x1': None,
                'horizon_x2': None,
                'horizon_y1': None,
                'horizon_y2': None,
                'average_brightness': float(average_brightness),
                'is_night': False
            }
        else:
            # Proceed with horizon detection for all other images
            horizon_x1, horizon_x2, horizon_y1, horizon_y2 = detect_horizon_line(
                image_grayscale
            )
            
            # Check if a horizon line was detected
            if horizon_y1 is not None:
                # Check if the horizon line is in the bottom 10% of the image
                image_height = image_original.shape[0]
                if horizon_y1 > image_height * LOW_HORIZON_THRESHOLD_PERCENTAGE:
                    print(f"Skipping horizon detection for image {fpath_input} because the horizon is too low.")
                    # --- CRITICAL FIX ---
                    # Add this line to save the original image
                    cv2.imwrite(str(fpath_output_image), image_original)
                    # --- END CRITICAL FIX ---
                    horizon_coords_data[str(fpath_input)] = {
                        'horizon_x1': None,
                        'horizon_x2': None,
                        'horizon_y1': None,
                        'horizon_y2': None,
                        'average_brightness': float(average_brightness),
                        'is_night': False
                    }
                else:
                    # Proceed with drawing the line
                    cv2.line(
                        image_original,
                        (horizon_x1, horizon_y1),
                        (horizon_x2, horizon_y2),
                        (0, 0, 255),
                        2
                    )
                    cv2.imwrite(str(fpath_output_image), image_original)
                    print(f"Saved horizon-annotated image to {fpath_output_image}")

                    # Store the coordinates in the dictionary
                    horizon_coords_data[str(fpath_input)] = {
                        'horizon_x1': int(horizon_x1),
                        'horizon_x2': int(horizon_x2),
                        'horizon_y1': int(horizon_y1),
                        'horizon_y2': int(horizon_y2),
                        'average_brightness': float(average_brightness),
                        'is_night': False
                    }
            else:
                # If no horizon line was detected, save the original image and store None values
                cv2.imwrite(str(fpath_output_image), image_original)
                print(f"Skipping horizon detection for an un-detectable image: {fpath_input}")
                horizon_coords_data[str(fpath_input)] = {
                    'horizon_x1': None,
                    'horizon_x2': None,
                    'horizon_y1': None,
                    'horizon_y2': None,
                    'average_brightness': float(average_brightness),
                    'is_night': False
                }

    # Save the coordinates dictionary to a JSON file
    with open(str(fpath_output_coords), 'w') as f:
        json.dump(horizon_coords_data, f, indent=4)
    print(f"Saved horizon coordinates to {fpath_output_coords}")


if __name__ == '__main__':
    main()
