# final_process_and_sort_v3.py
#
# A script to run the full MegaDetector post-processing workflow.
# This is a self-contained version that includes all logic for applying
# different thresholds for sky, ground, and night images.

import subprocess
import sys
import os
import json
import shutil
import argparse
import cv2
from tqdm import tqdm

# --- Configuration ---
TEMP_FILTERED_JSON = "filtered_temp_md_output.json"

# --- Helper Functions ---

def run_command(command):
    """A helper function to run a command."""
    print(f"\nRUNNING COMMAND: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR ---")
        print("STDOUT:\n" + e.stdout)
        print("STDERR:\n" + e.stderr)
        raise

def filter_detections_by_horizon(md_output_file, horizon_coords_file, output_file, image_dir,
                                 sky_threshold, ground_threshold, night_threshold):
    """
    Filters detections in a MegaDetector output file based on whether they
    fall above or below a detected horizon line, with a separate threshold for night images.
    """
    print("--- Starting horizon-based filtering ---")
    try:
        with open(md_output_file, 'r') as f:
            md_data = json.load(f)
        with open(horizon_coords_file, 'r') as f:
            horizon_data = json.load(f)
    except Exception as e:
        print(f"FATAL ERROR: Could not read or parse JSON files: {e}")
        return

    if 'images' not in md_data or not md_data['images']:
        print("Warning: MegaDetector JSON file contains no 'images'. Nothing to process.")
        return
        
    print(f"Loaded {len(md_data['images'])} image entries from MD results.")
    
    horizon_lookup = {os.path.basename(k): v for k, v in horizon_data.items()}
    filtered_images = []

    print("Processing images and applying thresholds...")
    # Removed tqdm wrapper to ensure print statements are visible for debugging
    for image in md_data['images']:
        image_basename = os.path.basename(image['file'])
        
        if image_basename in horizon_lookup:
            # Added this print statement for positive confirmation
            print(f"  - Horizon data FOUND for {image_basename}.")
            horizon_info = horizon_lookup[image_basename]
            new_detections = []
            
            if horizon_info.get('is_night', False):
                for det in image['detections']:
                    if det['conf'] >= night_threshold:
                        new_detections.append(det)
            elif horizon_info['horizon_y1'] is not None:
                h_y1, h_y2 = horizon_info['horizon_y1'], horizon_info['horizon_y2']
                
                # We need the full path to read the image for its dimensions
                full_image_path = os.path.join(image_dir, image['file'])
                try:
                    img = cv2.imread(full_image_path)
                    if img is None: raise IOError("Image not found or unreadable")
                    
                    image_height, image_width, _ = img.shape
                    
                    def get_horizon_y_at_x(x_px):
                        if image_width == 1: return h_y1
                        return h_y1 + (h_y2 - h_y1) * (x_px / (image_width - 1))

                    for det in image['detections']:
                        bbox_center_x_rel = det['bbox'][0] + det['bbox'][2] / 2
                        bbox_center_y_rel = det['bbox'][1] + det['bbox'][3] / 2
                        bbox_center_x_px = bbox_center_x_rel * image_width
                        bbox_center_y_px = bbox_center_y_rel * image_height
                        horizon_y_px = get_horizon_y_at_x(bbox_center_x_px)

                        if bbox_center_y_px < horizon_y_px: # smaller y is higher up
                            if det['conf'] >= sky_threshold: new_detections.append(det)
                        else:
                            if det['conf'] >= ground_threshold: new_detections.append(det)
                except Exception as e:
                    print(f"\nWarning: Could not process {image_basename} for horizon check: {e}. Applying ground threshold.")
                    for det in image['detections']:
                        if det['conf'] >= ground_threshold: new_detections.append(det)
            else: # Day image with no horizon, apply ground threshold
                for det in image['detections']:
                    if det['conf'] >= ground_threshold: new_detections.append(det)

            image['detections'] = new_detections
        else:
            # This warning for no match was already here
            print(f"Warning: No horizon data found for {image_basename}. Keeping original detections.")
        
        filtered_images.append(image)

    output_data = {'info': md_data.get('info', {}), 'detection_categories': md_data.get('detection_categories', {}), 'images': filtered_images}
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=1)
    print(f"\nFinished filtering. Saved {len(filtered_images)} image entries.")

# --- Main Orchestrator ---

def main():
    """Main execution function with new horizon logic."""
    parser = argparse.ArgumentParser(description="Full MegaDetector post-processing workflow with horizon filtering.")
    parser.add_argument("--results_json", type=str, required=True, help="Path to the raw MegaDetector output JSON file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the folder containing your images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path for the final output folder.")
    parser.add_argument("--horizon_data_json", type=str, required=True, help="Path to the JSON file with horizon line coordinates.")
    parser.add_argument("--sky_threshold", type=float, default=0.0001, help="Confidence threshold for detections in the sky.")
    parser.add_argument("--ground_threshold", type=float, default=0.15, help="Confidence threshold for detections on the ground.")
    parser.add_argument("--night_threshold", type=float, default=0.05, help="Confidence threshold for detections in night images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    python_executable = sys.executable

    # --- Step 1: Apply horizon-based thresholds ---
    print("\nSTEP 1: Applying horizon-based thresholds.")
    temp_json_path = os.path.join(args.output_dir, TEMP_FILTERED_JSON)
    filter_detections_by_horizon(
        md_output_file=args.results_json,
        horizon_coords_file=args.horizon_data_json,
        output_file=temp_json_path,
        image_dir=args.image_dir,
        sky_threshold=args.sky_threshold,
        ground_threshold=args.ground_threshold,
        night_threshold=args.night_threshold
    )

    print(f"✅ Filtered JSON saved to: {temp_json_path}")

    # --- Step 2: Generate visualized images and HTML reports using the new JSON ---
    print("\nSTEP 2: Generating visualized images and reports.")
    sort_command = [
        python_executable,
        "-m", "megadetector.postprocessing.postprocess_batch_results",
        temp_json_path,
        args.output_dir,
        "--image_base_dir", args.image_dir,
        "--confidence_threshold", str(min(args.sky_threshold, args.ground_threshold, args.night_threshold)),
        "--num_images_to_sample", "-1",
    ]
    run_command(sort_command)

    # --- Step 3: Copy original, un-boxed images based on the new filtered JSON ---
    print("\nSTEP 3: Copying original (un-boxed) images to new folders.")
    with open(temp_json_path) as f:
        data = json.load(f)

    category_map = {"1": "animal", "2": "human", "3": "vehicle"}
    image_index = {os.path.basename(f): f for dirpath, _, files in os.walk(args.image_dir) for f in files}

    for image in data['images']:
        category_name = "empty"
        if image['detections']:
            top_detection = max(image['detections'], key=lambda d: d['conf'])
            category_name = category_map.get(str(top_detection['category']), "empty")

        filename_to_find = os.path.basename(image['file'])
        if filename_to_find in image_index:
            found_path = os.path.join(args.image_dir, image_index[filename_to_find])
            dest_folder = os.path.join(args.output_dir, f"{category_name}_original")
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(found_path, dest_folder)
        else:
            print(f"Warning: Could not find '{filename_to_find}' anywhere within '{args.image_dir}'")
            
    print(f"\n✅ Processing complete! Find both original and visualized images in: {args.output_dir}")

    # --- Final Cleanup ---
    os.remove(temp_json_path)
    print(f"Cleaned up temporary file: {temp_json_path}")

if __name__ == '__main__':
    main()

