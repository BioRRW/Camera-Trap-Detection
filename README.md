# Camera-Trap-Detection
Camera Trap Detection Analysis focused on accurate detection


Automated Wildlife Detection Pipeline for Camera Trap Biosecurity Surveillance
This repository contains the scripts and documentation for an optimized data pipeline designed to process camera trap images. The project focuses on leveraging machine learning tools to efficiently detect wildlife presence on agricultural premises.

Project Overview
The primary goal of this project is high-sensitivity wildlife detection, aiming to identify if an animal is present in an image rather than classifying the specific species. This approach is critical for biosecurity surveillance, where minimizing false negatives (missed detections) is more important than avoiding false positives.

The pipeline processes a large dataset of images (30,000+) using a multi-stage approach that includes initial AI-based detection, removal of recurring false positives, horizon detection, and the application of variable confidence thresholds based on environmental conditions (e.g., sky vs. ground, day vs. night).

Key Findings

The automated pipeline significantly reduces manual effort, removing a substantial portion of empty images and saving hundreds of hours of review time.

Optimal detection requires different confidence thresholds for different conditions. For example, a lower threshold is needed for detecting airborne wildlife (birds) above the horizon than for animals on the ground.

A major challenge in this study was a recurring false positive from a center-pivot irrigation system. A suspicious duplicate detection filter was successfully used to identify and remove over 11,000 of these false detection events.

Physical camera movement can reduce the effectiveness of automated false positive removal tools that rely on a static background.

The Analysis Pipeline
The workflow uses a series of Python scripts to process the camera trap images in sequential steps.

<p align="center">
  <img src="images/Camera Trap Detection.pdf" width="300" height="600">
</p>


Initial Detection: Run MegaDetector on the entire image set to generate a primary JSON file with all potential detections.

Suspicious Duplicate Removal: Identify and manually curate recurring, stationary detections that are likely false positives (e.g., equipment, vegetation).

Horizon Detection: Analyze all images to determine the coordinates of the horizon line, which allows for different processing of sky and ground.

Final Filtering & Sorting: Apply a final set of variable confidence thresholds based on the horizon data and other conditions (day/night) to produce a clean, sorted output.

Scripts
This repository contains the following key scripts:

simple_megadetector_test.py: A wrapper script to run the initial MegaDetector batch processing. It generates the first raw JSON output.

find_repeat_detections.py (from MegaDetector): Identifies detections that repeatedly occur in the same location across multiple images.

remove_repeat_detections.py (from MegaDetector): Removes the detections that were manually confirmed as false positives.

detect_horizons_json.py: A modified script to detect the horizon line in each image and save the coordinates.

final_md_process_sort_v3.py: The main orchestrator script that applies the final horizon-based filtering and sorts the images into output folders.

postprocess_md_horizon.py: A helper module used by the final script to handle the filtering logic.

How to Run the Analysis
Follow these steps to replicate the entire data processing workflow. All paths should be updated to match your local file structure.

Step 1: Initial MegaDetector Run

This script runs MegaDetector on all images and generates the initial raw results JSON file.

python simple_megadetector_test.py

Output: photos_all_000001.json

Step 2: Find Suspicious Duplicate Detections

This script identifies recurring detections that are candidates for removal.

python /path/to/MegaDetector/megadetector/postprocessing/repeat_detection_elimination/find_repeat_detections.py \
    "/path/to/photos_all_000001.json" \
    --imageBase "/path/to/WBA_Camera_Trap_Photos/photos_all/" \
    --outputBase "/path/to/susp_dupes_removal_photos_all"

Output: A timestamped filtering_... folder containing images of suspicious detections.

Step 3: Manual Curation

Navigate into the newly created filtering_... folder. Delete all images that show real animals you want to keep. Leave behind only the images of the actual false positives (e.g., the irrigation system).

Step 4: Remove Confirmed Duplicates

This script removes the detections corresponding to the false positive images you left in the filtering folder.

python /path/to/MegaDetector/megadetector/postprocessing/repeat_detection_elimination/remove_repeat_detections.py \
    "/path/to/photos_all_000001.json" \
    "/path/to/dupes_removed_photos_all.json" \
    "/path/to/susp_dupes_removal_photos_all/filtering_2025.09.04.07.38.19/"

Input: The original JSON and the curated filtering folder.

Output: dupes_removed_photos_all.json

Step 5: Detect Horizons

This script runs in parallel to the other steps. It analyzes all original images to find the horizon line.

python horizon_detection/detect_horizons_json.py \
    --dirpath_input_images /path/to/WBA_Camera_Trap_Photos/photos_all/ \
    --dirpath_output_images /path/to/horizon_out_photos_all/ \
    --fpath_output_coords /path/to/horizon_out_photos_all.json

Output: horizon_out_photos_all.json

Step 6: Final Filtering and Sorting

This is the final step. It uses the cleaned JSON from Step 4 and the horizon data from Step 5 to apply variable thresholds and sort the final images.

python final_md_process_sort_v3.py \
    --results_json /path/to/dupes_removed_photos_all.json \
    --image_dir /path/to/WBA_Camera_Trap_Photos/photos_all/ \
    --output_dir /path/to/final_step_photos_all/ \
    --horizon_data_json /path/to/horizon_out_photos_all.json \
    --sky_threshold 0.0001 \
    --ground_threshold 0.0015 \
    --night_threshold 0.0000015

Output: A final output directory containing sorted folders of visualized and original images (animal_original, empty_original, etc.).

Citations
Beery, S., Morris, D., & Yang, S. (2019). Efficient Pipeline for Camera Trap Image Review. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops.

Gadot, T., et al. (2024). To crop or not to crop: comparing whole-image and cropped classification on a large dataset of camera trap images. IET Computer Vision, 18(8), 1193–1208.

Fall, S. Horizon Detection [Computer software]. http://github.com/sallamander/horizon-detection/

