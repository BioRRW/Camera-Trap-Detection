# simple_megadetector_test.py
#
# An updated script to run MegaDetector batch processing that now
# correctly generates relative paths in its output JSON.
#
# To run:
# conda activate megadetector
# python simple_megadetector_test.py

import subprocess
import sys
import argparse
import os

# --- Configuration ---
# It's good practice to keep these as defaults that can be overridden.
MODEL_FILE = "./path/to/megadetector/model/megadetector_models/md_v5a.0.0.pt"
IMAGE_FOLDER = "/path/to/photos/"
OUTPUT_FILE = "/desired/path/to/output/json/photos_all_000001.json"
NUM_CORES = "8"
THRESHOLD = "0.000001"

# --- Main execution ---
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run MegaDetector batch processing.")
    parser.add_argument("--model_file", type=str, default=MODEL_FILE,
                        help="Path to the MegaDetector model file.")
    parser.add_argument("--image_folder", type=str, default=IMAGE_FOLDER,
                        help="Path to the folder containing your images.")
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE,
                        help="Path for the JSON output file.")
    parser.add_argument("--threshold", type=str, default=THRESHOLD,
                        help="Confidence threshold for detections.")
    parser.add_argument("--ncores", type=str, default=NUM_CORES,
                        help="Number of parallel cores to use for processing.")
    args = parser.parse_args()
    
    # Set the CUDA_VISIBLE_DEVICES environment variable to force CPU use
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    python_executable = sys.executable

    # Create the full command as a list of strings
    command = [
        python_executable,
        "-m", "megadetector.detection.run_detector_batch",
        args.model_file,
        args.image_folder,
        args.output_file,
        "--threshold",
        args.threshold,
        "--ncores",
        args.ncores,
        "--recursive",
        
        # This is the key flag to solve the pathing issue.
        # It ensures all file paths in the output JSON are relative
        # to the --image_folder path.
        "--output_relative_filenames"
    ]

    # Execute the command
    print(f"\nRUNNING COMMAND: {' '.join(command)}\n")
    subprocess.run(command, check=True)

    print(f"\nBatch processing complete. Results saved to:\n{args.output_file}")


if __name__ == '__main__':
    main()
