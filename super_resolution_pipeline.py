"""
Super-Resolution Pipeline for Oral Images
Based on the paper methodology:
- Input: 256×256 generated oral images (healthy and decayed)
- Model: Pre-trained realesrgan-x4plus without fine-tuning
- Process: 6× upscaling to 1536×1536, then resize to 1920×1080 using Lanczos
"""

import glob
import cv2
import os
import shutil
import numpy as np
import subprocess
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class SuperResolutionPipeline:
    def __init__(self, input_dir, output_dir, temp_dir="temp_esrgan"):
        """
        Initialize the Super-Resolution Pipeline

        Args:
            input_dir (str): Directory containing 256×256 input images
            output_dir (str): Directory to save final 1920×1080 images
            temp_dir (str): Temporary directory for ESRGAN processing
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.esrgan_script_path = "inference_realesrgan.py"

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def validate_and_resize_input(self, img_path, target_size=(256, 256)):
        """
        Validate input image and resize to 256×256 if needed

        Args:
            img_path (str): Path to input image
            target_size (tuple): Target size (width, height)

        Returns:
            str: Path to processed image
        """
        img = Image.open(img_path)

        # Resize to 256×256 if not already
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)

        # Save to temp directory
        img_name = os.path.basename(img_path)
        temp_path = os.path.join(self.temp_dir, img_name)
        img.save(temp_path)

        return temp_path

    def prepare_input_images(self):
        """
        Prepare all input images by validating and resizing to 256×256

        Returns:
            list: List of processed image paths
        """
        input_images = glob.glob(os.path.join(self.input_dir, "*.png"))
        input_images.extend(glob.glob(os.path.join(self.input_dir, "*.jpg")))
        input_images.extend(glob.glob(os.path.join(self.input_dir, "*.jpeg")))

        print(f"Found {len(input_images)} input images")

        processed_paths = []
        for img_path in tqdm(input_images, desc="Preparing input images"):
            processed_path = self.validate_and_resize_input(img_path)
            processed_paths.append(processed_path)

        return processed_paths

    def run_esrgan_upscaling(self):
        """
        Run Real-ESRGAN for 6× upscaling (256×256 → 1536×1536)
        Using pre-trained realesrgan-x4plus model with default parameters
        """
        print("Running Real-ESRGAN 6× upscaling...")

        # ESRGAN command with 6× scaling
        command = [
            "python", self.esrgan_script_path,
            "-n", "RealESRGAN_x4plus",  # Pre-trained model
            "-i", self.temp_dir,        # Input directory
            "-o", self.temp_dir,        # Output directory
            "-s", "6",                  # 6× upscaling
            "--fp32"                    # Use FP32 for stability
        ]

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # Monitor progress
            for line in process.stdout:
                print(line.strip())

            process.wait()

            if process.returncode == 0:
                print("✓ Real-ESRGAN processing completed successfully")
            else:
                raise RuntimeError("Real-ESRGAN processing failed")

        except Exception as e:
            print(f"Error during ESRGAN processing: {e}")
            raise

    def resize_to_final_resolution(self):
        """
        Resize 1536×1536 images to final 1920×1080 resolution using Lanczos interpolation
        """
        print("Resizing to final 1920×1080 resolution...")

        # Find ESRGAN output images (with _out suffix)
        esrgan_outputs = glob.glob(os.path.join(self.temp_dir, "*_out.png"))

        def process_single_image(img_path):
            try:
                # Load 1536×1536 image
                img = Image.open(img_path)

                # Verify it's 1536×1536 (6× from 256×256)
                if img.size != (1536, 1536):
                    print(f"Warning: Expected 1536×1536, got {img.size} for {img_path}")

                # Resize to 1920×1080 using Lanczos interpolation
                final_img = img.resize((1920, 1080), Image.LANCZOS)

                # Save with original name (remove _out suffix)
                img_name = os.path.basename(img_path).replace("_out.png", ".png")
                output_path = os.path.join(self.output_dir, img_name)
                final_img.save(output_path)

                return output_path

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                return None

        # Process images in parallel
        with Pool(cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(process_single_image, esrgan_outputs),
                total=len(esrgan_outputs),
                desc="Resizing images"
            ))

        successful = [r for r in results if r is not None]
        print(f"✓ Successfully processed {len(successful)}/{len(esrgan_outputs)} images")

        return successful

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("✓ Temporary files cleaned up")

    def run_pipeline(self, cleanup=True):
        """
        Run the complete super-resolution pipeline

        Args:
            cleanup (bool): Whether to clean up temporary files after processing

        Returns:
            list: List of output image paths
        """
        try:
            print("=" * 60)
            print("SUPER-RESOLUTION PIPELINE")
            print("256×256 → 1536×1536 (6× ESRGAN) → 1920×1080 (Lanczos)")
            print("=" * 60)

            # Step 1: Prepare input images (ensure 256×256)
            print("\n1. Preparing input images...")
            processed_inputs = self.prepare_input_images()

            # Step 2: Run ESRGAN 6× upscaling
            print("\n2. Running Real-ESRGAN 6× upscaling...")
            self.run_esrgan_upscaling()

            # Step 3: Resize to final resolution
            print("\n3. Resizing to final resolution...")
            output_paths = self.resize_to_final_resolution()

            # Step 4: Cleanup
            if cleanup:
                print("\n4. Cleaning up...")
                self.cleanup_temp_files()

            print(f"\n✓ Pipeline completed! {len(output_paths)} images saved to {self.output_dir}")
            return output_paths

        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            if cleanup:
                self.cleanup_temp_files()
            raise


def main():
    """
    Example usage of the Super-Resolution Pipeline
    """
    # Configuration
    input_directory = "inputs/"    # Directory with 256×256 input images
    output_directory = "outputs/"  # Final output directory

    # Create pipeline instance
    pipeline = SuperResolutionPipeline(
        input_dir=input_directory,
        output_dir=output_directory
    )

    # Run the complete pipeline
    try:
        output_files = pipeline.run_pipeline()
        print(f"\nSuccess! Processed {len(output_files)} images.")
        print(f"Output location: {output_directory}")

    except Exception as e:
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
