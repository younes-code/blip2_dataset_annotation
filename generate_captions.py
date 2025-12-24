# import os
# import torch
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
# import time
# import gc

# def generate_caption(model, processor, img_path, device):
#     """
#     Generates a caption for the given image and measures processing time.
    
#     Args:
#         model: BLIP-2 model for caption generation.
#         processor: BLIP-2 processor for image preprocessing.
#         img_path: Path to the input image.
#         device: Device for inference (e.g., "cuda").
    
#     Returns:
#         tuple: (caption, time_taken in seconds).
#     """
#     img_name = os.path.basename(img_path)
#     raw_image = Image.open(img_path).convert('RGB')
    
#     # Start timing
#     start_time = time.time()
    
#     inputs = processor(raw_image, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         generated_ids = model.generate(**inputs, max_length=1000, num_beams=5)
    
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
#     # End timing
#     time_taken = time.time() - start_time
    
#     # Cleanup
#     del inputs, generated_ids
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     result = f'{os.path.basename(os.path.dirname(img_path))}/{img_name} ## {generated_text}\n'
#     return result, time_taken

# def main(ucf_path, save_path, max_frames=5):
#     """
#     Processes a sample of frames to estimate average caption generation time.
    
#     Args:
#         ucf_path: Path to directory containing frame images.
#         save_path: Path to save the generated captions.
#         max_frames: Maximum number of frames to process for timing estimation.
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Configure quantization
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         llm_int8_enable_fp32_cpu_offload=True,
#         bnb_4bit_compute_dtype=torch.float32
#     )
    
#     # Load BLIP-2 processor and model
#     print("Loading BLIP-2 model...")
#     processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#     model = Blip2ForConditionalGeneration.from_pretrained(
#         "Salesforce/blip2-opt-2.7b",
#         quantization_config=bnb_config,
#         device_map="auto"
#     )
    
#     # Initialize timing and frame counter
#     times = []
#     processed_frames = 0
    
#     # Process frames
#     for root, _, files in os.walk(ucf_path):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                 img_path = os.path.join(root, file)
                
#                 print(f"Processing frame: {img_path}")
#                 caption, time_taken = generate_caption(model, processor, img_path, device)
                
#                 # Save caption
#                 with open(save_path, 'a') as f:
#                     f.write(caption)
                
#                 print(f"Caption: {caption.strip()}")
#                 print(f"Time for this frame: {time_taken:.2f} seconds")
                
#                 times.append(time_taken)
#                 processed_frames += 1
                
#                 # Stop after max_frames
#                 if processed_frames >= max_frames:
#                     break
#             if processed_frames >= max_frames:
#                 break
#         if processed_frames >= max_frames:
#             break
    
#     # Calculate and display average time
#     if processed_frames > 0:
#         total_time = sum(times)
#         avg_time = total_time / processed_frames
#         print(f"\nProcessed {processed_frames} frames in {total_time:.2f} seconds")
#         print(f"Average time per frame: {avg_time:.2f} seconds")
#     else:
#         print("No frames processed.")
    
#     # Final cleanup
#     del model, processor
#     torch.cuda.empty_cache()
#     gc.collect()

# if __name__ == "__main__":
#     ucf_directory = 'test_video-frames'  # Adjust to your frame directory
#     captions_save_path = 'blip2_test_caption.txt'
#     main(ucf_directory, captions_save_path, max_frames=5)


###############################################Tempral Selection########################################

import os
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import time
import gc
import re

# ---------------------------
# Utility functions
# ---------------------------
def extract_frame_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def generate_caption(model, processor, img_path, device):
    img_name = os.path.basename(img_path)
    raw_image = Image.open(img_path).convert('RGB')

    start_time = time.time()

    inputs = processor(raw_image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=100, num_beams=3)

    caption_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    time_taken = time.time() - start_time

    # cleanup
    del inputs, generated_ids
    torch.cuda.empty_cache()
    gc.collect()

    caption = f"{os.path.basename(os.path.dirname(img_path))}/{img_name} ## {caption_text}\n"
    return caption, time_taken

def get_processed_frames(save_path):
    """Return a set of already processed frame paths (for resuming jobs)."""
    if not os.path.exists(save_path):
        return set()
    processed = set()
    with open(save_path, "r") as f:
        for line in f:
            processed.add(line.split(" ## ")[0])
    return processed

# ---------------------------
# Main processing
# ---------------------------
def main(ucf_path, save_filename, skip_frames=1):
    output_folder = "temporal_selection_blip"
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, save_filename)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quantized model loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_compute_dtype=torch.float32
    )

    print("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        quantization_config=bnb_config,
        device_map="auto"
    )

    processed_frames_set = get_processed_frames(save_path)
    processed_frames = 0
    total_time = 0
    frame_count = 0

    for root, dirs, files in os.walk(ucf_path):
        files.sort(key=extract_frame_number)

        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                img_path = os.path.join(root, file)

                if img_path in processed_frames_set:
                    continue  # skip already processed frames

                if frame_count % skip_frames == 0:
                    print(f"Processing: {img_path}")

                    caption, t = generate_caption(model, processor, img_path, device)

                    with open(save_path, "a") as f:
                        f.write(caption)

                    processed_frames += 1
                    total_time += t

                frame_count += 1

    avg_time = total_time / processed_frames if processed_frames else 0

    log_message = (
        f"Processed {processed_frames} frames with skip={skip_frames}\n"
        f"Total time: {total_time:.2f}s\n"
        f"Average time per frame: {avg_time:.2f}s\n\n"
    )

    print(log_message)

    # Write log file
    log_file_path = os.path.join(output_folder, "blip_time_log.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(log_message)

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    ucf_directory = '../Datasets/active_frames_output'  # INPUT DATA

    # Each tuple = (skip_frames, output_file)
    configs = [
        (1, "captions_all.txt"),
        (3, "captions_3.txt"),
        (5, "captions_5.txt"),
    ]

    for skip_frames, filename in configs:
        print(f"\nRunning BLIP with skip_frames={skip_frames}")
        main(ucf_directory, filename, skip_frames)
