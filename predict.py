# RunPod serverless handler for FLUX.1-Krea-dev
# https://docs.runpod.io/serverless/workers/handlers

import runpod
import os
import re
import time
import torch
import subprocess
import base64
import requests
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any
from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline
)
from torchvision import transforms
from weights import WeightsDownloadCache
from transformers import CLIPImageProcessor
import numpy as np

MAX_IMAGE_SIZE = 1440
MODEL_CACHE = "FLUX.1-Krea-dev"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}

def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

# Global variables for model caching
txt2img_pipe = None
img2img_pipe = None
weights_cache = None
last_loaded_lora = None
feature_extractor = None

def initialize_models():
    """Initialize models globally for efficiency"""
    global txt2img_pipe, img2img_pipe, weights_cache, last_loaded_lora, feature_extractor
    
    if txt2img_pipe is not None:
        return
        
    start = time.time()
    print("Initializing models...")
    
    weights_cache = WeightsDownloadCache()
    last_loaded_lora = None
    
    # Initialize CLIPImageProcessor like Cog does
    print("Loading CLIP feature extractor...")
    feature_extractor = CLIPImageProcessor.from_pretrained("./feature-extractor")
    
    print("Loading Flux txt2img Pipeline")
    dtype = torch.bfloat16
    txt2img_pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        torch_dtype=dtype,
        cache_dir=MODEL_CACHE,
        low_cpu_mem_usage=True
    ).to("cuda")

    print("Loading Flux img2img pipeline")
    img2img_pipe = FluxImg2ImgPipeline(
        transformer=txt2img_pipe.transformer,
        scheduler=txt2img_pipe.scheduler,
        vae=txt2img_pipe.vae,
        text_encoder=txt2img_pipe.text_encoder,
        text_encoder_2=txt2img_pipe.text_encoder_2,
        tokenizer=txt2img_pipe.tokenizer,
        tokenizer_2=txt2img_pipe.tokenizer_2,
    ).to("cuda")

    print("Model initialization took: ", time.time() - start)

def aspect_ratio_to_width_height(aspect_ratio: str) -> tuple[int, int]:
    return ASPECT_RATIOS.get(aspect_ratio, ASPECT_RATIOS["1:1"])

def get_image_from_url(image_url: str):
    """Download and process image from URL"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise Exception(f"Error downloading image from URL: {e}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2.0 * x - 1.0),
    ])
    img: torch.Tensor = transform(image)
    return img[None, ...]

def get_image_from_base64(image_b64: str):
    """Process image from base64 string"""
    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise Exception(f"Error processing base64 image: {e}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2.0 * x - 1.0),
    ])
    img: torch.Tensor = transform(image)
    return img[None, ...]

def make_multiple_of_16(n):
    return ((n + 15) // 16) * 16

def image_to_base64(image: Image.Image, format: str = "webp", quality: int = 80) -> str:
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    if format.lower() != 'png':
        image.save(buffer, format=format.upper(), quality=quality, optimize=True)
    else:
        image.save(buffer, format=format.upper())
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format};base64,{img_str}"

@torch.inference_mode()
def generate_images(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Main generation function"""
    global txt2img_pipe, img2img_pipe, weights_cache, last_loaded_lora
    
    # Initialize models if not already done
    initialize_models()
    
    # Parse inputs with defaults
    prompt = job_input.get("prompt", "")
    aspect_ratio = job_input.get("aspect_ratio", "1:1")
    image_input = job_input.get("image", None)  # URL or base64
    prompt_strength = job_input.get("prompt_strength", 0.8)
    num_outputs = job_input.get("num_outputs", 1)
    num_inference_steps = job_input.get("num_inference_steps", 28)
    guidance_scale = job_input.get("guidance_scale", 3.5)
    seed = job_input.get("seed", None)
    output_format = job_input.get("output_format", "webp")
    output_quality = job_input.get("output_quality", 80)
    hf_lora = job_input.get("hf_lora", None)
    lora_scale = job_input.get("lora_scale", 0.8)
    
    # Validation
    if not prompt:
        return {"error": "Prompt is required"}
    
    if aspect_ratio not in ASPECT_RATIOS:
        return {"error": f"Invalid aspect ratio. Must be one of: {list(ASPECT_RATIOS.keys())}"}
    
    if num_outputs < 1 or num_outputs > 4:
        return {"error": "num_outputs must be between 1 and 4"}
    
    if num_inference_steps < 1 or num_inference_steps > 50:
        return {"error": "num_inference_steps must be between 1 and 50"}
    
    if guidance_scale < 0 or guidance_scale > 10:
        return {"error": "guidance_scale must be between 0 and 10"}
    
    if output_format not in ["webp", "jpg", "png"]:
        return {"error": "output_format must be webp, jpg, or png"}
    
    if output_quality < 0 or output_quality > 100:
        return {"error": "output_quality must be between 0 and 100"}
    
    if lora_scale < 0 or lora_scale > 2:
        return {"error": "lora_scale must be between 0 and 2"}
    
    if prompt_strength < 0 or prompt_strength > 1:
        return {"error": "prompt_strength must be between 0 and 1"}
    
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    
    print(f"Using seed: {seed}")
    print(f"Prompt: {prompt}")
    
    width, height = aspect_ratio_to_width_height(aspect_ratio)
    max_sequence_length = 512
    
    flux_kwargs = {"width": width, "height": height}
    device = txt2img_pipe.device

    # Handle image input for img2img
    if image_input:
        pipe = img2img_pipe
        print("img2img mode")
        
        # Determine if input is URL or base64
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                init_image = get_image_from_url(image_input)
            elif image_input.startswith('data:image/'):
                # Extract base64 part from data URL
                base64_data = image_input.split(',')[1]
                init_image = get_image_from_base64(base64_data)
            else:
                # Assume it's plain base64
                init_image = get_image_from_base64(image_input)
        else:
            return {"error": "Image input must be a URL or base64 string"}
        
        width = init_image.shape[-1]
        height = init_image.shape[-2]
        print(f"Input image size: {width}x{height}")
        
        # Calculate the scaling factor if the image exceeds MAX_IMAGE_SIZE
        scale = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height, 1)
        if scale < 1:
            width = int(width * scale)
            height = int(height * scale)
            print(f"Scaling image down to {width}x{height}")

        # Round image width and height to nearest multiple of 16
        width = make_multiple_of_16(width)
        height = make_multiple_of_16(height)
        print(f"Input image size set to: {width}x{height}")
        
        # Resize
        init_image = init_image.to(device)
        init_image = torch.nn.functional.interpolate(init_image, (height, width))
        init_image = init_image.to(torch.bfloat16)
        
        # Set params
        flux_kwargs["image"] = init_image
        flux_kwargs["strength"] = prompt_strength
    else:
        print("txt2img mode")
        pipe = txt2img_pipe

    # Handle LoRA loading
    if hf_lora:
        flux_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
        t1 = time.time()
        # check if extra_lora is new
        if hf_lora != last_loaded_lora:
            pipe.unload_lora_weights()
            if re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$", hf_lora):
                print(f"Downloading LoRA weights from - HF path: {hf_lora}")
                pipe.load_lora_weights(hf_lora)
            # Check for Replicate tar file
            elif re.match(r"^https?://replicate.delivery/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/trained_model.tar", hf_lora):
                print(f"Downloading LoRA weights from - Replicate URL: {hf_lora}")
                local_weights_cache = weights_cache.ensure(hf_lora)
                lora_path = os.path.join(local_weights_cache, "output/flux_train_replicate/lora.safetensors")
                pipe.load_lora_weights(lora_path)
            # Check for Huggingface URL
            elif re.match(r"^https?://huggingface.co", hf_lora):
                print(f"Downloading LoRA weights from - HF URL: {hf_lora}")
                huggingface_slug = re.search(r"^https?://huggingface.co/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)", hf_lora).group(1)
                weight_name = hf_lora.split('/')[-1]
                print(f"HuggingFace slug from URL: {huggingface_slug}, weight name: {weight_name}")
                pipe.load_lora_weights(huggingface_slug, weight_name=weight_name)
            # Check for Civitai URL
            elif re.match(r"^https?://civitai.com/api/download/models/[0-9]+\?type=Model&format=SafeTensor", hf_lora):
                # split url to get first part of the url, everythin before '?type'
                civitai_slug = hf_lora.split('?type')[0]
                print(f"Downloading LoRA weights from - Civitai URL: {civitai_slug}")
                lora_path = weights_cache.ensure(hf_lora, file=True)
                pipe.load_lora_weights(lora_path)
            # Check for URL to a .safetensors file
            elif hf_lora.endswith('.safetensors'):
                print(f"Downloading LoRA weights from - safetensor URL: {hf_lora}")
                try:
                    lora_path = weights_cache.ensure(hf_lora, file=True)
                except Exception as e:
                    return {"error": f"Error downloading LoRA weights from URL: {e}"}
                pipe.load_lora_weights(lora_path)
            else:
                return {"error": f"Invalid lora, must be either a: HuggingFace path, Replicate model.tar URL, or a URL to a .safetensors file: {hf_lora}"}

            # Move the entire pipeline to GPU after loading LoRA weights
            pipe = pipe.to("cuda")
            
            last_loaded_lora = hf_lora
        t2 = time.time()
        print(f"Loading LoRA took: {t2 - t1:.2f} seconds")
    else:
        flux_kwargs["joint_attention_kwargs"] = None
        pipe.unload_lora_weights()
        last_loaded_lora = None

    # Ensure the pipeline is on GPU
    pipe = pipe.to("cuda")

    generator = torch.Generator("cuda").manual_seed(seed)

    common_args = {
        "prompt": [prompt] * num_outputs,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "num_inference_steps": num_inference_steps,
        "max_sequence_length": max_sequence_length,
        "output_type": "pil"
    }

    try:
        output = pipe(**common_args, **flux_kwargs)
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

    # Convert images to base64
    images_b64 = []
    for i, image in enumerate(output.images):
        try:
            img_b64 = image_to_base64(image, output_format, output_quality)
            images_b64.append(img_b64)
        except Exception as e:
            print(f"Error converting image {i} to base64: {e}")
            continue

    if len(images_b64) == 0:
        return {"error": "Failed to process any images"}

    return {
        "images": images_b64,
        "seed": seed,
        "prompt": prompt,
        "num_outputs": len(images_b64)
    }

def handler(job):
    """RunPod serverless handler function"""
    try:
        job_input = job.get("input", {})
        result = generate_images(job_input)
        
        if "error" in result:
            return {"error": result["error"]}
        
        return result
        
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

# Initialize RunPod serverless
if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
    