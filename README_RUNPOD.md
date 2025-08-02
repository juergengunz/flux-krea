# FLUX.1-Krea-dev RunPod Serverless Worker

This repository contains a RunPod serverless worker for the FLUX.1-Krea-dev text-to-image model with LoRA support.

## Features

- **Text-to-Image Generation**: High-quality image generation using FLUX.1-Krea-dev
- **Image-to-Image**: Support for img2img workflows
- **LoRA Support**: Load LoRAs from HuggingFace, Replicate, CivitAI, or direct URLs
- **Multiple Aspect Ratios**: Support for various aspect ratios
- **Base64 Output**: Images returned as base64-encoded strings
- **Batch Generation**: Generate up to 4 images per request

## Quick Start

### 1. Deploy to RunPod

1. **Build the Docker image:**
   ```bash
   docker build -t flux-krea-runpod .
   ```

2. **Push to a registry** (Docker Hub, GHCR, etc.)

3. **Create a RunPod Serverless endpoint:**
   - Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
   - Create a new endpoint
   - Use your Docker image
   - Set minimum GPU memory to at least 24GB (A100 40GB recommended)
   - Configure auto-scaling as needed

### 2. API Usage

#### Basic Text-to-Image

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# RunPod endpoint URL
endpoint_url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"

# API request
payload = {
    "input": {
        "prompt": "A beautiful landscape with mountains and a lake",
        "aspect_ratio": "16:9",
        "num_outputs": 1,
        "num_inference_steps": 28,
        "guidance_scale": 3.5
    }
}

headers = {
    "Authorization": "Bearer YOUR_RUNPOD_API_KEY",
    "Content-Type": "application/json"
}

response = requests.post(endpoint_url, json=payload, headers=headers)
result = response.json()

# Decode base64 image
if "output" in result and "images" in result["output"]:
    image_b64 = result["output"]["images"][0]
    # Remove data URL prefix if present
    if image_b64.startswith('data:image/'):
        image_b64 = image_b64.split(',')[1]
    
    image_data = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_data))
    image.save("output.webp")
```

#### Image-to-Image

```python
# Convert your input image to base64
with open("input_image.jpg", "rb") as f:
    input_image_b64 = base64.b64encode(f.read()).decode()

payload = {
    "input": {
        "prompt": "Transform this into a cyberpunk scene",
        "image": f"data:image/jpeg;base64,{input_image_b64}",
        "prompt_strength": 0.7,
        "num_inference_steps": 28
    }
}
```

#### Using LoRA

```python
payload = {
    "input": {
        "prompt": "A portrait in anime style",
        "hf_lora": "alvdansen/frosting_lane_flux",
        "lora_scale": 0.8,
        "num_inference_steps": 28
    }
}
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | Required | Text prompt for image generation |
| `aspect_ratio` | string | "1:1" | Image aspect ratio. Options: "1:1", "16:9", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4", "4:3", "9:16", "9:21" |
| `image` | string | null | Input image for img2img (URL or base64) |
| `prompt_strength` | float | 0.8 | Strength for img2img (0.0-1.0) |
| `num_outputs` | integer | 1 | Number of images to generate (1-4) |
| `num_inference_steps` | integer | 28 | Number of inference steps (1-50) |
| `guidance_scale` | float | 3.5 | Guidance scale (0.0-10.0) |
| `seed` | integer | random | Random seed for reproducibility |
| `output_format` | string | "webp" | Output format: "webp", "jpg", "png" |
| `output_quality` | integer | 80 | JPEG/WebP quality (0-100) |
| `hf_lora` | string | null | LoRA identifier (HF path, URL, etc.) |
| `lora_scale` | float | 0.8 | LoRA weight scale (0.0-1.0) |

## Output Format

```json
{
  "images": ["data:image/webp;base64,/9j/4AAQSkZJRgABAQAAAQ..."],
  "seed": 12345,
  "prompt": "Your prompt here",
  "num_outputs": 1
}
```

## Supported LoRA Sources

- **HuggingFace**: `username/model-name`
- **HuggingFace URL**: `https://huggingface.co/username/model-name/resolve/main/file.safetensors`
- **Replicate**: `https://replicate.delivery/.../trained_model.tar`
- **CivitAI**: `https://civitai.com/api/download/models/12345?type=Model&format=SafeTensor`
- **Direct SafeTensors URL**: Any URL ending in `.safetensors`

## Performance Recommendations

- **GPU**: A100 40GB or better for optimal performance
- **Memory**: At least 24GB VRAM required
- **Cold Start**: First request may take 30-60 seconds for model loading
- **Warm Requests**: Subsequent requests typically complete in 5-15 seconds

## Environment Variables

You can set these in your RunPod environment:

- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: 0)
- `TORCH_DTYPE`: Model precision (default: bfloat16)

## Error Handling

The worker returns detailed error messages for common issues:

- Invalid parameters (out of range values)
- Missing required fields
- Image download/processing failures
- LoRA loading errors
- Generation failures

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires CUDA GPU)
python predict.py
```

## Troubleshooting

1. **Out of Memory**: Reduce `num_outputs` or use a smaller image size
2. **LoRA Loading Fails**: Check URL accessibility and format
3. **Slow Performance**: Ensure using appropriate GPU instance
4. **Base64 Errors**: Check image encoding format

## License

This project follows the FLUX.1-dev license terms. See [LICENSE](LICENSE) for details. 