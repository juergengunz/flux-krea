#!/usr/bin/env python3
"""
Local test script for the RunPod FLUX worker
This allows you to test the worker locally before deploying to RunPod
"""

import json
import base64
from PIL import Image
from io import BytesIO
from predict import generate_images

def test_txt2img():
    """Test basic text-to-image generation"""
    print("Testing text-to-image generation...")
    
    job_input = {
        "prompt": "A beautiful sunset over mountains, highly detailed, 8k",
        "aspect_ratio": "16:9",
        "num_outputs": 1,
        "num_inference_steps": 20,  # Reduced for faster testing
        "guidance_scale": 3.5,
        "seed": 42
    }
    
    result = generate_images(job_input)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return False
    
    print(f"Generated {len(result['images'])} image(s)")
    print(f"Seed used: {result['seed']}")
    
    # Save the first image
    if result['images']:
        image_b64 = result['images'][0]
        if image_b64.startswith('data:image/'):
            image_b64 = image_b64.split(',')[1]
        
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        image.save("test_output_txt2img.webp")
        print("Saved test output to test_output_txt2img.webp")
    
    return True

def test_img2img():
    """Test image-to-image generation with a URL"""
    print("\nTesting image-to-image generation...")
    
    # You can replace this with any accessible image URL
    test_image_url = "https://picsum.photos/512/512"
    
    job_input = {
        "prompt": "Transform this into a cyberpunk cityscape, neon lights, futuristic",
        "image": test_image_url,
        "prompt_strength": 0.7,
        "num_outputs": 1,
        "num_inference_steps": 20,
        "guidance_scale": 3.5,
        "seed": 123
    }
    
    result = generate_images(job_input)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return False
    
    print(f"Generated {len(result['images'])} image(s)")
    print(f"Seed used: {result['seed']}")
    
    # Save the first image
    if result['images']:
        image_b64 = result['images'][0]
        if image_b64.startswith('data:image/'):
            image_b64 = image_b64.split(',')[1]
        
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        image.save("test_output_img2img.webp")
        print("Saved test output to test_output_img2img.webp")
    
    return True

def test_lora():
    """Test LoRA loading and generation"""
    print("\nTesting LoRA generation...")
    
    job_input = {
        "prompt": "A portrait of a person in anime style, highly detailed",
        "hf_lora": "alvdansen/frosting_lane_flux",  # Popular FLUX LoRA
        "lora_scale": 0.8,
        "aspect_ratio": "1:1",
        "num_outputs": 1,
        "num_inference_steps": 20,
        "guidance_scale": 3.5,
        "seed": 456
    }
    
    result = generate_images(job_input)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return False
    
    print(f"Generated {len(result['images'])} image(s)")
    print(f"Seed used: {result['seed']}")
    
    # Save the first image
    if result['images']:
        image_b64 = result['images'][0]
        if image_b64.startswith('data:image/'):
            image_b64 = image_b64.split(',')[1]
        
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        image.save("test_output_lora.webp")
        print("Saved test output to test_output_lora.webp")
    
    return True

def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")
    
    # Test with invalid parameters
    job_input = {
        "prompt": "",  # Empty prompt should cause error
        "num_outputs": 10,  # Too many outputs
        "guidance_scale": 15  # Invalid guidance scale
    }
    
    result = generate_images(job_input)
    
    if "error" in result:
        print(f"Expected error caught: {result['error']}")
        return True
    else:
        print("Error: Should have caught validation error")
        return False

if __name__ == "__main__":
    print("Starting local tests for FLUX RunPod worker...")
    print("=" * 50)
    
    tests = [
        ("Text-to-Image", test_txt2img),
        ("Image-to-Image", test_img2img),
        ("LoRA Generation", test_lora),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Worker is ready for deployment.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.") 