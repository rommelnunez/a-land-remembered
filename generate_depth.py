#!/usr/bin/env python3
"""
Generate depth map using Depth Anything V2.
Outputs a grayscale depth map image for use with the parallax viewer.

Usage:
    python generate_depth.py landscape.png
    python generate_depth.py landscape.png --output depth.png
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate depth map using Depth Anything V2')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--output', '-o', help='Output depth map path (default: <input>_depth.png)')
    args = parser.parse_args()

    input_path = Path(args.image)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_depth.png")

    # Check dependencies
    try:
        import torch
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install torch torchvision opencv-python")
        sys.exit(1)

    print(f"Loading Depth Anything V2 model...")

    # Load model from torch hub
    try:
        model = torch.hub.load('huggingface/pytorch-image-models', 'timm/depth_anything_vitl14.dav2', pretrained=True, trust_repo=True)
    except Exception:
        # Fallback: use transformers pipeline (more reliable)
        try:
            from transformers import pipeline
            print("Using transformers pipeline...")
            pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

            from PIL import Image
            image = Image.open(input_path).convert('RGB')

            print(f"Processing {input_path}...")
            result = pipe(image)
            depth = result["depth"]

            # Convert to numpy and normalize
            depth_array = np.array(depth)
            depth_normalized = ((depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)

            # Save depth map
            cv2.imwrite(str(output_path), depth_normalized)
            print(f"✓ Depth map saved to: {output_path}")
            return

        except ImportError:
            print("\nInstall transformers for Depth Anything V2:")
            print("  pip install transformers torch torchvision opencv-python pillow")
            sys.exit(1)

    # If torch hub worked
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        model = model.to('mps')
        print("Using Apple MPS")
    else:
        print("Using CPU")

    # Load and preprocess image
    print(f"Processing {input_path}...")
    img = cv2.imread(str(input_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize and convert to tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img_rgb).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    elif torch.backends.mps.is_available():
        input_tensor = input_tensor.to('mps')

    # Run inference
    with torch.no_grad():
        depth = model(input_tensor)

    # Post-process
    depth = depth.squeeze().cpu().numpy()
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)

    # Resize to match input if needed
    if depth_normalized.shape[:2] != img.shape[:2]:
        depth_normalized = cv2.resize(depth_normalized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Save
    cv2.imwrite(str(output_path), depth_normalized)
    print(f"✓ Depth map saved to: {output_path}")

if __name__ == '__main__':
    main()
