"""
03_gate_detection_synthetic.py — Generate synthetic gate detection training data.

Creates training images with colored racing gate rectangles on random backgrounds,
with bounding box labels in YOLO format. This is prep for training a YOLOv8
gate detector.

Output:
  data/synthetic_gates/images/train/  — PNG images
  data/synthetic_gates/labels/train/  — YOLO format label files
  data/synthetic_gates/data.yaml      — dataset config for Ultralytics

Run: python scripts/03_gate_detection_synthetic.py
"""

import numpy as np
import cv2
from pathlib import Path
import yaml


def random_background(h, w, rng):
    """Generate a random background image."""
    bg_type = rng.choice(['noise', 'gradient', 'solid', 'sky'])
    
    if bg_type == 'noise':
        bg = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        # Smooth it a bit for realism
        bg = cv2.GaussianBlur(bg, (15, 15), 0)
    
    elif bg_type == 'gradient':
        # Vertical gradient (sky-like)
        top_color = rng.integers(100, 255, 3)
        bottom_color = rng.integers(0, 150, 3)
        gradient = np.linspace(top_color, bottom_color, h).astype(np.uint8)
        bg = np.tile(gradient[:, np.newaxis, :], (1, w, 1))
        # Add some noise
        noise = rng.integers(-20, 20, bg.shape)
        bg = np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)
    
    elif bg_type == 'solid':
        color = rng.integers(0, 255, 3).tolist()
        bg = np.full((h, w, 3), color, dtype=np.uint8)
        noise = rng.integers(-15, 15, bg.shape)
        bg = np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)
    
    else:  # sky
        # Blue-ish sky gradient
        top = np.array([200 + rng.integers(-30, 30), 
                        150 + rng.integers(-30, 30), 
                        50 + rng.integers(-20, 20)])
        bottom = np.array([100 + rng.integers(-30, 30),
                          180 + rng.integers(-30, 30),
                          100 + rng.integers(-30, 30)])
        gradient = np.linspace(top, bottom, h).astype(np.uint8)
        bg = np.tile(gradient[:, np.newaxis, :], (1, w, 1))
        noise = rng.integers(-10, 10, bg.shape)
        bg = np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return bg


def draw_gate(img, cx, cy, gate_w, gate_h, color, thickness, rng, angle=0):
    """Draw a racing gate rectangle (hollow) with optional rotation.
    
    Returns bounding box [x_min, y_min, x_max, y_max] in pixels.
    """
    h, w = img.shape[:2]
    
    # Gate corners (before rotation)
    half_w = gate_w / 2
    half_h = gate_h / 2
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h],
    ], dtype=np.float32)
    
    # Inner corners (hollow gate)
    inner_half_w = max(half_w - thickness, thickness)
    inner_half_h = max(half_h - thickness, thickness)
    inner_corners = np.array([
        [-inner_half_w, -inner_half_h],
        [inner_half_w, -inner_half_h],
        [inner_half_w, inner_half_h],
        [-inner_half_w, inner_half_h],
    ], dtype=np.float32)
    
    # Rotation
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    corners = (rot @ corners.T).T + np.array([cx, cy])
    inner_corners = (rot @ inner_corners.T).T + np.array([cx, cy])
    
    # Draw outer and inner polygons
    outer_pts = corners.astype(np.int32)
    inner_pts = inner_corners.astype(np.int32)
    
    # Draw filled outer, then cut out inner (creates hollow gate)
    overlay = img.copy()
    cv2.fillPoly(overlay, [outer_pts], color)
    cv2.fillPoly(overlay, [inner_pts], (0, 0, 0))  # Placeholder
    
    # Blend: use the overlay for outer, original for inner
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [outer_pts], 255)
    cv2.fillPoly(mask, [inner_pts], 0)
    
    mask_3ch = mask[:, :, np.newaxis] / 255.0
    result = (overlay * mask_3ch + img * (1 - mask_3ch)).astype(np.uint8)
    np.copyto(img, result)
    
    # Compute bounding box
    all_pts = corners
    x_min = max(0, int(all_pts[:, 0].min()))
    y_min = max(0, int(all_pts[:, 1].min()))
    x_max = min(w, int(all_pts[:, 0].max()))
    y_max = min(h, int(all_pts[:, 1].max()))
    
    return [x_min, y_min, x_max, y_max]


def bbox_to_yolo(bbox, img_w, img_h):
    """Convert [x_min, y_min, x_max, y_max] to YOLO format [cx, cy, w, h] normalized."""
    x_min, y_min, x_max, y_max = bbox
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return [cx, cy, w, h]


# Gate color palettes (BGR for OpenCV)
GATE_COLORS = [
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 165, 255),    # Orange
    (255, 255, 0),    # Cyan
    (255, 255, 255),  # White
]


def generate_dataset(num_images=500, img_size=(640, 640), max_gates_per_image=3, seed=42):
    """Generate synthetic gate detection dataset."""
    
    rng = np.random.default_rng(seed)
    
    # Output directories
    base_dir = Path(__file__).parent.parent / "data" / "synthetic_gates"
    img_dir = base_dir / "images" / "train"
    label_dir = base_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    h, w = img_size
    
    print(f"\n[*] Generating {num_images} synthetic gate images...")
    print(f"    Image size: {w}x{h}")
    print(f"    Output: {base_dir}")
    
    for i in range(num_images):
        # Generate background
        img = random_background(h, w, rng)
        
        # Number of gates in this image
        num_gates = rng.integers(1, max_gates_per_image + 1)
        labels = []
        
        for _ in range(num_gates):
            # Random gate parameters
            gate_w = rng.integers(60, 300)
            gate_h = rng.integers(60, 300)
            thickness = rng.integers(8, max(10, gate_w // 4))
            
            # Position (ensure gate is mostly visible)
            margin = 30
            cx = rng.integers(margin + gate_w//2, w - margin - gate_w//2)
            cy = rng.integers(margin + gate_h//2, h - margin - gate_h//2)
            
            color = GATE_COLORS[rng.integers(0, len(GATE_COLORS))]
            angle = rng.uniform(-15, 15)  # Slight rotation
            
            bbox = draw_gate(img, cx, cy, gate_w, gate_h, color, thickness, rng, angle)
            
            # Convert to YOLO format (class 0 = gate)
            yolo_bbox = bbox_to_yolo(bbox, w, h)
            labels.append(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
        
        # Save image
        img_path = img_dir / f"gate_{i:05d}.png"
        cv2.imwrite(str(img_path), img)
        
        # Save label
        label_path = label_dir / f"gate_{i:05d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        if (i + 1) % 100 == 0:
            print(f"    Generated {i+1}/{num_images} images")
    
    # Create data.yaml for Ultralytics
    data_config = {
        'path': str(base_dir.resolve()),
        'train': 'images/train',
        'val': 'images/train',  # Using same for now; split later
        'names': {0: 'gate'},
        'nc': 1,
    }
    
    yaml_path = base_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"\n[*] Dataset generated!")
    print(f"    Images: {img_dir}")
    print(f"    Labels: {label_dir}")
    print(f"    Config: {yaml_path}")
    print(f"    Total images: {num_images}")
    
    # Show a sample
    print(f"\n[*] Generating sample grid...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Synthetic Gate Detection — Sample Images", fontsize=14, fontweight='bold')
    
    sample_indices = rng.choice(num_images, min(8, num_images), replace=False)
    for idx, ax in zip(sample_indices, axes.flat):
        img_path = img_dir / f"gate_{idx:05d}.png"
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes
        label_path = label_dir / f"gate_{idx:05d}.txt"
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cx_n, cy_n, w_n, h_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx_n - w_n/2) * w)
                y1 = int((cy_n - h_n/2) * h)
                x2 = int((cx_n + w_n/2) * w)
                y2 = int((cy_n + h_n/2) * h)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        ax.imshow(img_rgb)
        ax.set_title(f"Image {idx}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    sample_path = base_dir / "sample_grid.png"
    plt.savefig(sample_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    Saved sample grid to: {sample_path}")


# Need matplotlib for the sample grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("  404 Pilots Not Found — Synthetic Gate Data Generator")
    print("  AI Grand Prix Prep")
    print("=" * 60)
    
    generate_dataset(
        num_images=500,
        img_size=(640, 640),
        max_gates_per_image=3,
        seed=42,
    )
    
    print(f"\n[*] Done! Ready for YOLOv8 training:")
    print(f"    yolo detect train data=data/synthetic_gates/data.yaml model=yolov8n.pt epochs=50")


if __name__ == "__main__":
    main()
