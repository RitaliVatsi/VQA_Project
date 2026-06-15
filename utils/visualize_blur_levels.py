
import os
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import matplotlib.pyplot as plt

DATA_PATH = Path("Github Repo/DATA")

def create_blur_grid():
    # Find a good text image
    images = list(DATA_PATH.glob("*.jpg"))
    if not images:
        print("No images found!")
        return
        
    # Pick a few random samples
    sample_imgs = random.sample(images, 3) 
    
    sigmas = [5, 4, 3, 2]
    rows = len(sample_imgs)
    cols = len(sigmas)
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    
    for r, img_path in enumerate(sample_imgs):
        orig = Image.open(img_path).convert("RGB")
        
        for c, sig in enumerate(sigmas):
            blurred = orig.filter(ImageFilter.GaussianBlur(radius=sig))
            
            ax = axes[r][c] if rows > 1 else axes[c]
            ax.imshow(blurred)
            ax.set_title(f"Sigma = {sig}", fontsize=14)
            ax.axis('off')
            
    plt.tight_layout()
    out_path = "blur_comparison_grid.jpg"
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison to {out_path}")

if __name__ == "__main__":
    create_blur_grid()
