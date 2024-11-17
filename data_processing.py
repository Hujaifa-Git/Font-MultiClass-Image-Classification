import os
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

dataset_path = "F:\Multiclass_Image_Classification\dataset"

def log_dataset_statistics(dataset_path):
    class_distribution = Counter()
    image_dimensions = []
    avg_intensities = []
    
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            class_distribution[class_folder] = len(images)
            for image_file in images:
                image_path = os.path.join(class_path, image_file)
                try:
                    with Image.open(image_path) as img:
                        image_dimensions.append(img.size)  # (width, height)
                        avg_intensities.append( np.array( img.convert("L") ).mean() )
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    
    print("\nClass Distribution:")
    for font_class, count in class_distribution.items():
        print(f"  {font_class}: {count} images")
    
    plt.figure(figsize=(12, 6))
    plt.bar(class_distribution.keys(), class_distribution.values(), color="skyblue")
    plt.xticks(rotation=90)
    plt.title("Class Distribution")
    plt.ylabel("Number of Images")
    plt.xlabel("Font Classes")
    plt.tight_layout()
    plt.savefig("figs/class_distribution.png")
    plt.close()
    
    print("Class distribution plot saved as 'class_distribution.png'")
    

    widths = [dim[0] for dim in image_dimensions]
    heights = [dim[1] for dim in image_dimensions]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].hist(widths, bins=20, color="lightgreen", edgecolor="black")
    axs[0].set_title("Width Distribution")
    axs[0].set_xlabel("Width (pixels)")
    axs[0].set_ylabel("Count")
    
    axs[1].hist(heights, bins=20, color="lightcoral", edgecolor="black")
    axs[1].set_title("Height Distribution")
    axs[1].set_xlabel("Height (pixels)")
    axs[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("figs/image_dimensions_distribution.png")
    plt.close()
    
    print("Image dimensions distribution plot saved as 'image_dimensions_distribution.png'")

    plt.figure(figsize=(10, 6))
    plt.hist(avg_intensities, bins=30, color="purple", edgecolor="black")
    plt.title("Average Intensity Distribution")
    plt.xlabel("Average Intensity (0-255)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("figs/intensity_distribution.png")
    plt.close()
    print("Intensity distribution plot saved as 'intensity_distribution.png'")
    
    return class_distribution, image_dimensions

