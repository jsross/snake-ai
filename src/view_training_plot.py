#!/usr/bin/env python3
"""
Simple script to view training progress plots saved during training.
Run this after training to see the visualization without risking crashes.
"""

import matplotlib.pyplot as plt
import os

def view_training_plot(models_dir="models"):
    """Load and display the saved training progress plot."""
    plot_path = f"{models_dir}/training_progress.png"
    
    if os.path.exists(plot_path):
        from PIL import Image
        img = Image.open(plot_path)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Training Progress")
        plt.tight_layout()
        plt.show()
        print(f"Displayed plot from: {plot_path}")
    else:
        print(f"No training plot found at: {plot_path}")
        print("Run some training first to generate a plot.")

if __name__ == "__main__":
    view_training_plot()
