# CFSM (Content and Style Fusion Model)

## Overview
The CFSM (Content and Style Fusion Model) is a neural network architecture designed for image style transfer. It integrates content encoding, style decoding, and adversarial training through a discriminator to achieve high-quality image generation with diverse styles.

## Features
- **Image Encoder**: Extracts content features from input images.
- **Style Decoder**: Applies learned styles to the encoded content.
- **Discriminator**: Evaluates the authenticity of generated images.
- **Linear Subspace**: Samples styles to guide the generation process.

## Installation
Make sure you have Python and NumPy installed. You can install NumPy using pip:

```bash
pip install numpy

# Usage

##Example Usage


import numpy as np

# Initialize CFSM model
model = CFSM(img_size=112, style_dim=128, n_bases=10)

# Example input image (dummy data)
input_image = np.random.rand(1, 3, 112, 112)  # Batch size of 1, RGB image

# Forward pass
output_image, style = model.forward(input_image)

# Compute losses (dummy data for real and fake images)
real_image = np.random.rand(1, 3, 112, 112)  # Batch size of 1, RGB image
losses = model.compute_losses(real_image, output_image, style)

# Print losses
print(losses)

