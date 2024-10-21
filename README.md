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
```
#Classes

##CFSM

##Parameters:
 -img_size (int): Size of the input images (default: 112).
 -style_dim (int): Dimension of the style representation (default: 128).
 -n_bases (int): Number of bases for the linear subspace (default: 10).
 -l_a, u_a (float): Lower and upper bounds for the style norms (default: 0, 6).
 -l_m, u_m (float): Lower and upper margins for the identity loss (default: 0.05, 0.65).

###ImageEncoder
 - Purpose: Encodes input images to extract content features.
###StyleDecoder
 - Purpose: Decodes the content features and applies styles to generate output images.
###Discriminator
 - Purpose: Discriminates between real and generated images to train the generator.
###LinearSubspace
 - Purpose: Samples styles from a learned linear subspace to guide the style generation.
###MLP (Multi-Layer Perceptron)
 - Purpose: A basic fully connected network used in the style decoder.

