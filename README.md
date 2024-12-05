# Fast Fourier Transform and Applications 
ECSE 316 | Assignment 2 | Julien Heng, Sophia Li

## Run the Program
This program was written and tested with Python version 3.13.0.

### How to Run

The program should be invoked at the command line. The command is structured as follows:
```
python3 fft.py [-m mode] [-i image]
```
- `mode (optional)`
    - [1] (default) Fast Mode: converts an image to FFT and diplays it
    - [2] Denoise: The image is denoise by performing FFT and truncating high frequencies
    - [3] Compression: Compresses the image and plots it 
    - [4] Plot: plots runtime of algorithms 
- `image (optional)` filename of the image on which the transforms are performed on 
```