import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import math


def set_arguments(args):
    parameters = {
        'mode': 1,
        'image' : 'moonlanding.png'
    }
    i = 1
    while i < len(args):
        if args[i] == '-m':
            if i+1 < len(args) and args[i+1].isdigit():
                parameters['mode'] = int(args[i+1])
                i += 2
            else:
                print(f"ERROR\tIncorrect input syntax: expected integer after argument {args[i]}")
                return None
        elif args[i] == '-i':
            if i+1 < len(args):
                parameters['image'] = args[i+1]
                i += 2
            else:
                print(f"ERROR\tIncorrect input syntax: expected image after argument {args[i]}")
                return None   
    return parameters

"""
GOAL OUTPUT:
def fft_image(image):
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
 
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
"""

def fft_image(image):
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)

def dft(signal):
    n = len(signal)
    dft_result = np.zeros(n, dtype=complex)
    
    for k in range(n):  
        for i in range(n):  
            angle = -1j * 2 * np.pi * k * i / n
            dft_result[k] += signal[n] * np.exp(angle)
    
    return dft_result

def inverse_dft(signal):
    n = len(signal)
    inverse_dft_result = np.zeros(n, dtype=complex)
    
    for k in range(n):  
        for i in range(n):  
            angle = 1j * 2 * np.pi * k * i / n
            inverse_dft_result[k] += signal[n] * np.exp(angle) / n
    
    return inverse_dft_result




if __name__ == "__main__":
    
    parameters = set_arguments(sys.argv)
    fft_image('moonlanding.png')
