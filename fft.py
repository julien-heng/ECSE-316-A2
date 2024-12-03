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
"""
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

def dft(signal):
    N = len(signal)
    dft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):  
            angle = -1j * 2 * np.pi * k * n / N
            dft_result[n] = signal[n] * np.exp(angle)
    return dft_result

def inverse_dft(signal):
    N = len(signal)
    inverse_dft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        for i in range(N):  
            angle = 1j * 2 * np.pi * k * i / N
            inverse_dft_result[k] = signal[k] * np.exp(angle) / N
    return inverse_dft_result

def fft_aux(signal, k):
    N = len(signal)
    if N == 1:  # Base case: single element
        return signal[0]
    else:
        # Split into even and odd parts
        even = fft_aux(signal[::2], k)
        odd = fft_aux(signal[1::2], k)
        
        # The factor for the odd elements, applying the FFT combination formula
        factor = np.exp(-2j * np.pi * k / N)
        
        # Combine even and odd results
        return even + factor * odd
    """
    N = len(signal)
    if N < 4:
        result = 0
        for n in range(N):
            angle = -1j * 2 * np.pi * k * n / N
            result += signal[n] * np.exp(angle)
        return result
    else:
        even = [signal[i] for i in range(0, N, 2)]
        odd = [signal[i] for i in range(1, N, 2)]
        return fft_aux(even, k) + fft_aux(odd, k) * np.exp(-1j * 2 * np.pi * k / N)
    """

def fft(signal):
    N = len(signal)
    fft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        fft_result[k] = fft_aux(signal, k)
    return fft_result

def twod_fft(image):
    fft_row = np.array([fft(row) for row in image])
    fft_col = np.array([fft(col) for col in fft_row.T]).T
    return fft_col

# TODO
def inverse_fft(ft):
    N = len(ft)
    return ""

def process_image(image):
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    f = twod_fft(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
 
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    parameters = set_arguments(sys.argv)
    process_image(parameters['image'])
    fft_image(parameters['image'])
    print("done")
