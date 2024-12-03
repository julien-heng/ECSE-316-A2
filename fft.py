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
        result = 0
        for n in range(N):  
            angle = -1j * 2 * np.pi * k * n / N
            result += signal[n] * np.exp(angle)
        dft_result[k] = result
    return dft_result

def inverse_dft(signal):
    N = len(signal)
    inverse_dft_result = np.zeros(N, dtype=complex)
    for n in range(N):
        result = 0
        for k in range(N):  
            angle = 1j * 2 * np.pi * k * n / N
            result += signal[k] * np.exp(angle)
        inverse_dft_result[n] = result / N
    return inverse_dft_result

def fft_aux(signal, k):
    N = len(signal)
    if N <= 4:
        result = 0
        for m in range(N):
            angle = -1j * 2 * np.pi * k * m / N
            result += signal[m] * np.exp(angle)
        return result
    else:
        even = [signal[i] for i in range(0, N, 2)]
        odd = [signal[i] for i in range(1, N, 2)]
        return fft_aux(even, k) + fft_aux(odd, k) * np.exp(-1j * 2 * np.pi * k / N)

def fft(signal):
    N = len(signal)
    power = math.ceil(np.log2(N))
    N_final = 2 ** power
    signal = np.append(signal, np.zeros(N_final - N))
    N = N_final

    fft_result = np.zeros(N_final, dtype=complex)
    for k in range(N):
        fft_result[k] = fft_aux(signal, k)
    return fft_result

def twod_fft(image):
    fft_row = np.array([fft(row) for row in image])
    fft_col = np.array([fft(col) for col in fft_row.T]).T
    return fft_col

## 
def two_d_fft(image):
    N = len(image)
    M = len(image[0])
    fft_result = np.zeros((N, M), dtype=complex)
    for i in range(N):
        fft_result[i] = fft(image[i])
    for j in range(M):
        fft_result[:, j] = fft(fft_result[:, j])
    return fft_result


# TODO
def inverse_fft(ft):
    N = len(ft)
    return ""

def process_image(image):
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    #print(img[0])
    print(len(img[0]))
    print(fft(img[0]))

    """
    dft_result = fft(img[0])
    
    print(dft_result)
    print(np.fft.fft(img[0]))
    #print(fft(img[0]))
    """
    """
    f = twod_fft(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
 
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    """

if __name__ == "__main__":
    
    parameters = set_arguments(sys.argv)
    process_image(parameters['image'])
    
    #fft_image(parameters['image'])
    #print("done")


