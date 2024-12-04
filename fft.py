import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import math
import time

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
        else:
            print(f"ERROR\tIncorrect input syntax: unknown argument {args[i]}")
            return None
    return parameters

"""
GOAL OUTPUT:
"""
def fft_image(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift)+ 1)
 
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks(), plt.yticks()
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray', norm=LogNorm(vmin=1))
    plt.title('Magnitude Spectrum'), plt.xticks(), plt.yticks()
    plt.colorbar()
    plt.show()

def npfft_image(image):
    f = np.fft.fft2(image)
    fft_magnitude = np.log(np.abs(f) + 1)
 
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(fft_magnitude, cmap = 'gray', norm=LogNorm(vmin=1))
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

def dft(signal):
    N = len(signal)
    power = math.ceil(np.log2(N))
    N_final = 2 ** power
    signal = np.append(signal, np.zeros(N_final - N))
    N = N_final

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
    power = math.ceil(np.log2(N))
    N_final = 2 ** power
    signal = np.append(signal, np.zeros(N_final - N))
    N = N_final

    inverse_dft_result = np.zeros(N, dtype=complex)
    for n in range(N):
        result = 0
        for k in range(N):  
            angle = 1j * 2 * np.pi * k * n / N
            result += signal[k] * np.exp(angle)
        inverse_dft_result[n] = result / N
    return inverse_dft_result

# Unused
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

def fft(x):
    N = len(x)
    power = math.ceil(np.log2(N))
    N_final = 2 ** power
    x = np.append(x, np.zeros(N_final - N))
    N = N_final
    if N <= 1:
        return x
    even = fft(x[::2])
    odd = fft(x[1::2])
    factor = [np.exp(-2j * np.pi * i / N) for i in range(N)]

    result_1 = [even[i] + factor[i] * odd[i] for i in range(N//2)]
    result_2 = [even[i] + factor[i+N//2] * odd[i] for i in range(N//2)]
    # return np.concatenate([even + factor[:N//2] * odd, even + factor[N//2:] * odd])
    return np.concatenate([result_1, result_2])

def twod_fft(image):
    print('start fft_row')
    fft_row = np.array([fft(row) for row in image])
    print('done fft_row')
    fft_col = np.transpose(np.array([fft(col) for col in np.transpose(fft_row)]))
    return fft_col

def twod_dft(image):
    print('start dft_row')
    dft_row = np.array([dft(row) for row in image])
    print('done dft_row')
    dft_col = np.transpose(np.array([dft(col) for col in np.transpose(dft_row)]))
    return dft_col

def inverse_fft(ft):
    N = len(ft)
    ft_conj = np.conj(ft)
    result = fft(ft_conj)
    return np.conj(result) / N

def inverse_2d_fft(image):
    # return np.fft.ifft2(np.fft.ifftshift(image))
    fft_row = np.array([inverse_fft(row) for row in image])
    fft_col = np.array([inverse_fft(col) for col in fft_row.T]).T
    return fft_col

def fft2d(image):
    """ Perform 2D FFT and shift the result to center low frequencies. """
    return np.fft.fftshift(np.fft.fft2(image))

def denoise_image(image, cutoff_frequency):
    fft_result = twod_fft(image)

    rows, cols = fft_result.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if dist >= cutoff_frequency:
                mask[i, j] = 1

    fft_result_filtered = fft_result * mask
    
    non_zero_coefficients = np.count_nonzero(mask)
    total_coefficients = image.size
    fraction = non_zero_coefficients / total_coefficients
    print(f"Number of non-zero Fourier coefficients: {non_zero_coefficients}")
    print(f"Fraction of original Fourier coefficients: {fraction:.4f}")

    denoised = np.abs(inverse_2d_fft(fft_result_filtered))
    
    plt.subplot(121),plt.imshow(image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks(), plt.yticks()
    plt.subplot(122),plt.imshow(denoised, cmap = 'gray')
    plt.title('Denoised Image'), plt.xticks(), plt.yticks()
    plt.show()

def compress_image(image, compression_percent):
    fft_result = twod_fft(image)
    magnitude = np.abs(fft_result)
    num_coefficients_to_zero = int(fft_result.size * (1 - compression_percent / 100))
    flat_indices = np.argsort(magnitude.flatten())
    fft_result.flatten()[flat_indices[:num_coefficients_to_zero]] = 0
    compressed = np.abs(inverse_2d_fft(fft_result))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks(), plt.yticks()
    plt.subplot(122),plt.imshow(compressed, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks(), plt.yticks()
    plt.show()

def process_image(img):
    print('start process img')
    f = twod_fft(img)
    print('done process img')
    fft_magnitude = np.log(np.abs(f) + 1)

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks(), plt.yticks()
    plt.subplot(122),plt.imshow(fft_magnitude, cmap = 'gray', norm=LogNorm(vmin=1))
    plt.title('Magnitude Spectrum'), plt.xticks(), plt.yticks()
    plt.colorbar()
    plt.show()

def plot_runtime():
    sizes = [2**i for i in range(5, 8)]
    naive_times = []
    fft_times = []
    npfft_times = []

    for size in sizes:
        random_image = np.random.random((size, size))
        print(random_image)
        
        # Measure runtime for DFT
        start = time.time()
        twod_dft(random_image)
        naive_times.append(time.time() - start)

        # Measure runtime for FFT
        start = time.time()
        twod_fft(random_image)
        fft_times.append(time.time() - start)

        # Measure runtime for Numpy FFT
        start = time.time()
        np.fft.fft2(random_image)
        npfft_times.append(time.time() - start)

    # Plot the runtime comparison
    plt.plot(sizes, naive_times, label='DFT')
    plt.plot(sizes, fft_times, label='FFT')
    # plt.plot(sizes, npfft_times, label='Numpy FFT')
    plt.xlabel('Problem Size (N x N)')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parameters = set_arguments(sys.argv)
    print(parameters)
    if parameters is not None:
        img = cv.imread(parameters['image'], cv.IMREAD_GRAYSCALE)
        if parameters['mode'] == 1:
            process_image(img, 1)
        elif parameters['mode'] == 2:
            denoise_image(img, 500) # 200 was good when using built in fft
        elif parameters['mode'] == 3:
            compress_image(img, 1000)
        elif parameters['mode'] == 4:
            plot_runtime()
    print("done")
