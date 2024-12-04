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

"""
OUR CODE:
"""

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
    return np.concatenate([result_1, result_2])

def inverse_fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = inverse_fft(x[::2])
    odd = inverse_fft(x[1::2])
    factor = [np.exp(2j * np.pi * i / N) for i in range(N)]

    result_1 = [(even[i] + factor[i] * odd[i])/2 for i in range(N//2)]
    result_2 = [(even[i] + factor[i+N//2] * odd[i])/2 for i in range(N//2)]
    return np.concatenate([result_1, result_2])

def twod_fft(image):
    fft_row = np.array([fft(row) for row in image])
    fft_col = np.transpose(np.array([fft(col) for col in np.transpose(fft_row)]))
    return fft_col

def twod_dft(image):
    dft_row = np.array([dft(row) for row in image])
    dft_col = np.transpose(np.array([dft(col) for col in np.transpose(dft_row)]))
    return dft_col

def inverse_2d_fft(image):
    fft_row = np.array([inverse_fft(row) for row in image])
    fft_col = np.transpose(np.array([inverse_fft(col) for col in np.transpose(fft_row)]))
    return fft_col

def denoise_image(image, cutoff_x, cutoff_y):
    fft_result = twod_fft(image)
    rows, cols = fft_result.shape
    for i in range(rows):
        for j in range(cols):
           if i > cutoff_x or j > cutoff_y:
                fft_result[i, j] = 0

    denoised = np.abs(inverse_2d_fft(fft_result))

    non_zero_coefficients = cutoff_x * cutoff_y
    total_coefficients = rows * cols

    fraction = non_zero_coefficients / total_coefficients
    print(f"Number of non-zero Fourier coefficients: {non_zero_coefficients}")
    print(f"Fraction of original Fourier coefficients: {fraction:.4f}")
    
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

def measure_runtime(method, image):
    runtimes = []
    for _ in range(10): # 10 trials
        start_time = time.time()
        method(image)
        runtimes.append(time.time() - start_time)
    return np.mean(runtimes), np.std(runtimes)

def plot():
    sizes = [2**i for i in range(5, 8)]
    mean_runtimes_naive = []
    std_runtimes_naive = []
    mean_runtimes_fft = []
    std_runtimes_fft = []

    for size in sizes:
        image = np.random.rand(size, size).astype(np.float32)
        
        mean_naive, std_naive = measure_runtime(twod_dft, image)
        mean_runtimes_naive.append(mean_naive)
        std_runtimes_naive.append(std_naive)
        
        mean_fft, std_fft = measure_runtime(twod_fft, image)
        mean_runtimes_fft.append(mean_fft)
        std_runtimes_fft.append(std_fft)

    # Error bars (97% confidence interval, 2 * std deviation)
    error_bars_naive = [2 * std for std in std_runtimes_naive]
    error_bars_fft = [2 * std for std in std_runtimes_fft]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, mean_runtimes_naive, label='Naive Method', marker='o', color='b')
    plt.fill_between(sizes, 
                    np.array(mean_runtimes_naive) - np.array(error_bars_naive), 
                    np.array(mean_runtimes_naive) + np.array(error_bars_naive), 
                    color='b', alpha=0.2)

    plt.plot(sizes, mean_runtimes_fft, label='FFT Method', marker='o', color='r')
    plt.fill_between(sizes, 
                    np.array(mean_runtimes_fft) - np.array(error_bars_fft), 
                    np.array(mean_runtimes_fft) + np.array(error_bars_fft), 
                    color='r', alpha=0.2)
    plt.xlabel('Problem Size (Image Dimensions)')
    plt.ylabel('Mean Runtime (Seconds)')
    plt.title('Runtime Comparison: Naive Method vs. FFT-based Denoising')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    parameters = set_arguments(sys.argv)
    print(parameters)
    if parameters is not None:
        img = cv.imread(parameters['image'], cv.IMREAD_GRAYSCALE)
        if parameters['mode'] == 1:
            process_image(img)
        elif parameters['mode'] == 2:
            denoise_image(img, 300, 250)
        elif parameters['mode'] == 3:
            compress_image(img, 1000)
        elif parameters['mode'] == 4:
            plot()
    print("done")
