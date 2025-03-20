import numpy as np
from PIL import Image
from scipy import ndimage

def apply_filter(image):
    kernel = np.array([[1,1,1],
                       [1,15,1],
                       [1,1,1]], dtype=np.float32)
    kernel /= kernel.sum()

    image = image.astype(np.float32)

    padded = np.pad(image, 1, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.float32)

    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            region = padded[i:i+3, j:j+3]
            filtered[i, j] = np.sum(region * kernel)
    return filtered

def circle_coords(angle_shift, angle_range, count, radius=1, center=(0, 0)):
    angles = np.linspace(0, angle_range, count) + angle_shift
    cx, cy = center
    x = radius * np.cos(angles) - cx
    y = radius * np.sin(angles) - cy
    points = np.array(list(zip(x, y)))
    return np.floor(points).astype(int)

def detector_coords(alpha, angle_range, count, radius=1, center=(0,0)):
    return circle_coords(np.radians(alpha - angle_range/2), np.radians(angle_range), count, radius, center)

def emitter_coords(alpha, angle_range, count, radius=1, center=(0,0)):
    return circle_coords(np.radians(alpha - angle_range/2 + 180), np.radians(angle_range), count, radius, center)[::-1]

def bresenham(x0, y0, x1, y1):
    if abs(y1 - y0) > abs(x1 - x0):
        swapped = True
        x0, y0, x1, y1 = y0, x0, y1, x1
    else:
        swapped = False
    m = (y1 - y0) / (x1 - x0) if x1 - x0 != 0 else 1
    q = y0 - m * x0
    if x0 < x1:
        xs = np.arange(np.floor(x0), np.ceil(x1) + 1, +1, dtype=int)
    else:
        xs = np.arange(np.ceil(x0), np.floor(x1) - 1, -1, dtype=int)
    ys = np.round(m * xs + q).astype(int)
    if swapped:
        xs, ys = ys, xs
    return np.array([xs, ys])

def draw_lines(emitters, detectors):
    lines = list()
    for (x0, y0), (x1, y1) in zip(emitters, detectors):
        lines.append(np.array(bresenham(x0, y0, x1, y1)))
    return lines

def image_pad(array):
    w, h = array.shape
    side = int(np.ceil((w**2 + h**2)**0.5))
    shape = (side, side)
    pad = (np.array(shape) - np.array(array.shape)) / 2
    pad = np.array([np.floor(pad), np.ceil(pad)]).T.astype(int)
    return np.pad(array, pad)

def rescale(array):
    res = array.astype('float32')
    res -= np.min(res)
    max_val = np.max(res)
    if max_val > 0:
        res /= max_val
    return res * 255

def unpad(img, height, width):
    y, x = img.shape
    startx = x//2 - (width//2)
    starty = y//2 - (height//2)
    return img[starty:starty+height, startx:startx+width]

def radon(detector_count, angle_range, image, radius, center, alpha):
    emitters = emitter_coords(alpha, angle_range, detector_count, radius, center)
    detectors = detector_coords(alpha, angle_range, detector_count, radius, center)
    lines = draw_lines(emitters, detectors)
    result = rescale(np.array([np.sum(image[tuple(line)]) for line in lines]))
    return result

def radon_all(image, scan_count, detector_count, angle_range):
    image = image_pad(image)
    center = np.floor(np.array(image.shape) / 2).astype(int)
    width = height = image.shape[0]
    radius = width // 2
    alphas = np.linspace(0, 180, scan_count)
    results = np.zeros((scan_count, detector_count))
    
    for i, alpha in enumerate(alphas):
        results[i] = radon(detector_count, angle_range, image, radius, center, alpha)
    
    return np.swapaxes(results, 0, 1)

def inverse_radon(image, num_of_lines, single_alpha_sinogram, alpha, detector_count, angle_range, radius, center):
    emitters = emitter_coords(alpha, angle_range, detector_count, radius, center)
    detectors = detector_coords(alpha, angle_range, detector_count, radius, center)
    lines = draw_lines(emitters, detectors)
    for i, line in enumerate(lines):
        image[tuple(line)] += single_alpha_sinogram[i]
        num_of_lines[tuple(line)] += 1

def inverse_radon_all(shape, sinogram, angle_range, use_filter=False):
    number_of_detectors, number_of_scans = sinogram.shape
    sinogram = np.swapaxes(sinogram, 0, 1)
    
    result = np.zeros(shape)
    result = image_pad(result)
    num_of_lines = np.zeros(result.shape)
    
    center = np.floor(np.array(result.shape) / 2).astype(int)
    width = height = result.shape[0]
    radius = width // 2
    alphas = np.linspace(0, 180, number_of_scans)
    
    for i, alpha in enumerate(alphas):
        inverse_radon(result, num_of_lines, sinogram[i], alpha, number_of_detectors, angle_range, radius, center)
    
    # Unikaj dzielenia przez zero
    num_of_lines[num_of_lines == 0] = 1
    temp = result / num_of_lines
    if use_filter:
        temp = apply_filter(temp)
    temp = rescale(temp)
    temp = unpad(temp, *shape)
    return temp

def calculate_rmse(original, reconstructed):
    orig_norm = original / np.max(original)
    recon_norm = reconstructed / np.max(reconstructed)
    mse = np.mean((orig_norm - recon_norm) ** 2)
    rmse = np.sqrt(mse)
    return rmse