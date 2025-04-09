import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from algorithms import radon_all, inverse_radon_all, calculate_rmse, apply_filter_to_sinogram, rescale
import sys

# Załaduj obraz testowy
def load_test_image(path):
    img = Image.open(path).convert('L')
    img = np.array(img)
    return img

# Eksperyment: zmiana parametru i logowanie RMSE
def run_experiment(image, param_name, param_values, default_params, use_filter=False):
    rmse_results = []

    for value in param_values:
        params = default_params.copy()
        params[param_name] = value

        # Generuj sinogram
        sinogram = radon_all(
            image,
            scan_count=params['scan_count'],
            detector_count=params['detector_count'],
            angle_range=params['angle_range']
        )

        # Rekonstrukcja
        reconstructed = inverse_radon_all(
            shape=image.shape,
            sinogram=sinogram,
            angle_range=params['angle_range'],
            use_filter=use_filter
        )

        # RMSE
        rmse = calculate_rmse(image, reconstructed)
        rmse_results.append(rmse)

        # Opcjonalnie: zapis obrazów do folderu wynikowego
        save_dir = f"results/{param_name}_{'filter' if use_filter else 'nofilter'}"
        os.makedirs(save_dir, exist_ok=True)
        Image.fromarray(reconstructed.astype(np.uint8)).save(f"{save_dir}/{param_name}_{value}.png")

    return rmse_results

# Generuj wykres RMSE
def plot_rmse(param_values, rmse_results, param_name, use_filter):
    plt.figure()
    plt.plot(param_values, rmse_results, marker='o')
    plt.title(f'RMSE vs {param_name} {"with filter" if use_filter else "without filter"}')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig(f'results/rmse_{param_name}_{"filter" if use_filter else "nofilter"}.png')
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Error: Please provide the image path as an argument.")
        sys.exit(1)
    image_path = sys.argv[1]
    
    # Utwórz folder results jeśli nie istnieje
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
    
    # Załaduj obraz
    image = load_test_image(image_path)
    
    default_params = {
        'detector_count': 180,
        'scan_count': 180,
        'angle_range': 180
    }


    # Zakresy testowe
    detectors_range = range(90, 721, 90)
    scans_range = range(90, 721, 90)
    angles_range = range(45, 271, 45)

    # Testujemy z i bez filtra
    for use_filter in [False, True]:
        # Liczba detektorów
        rmse_detectors = run_experiment(image, 'detector_count', detectors_range, default_params, use_filter)
        plot_rmse(list(detectors_range), rmse_detectors, 'detector_count', use_filter)

        # Liczba skanów
        rmse_scans = run_experiment(image, 'scan_count', scans_range, default_params, use_filter)
        plot_rmse(list(scans_range), rmse_scans, 'scan_count', use_filter)

        # Rozpiętość kąta
        rmse_angles = run_experiment(image, 'angle_range', angles_range, default_params, use_filter)
        plot_rmse(list(angles_range), rmse_angles, 'angle_range', use_filter)

if __name__ == "__main__":
    main()
