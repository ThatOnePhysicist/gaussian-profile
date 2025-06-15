import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude
from PIL import Image


def create_gaussian(size=1024, sigma=100, noise_level=0.0, black_line=False):
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    if noise_level > 0.0:
        max_amplitude = np.max(gaussian)
        noise = np.random.normal(0.0, noise_level * max_amplitude, gaussian.shape)
        gaussian += noise
        gaussian = np.clip(gaussian, 0.0, 1.0)

    if black_line:
        row_index = int(size * 0.4)
        gaussian[600:610,row_index:row_index+2] = 0.0
        gaussian[400:410,row_index+20:row_index+22] = 0.0
    return gaussian

def plot_all():
    sigma_default = 100
    sigma_wide = 1000

    g_default = create_gaussian(sigma=sigma_default)
    g_wide = create_gaussian(sigma=sigma_wide)
    g_line = create_gaussian(sigma=sigma_default, black_line=True)

    # Copy of g_line for gradient magnitude (so it doesn't get affected)
    # g_line_for_grad = np.copy(g_line)
    grad_mag = gaussian_gradient_magnitude(g_line, sigma=1)

    f_default = compute_fourier_transform(g_default)
    f_wide = compute_fourier_transform(g_wide)
    f_line = compute_fourier_transform(g_line)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    axs[0, 0].imshow(g_default, cmap='gray')
    axs[0, 0].set_title("Gaussian (σ=100)")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(g_wide, cmap='gray')
    axs[0, 1].set_title("Gaussian (σ=1000)")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(g_line, cmap='gray')
    axs[0, 2].set_title("Gaussian + Black Line")
    axs[0, 2].axis('off')

    axs[0, 3].imshow(g_line, cmap='gray')
    axs[0, 3].set_title("Gradient Magnitude (scipy)")
    axs[0, 3].axis('off')

    axs[1, 0].imshow(f_default, cmap='gray')
    axs[1, 0].set_title("Fourier (σ=100)")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(f_wide, cmap='gray')
    axs[1, 1].set_title("Fourier (σ=1000)")
    axs[1, 1].axis('off')

    axs[1, 2].imshow(f_line, cmap='gray')
    axs[1, 2].set_title("Fourier (Black Line)")
    axs[1, 2].axis('off')

    axs[1, 3].imshow(grad_mag, cmap='gray')
    axs[1, 3].set_title("Gradient Mag (again)")
    axs[1, 3].axis('off')

    plt.suptitle("2D Gaussians, Fourier Transforms, and Gradient Magnitude", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def compute_fourier_transform(image):
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))
    return magnitude

def plot_image_analysis_from_files(ideal_path, noisy_path):
    ideal_img = Image.open(ideal_path)
    ideal_img_filter = ideal_img.convert('F')
    noisy_img = Image.open(noisy_path)
    noisy_img_filter = noisy_img.convert('F')

    ideal_array = np.array(ideal_img_filter, dtype=np.float32)
    noisy_array = np.array(noisy_img_filter, dtype=np.float32)


    fft_ideal = compute_fourier_transform(ideal_array)
    fft_noisy = compute_fourier_transform(noisy_array)

    grad_ideal = gaussian_gradient_magnitude(ideal_array, sigma=100)
    grad_noisy = gaussian_gradient_magnitude(noisy_array, sigma=100)


    fig, [[col_base, base_case, filter_case, col_filter], [empty_1, row_base, row_filter, empty_2]] = plt.subplots(2, 4, 
                            figsize=(12, 12),
                            gridspec_kw={"height_ratios":[1,2], 
                                         "width_ratios":[1,2,2,1]})

    values = {
        col_base : ["off", "Fourier Transform (Noisy)", fft_noisy,"YlGnBu"],
        base_case : ["off", "Ideal Image", ideal_img],
        filter_case : ["off", "Noisy Image", noisy_img], 
        col_filter : ["off", "Gradient Magnitude (Ideal)", grad_ideal, "YlGnBu"],
        row_base : ["off", "Gradient Magnitude (Noisy)", grad_noisy, "YlGnBu"], 
        row_filter : ["off", "Fourier Transform (Ideal)", fft_ideal, "YlGnBu"]
    }

    try:
        # print(1/9)
        for k,v in values.items():
            k.axis(v[0])
            k.set_title(v[1])
            k.imshow(v[2])

        empty_1.set_visible(False)
        empty_2.set_visible(False)
    except:
        # base_case.set_title("Ideal image").axis("off").imshow(ideal_img)
        base_case.axis('off')
        base_case.set_title("Ideal Image")
        base_case.imshow(ideal_img)

        filter_case.imshow(noisy_img)
        filter_case.set_title("Noisy Image")
        filter_case.axis('off')

        row_filter.imshow(fft_ideal, cmap='YlGnBu')
        row_filter.set_title("Fourier Transform (Ideal)")
        row_filter.axis('off')

        col_base.imshow(fft_noisy, cmap='YlGnBu')
        col_base.set_title("Fourier Transform (Noisy)")
        col_base.axis('off')

        col_filter.imshow(grad_ideal, cmap='YlGnBu')
        col_filter.set_title("Gradient Magnitude (Ideal)")
        col_filter.axis('off')

        row_base.imshow(grad_noisy, cmap='YlGnBu')
        row_base.set_title("Gradient Magnitude (Noisy)")
        row_base.axis('off')

        empty_1.set_visible(False)
        empty_2.set_visible(False)

    plt.suptitle("Image Analysis: Ideal vs. Noisy", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


if __name__ == "__main__":
    plot_image_analysis_from_files("ideal.png", "profile.jpg")
