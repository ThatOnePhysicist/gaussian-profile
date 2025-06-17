import numpy as np
import matplotlib.pyplot as plt
from numpy._typing._array_like import NDArray
from scipy.ndimage import gaussian_gradient_magnitude
from PIL import Image
import traceback
import pandas as pd
import matplotlib as mpl

# Docstrings
from typing import Any
from numpy import floating


# Formats plots to LaTeX style
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Use LaTeX-style serif font
    "font.serif": ["Computer Modern Roman"],  # Default LaTeX serif font
    "text.latex.preamble": r"\usepackage{amsmath}"  # Optional, for math support
})

def create_gaussian(size=1024, sigma=100, noise_level=0.0, black_line=False) -> NDArray[floating[Any]] | NDArray[Any]:
    """
    Creates 2D gaussian images.

    Parameters
    ----------
    size : int, default 1024
        Size for image.
    sigma : int, default 100
        Measure of spread for the gaussian.
    noise_level : float, default 0.0
        Amount of randomly distributed noise to add to image. TODO: Not very useful when applying gaussian gradient filter.
    black_line : bool, default False
        Artificial black line inserted by zeroing values. 
    
    Returns
    -------
    Returns an n-dimensional array. 
   
    Notes
    -----
    https://en.wikipedia.org/wiki/Gaussian_function
    """
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

def compute_fourier_transform(image) -> NDArray:
    """
    Computes Fourier transform of input image.

    Parameters
    ----------
    image : array like
        Input image. 
    
    Returns
    -------
    magnitude : NDArray
        Returns an n-dimensional array. 
    """
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))
    return magnitude

def one_dimensional_plot(image:NDArray) -> tuple[pd.Series, pd.Series]:
    """
    Computes the 1-dimensional summations of pixel values of an image.

    Parameters
    ----------
    image : NDArray
        Image in array form.
    
    Returns
    -------
    df_col, df_row : pd.Series[Any], pd.Series[Any]
        Tuple of series representing column and row sums of input image.
    """
    df = pd.DataFrame(image)
    df_col = df.sum(axis=1)
    df_row = df.sum(axis=0)
    return df_col, df_row

def plot_image_analysis_from_files(ideal_path:str) -> None:
    """
    Generates 2x4 subplots of image, filtered image, and respective row and column 1-D plots.

    Parameters
    ----------
    ideal_path : str
        Path to image being plotted.
    
    Returns
    -------
    None

    Notes
    -----
    The 2x4 image is arranged in the following configuration,
    1st row: row sum of unfiltered image, unfiltered image, filtered image, row sum of filtered image
    2nd row: empty, col sum of unfiltered image, col sum of filtered image, empty.

    For the gridspec_kw={"height_ratios":[a,b], "width_ratios":[c,d,e,f]}
    argument in subplots, I haven't seen sufficient documentation on this
    so I'll decipher the process.
    
    `a` and `b` are heights for the 1st and 2nd row respectively,
    `c`, `d`, `e`, and `f` represent the widths for columns 1, 2, 3, and 4 respectively
    accounting for the full 2x4 grid. Finally, the dimensions are relative to the largest value, 
    i.e. for height_rations = [2, 1] means the 1st row is twice the height of the 2nd row. 
    Likewise, for width_rations = [1, 2, 2, 1], the first and last columns are half the 
    width of the 2nd and 3rd column. 
    """
    ideal_img = Image.open(ideal_path)
    ideal_img_filter = ideal_img.convert('F') # F converts to black and white

    ideal_array = np.array(ideal_img_filter, dtype=np.float32)
    row_ideal, col_ideal = one_dimensional_plot(ideal_array)

    grad_ideal = gaussian_gradient_magnitude(ideal_array, sigma=1)
    grad_row, grad_col = one_dimensional_plot(grad_ideal)

    fig, [
        [row_base, base_case, filter_case, row_filter], 
        [empty_1, col_base, col_filter, empty_2]
        ] = plt.subplots(2, 4, figsize=(16, 9),
                            gridspec_kw={
                                "height_ratios":[2,1], 
                                "width_ratios":[1,2,2,1]
                                })
    
    values = {
        col_base: ["on", "col_b", col_ideal],
        base_case: ["on", "img_b", ideal_array],
        filter_case: ["on", "img_f", grad_ideal], 
        col_filter: ["on", "col_f", grad_col],
        row_base: ["on", "row_b", row_ideal], 
        row_filter: ["on", "row_f", grad_row]
    }
    try:
        for k,v in values.items():
            k.axis(v[0])
            k.set_title(v[1])
            match v[1][0:3]:
                case "row":
                    k.plot(v[-1], np.arange(len(v[-1])))
                    k.set_ylim(0, len(v[-1]))
                    k.invert_yaxis()
                case "col":
                    k.plot(np.arange(len(v[-1])), v[-1])
                    k.set_xlim(0, len(v[-1]))
                case "img":
                    img_array = v[-1]
                    height, width = img_array.shape
                    k.imshow(img_array, cmap='gray', origin='upper', extent=[0, width, height, 0])
                    k.set_xlim(0, width)
                    k.set_ylim(height, 0)
                    k.set_aspect("auto")


        empty_1.set_visible(False)
        empty_2.set_visible(False)
        plt.suptitle("Image Analysis: Ideal vs. Noisy", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()
    except Exception as e:
        traceback.print_exc()


def main():
    plot_image_analysis_from_files("ideal.png")
    
if __name__ == "__main__":
    main()