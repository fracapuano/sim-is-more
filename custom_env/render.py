import pygame
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def figure_to_image(fig:plt.figure, screen_width:int, screen_height:int)->pygame.Surface:
    """
    Converts a matplotlib figure to a pygame Surface object.

    Parameters:
    fig (plt.figure): The matplotlib figure to convert.
    screen_width (int): The width of the screen.
    screen_height (int): The height of the screen.

    Returns:
    pygame.Surface: The converted pygame Surface object.
    """    
    fig.canvas.draw()
    plt.close(fig)
    # Get the RGBA buffer from the figure
    X = np.array(fig.canvas.renderer.buffer_rgba())
    # Convert to RGB, and transpose to fit pygame's format
    figure_rgb_array = np.transpose(
        np.array(Image.fromarray(X).convert('RGB')),
        axes=(1, 0, 2)
    )
    
    # Make a surface from the frame array
    surface = pygame.surfarray.make_surface(figure_rgb_array)
    # rescaling the surfaces to fit screen size
    scaled_surface = pygame.transform.scale(surface, (screen_width, screen_height))

    return scaled_surface

