import pygame
import numpy as np
from typing import (
    List,
    Text,
    Optional
)
from PIL import Image
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def is_pareto_efficient(costs:NDArray, return_mask:bool=True)->NDArray:
    """
    Find the pareto-efficient points.

    Args:
        costs (ndarray): An (n_points, n_costs) array.
        return_mask (bool, optional): True to return a mask. 
            Defaults to True.

    Returns:
        NDArray: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array.
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

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

def create_background_scatter(ax:plt.Axes, x_coordinates:NDArray, y_coordinates:NDArray, c:Optional[NDArray]=None)->plt.Axes:
    """
    Create a scatter plot of background points on the given axes.

    Args:
        ax (plt.Axes): The axes on which to create the scatter plot.
        x_coordinates (NDArray): The x-coordinates of the points.
        y_coordinates (NDArray): The y-coordinates of the points.
        c (Optional[NDArray], optional): The color of the points. Defaults to None.

    Returns:
        plt.Axes: The modified axes with the scatter plot.
    """
    ax.scatter(
        x_coordinates,
        y_coordinates,
        s=5,
        c=c, # light gray
        zorder=0
    )

    return ax

def create_background_vlines(ax:plt.Axes, x_coordinates:NDArray, label:Optional[Text]=None)->plt.Axes:
    """
    Create vertical lines on the background of the plot.

    Args:
        ax (plt.Axes): The matplotlib Axes object on which to create the lines.
        x_coordinates (NDArray): The x-coordinates of the vertical lines.
        y_coordinates (NDArray): The y-coordinates of the vertical lines.

    Returns:
        plt.Axes: The modified matplotlib Axes object.
    """
    ymin, ymax = ax.get_ylim()
    # displaying the latency cutoff
    ax.vlines(
        x=x_coordinates,
        ymin=ymin,
        ymax=ymax,
        colors="red",
        linestyles="dashed",
        label=label,
        lw=2
    )

    return ax

def draw_architectures_on_background(ax:plt.Axes, x_coordinates:NDArray, y_coordinates:NDArray, **kwargs)->plt.Axes:
    """
    Draws architectures onto the background of the given Axes object.

    Args:
        ax (plt.Axes): The Axes object to draw on.
        x_coordinates (NDArray): The x-coordinates of the architectures.
        y_coordinates (NDArray): The y-coordinates of the architectures.

    Returns:
        plt.Axes: The modified Axes object.
    """    
    line, = ax.plot([], [], zorder=1, c="red")
    scatt = ax.scatter([],[], s=75, c="red", marker="X", **kwargs)

    line.set_data(x_coordinates, y_coordinates)
    scatt.set_offsets(np.c_[x_coordinates, y_coordinates])
    ax.legend(loc="upper right", framealpha=1., fontsize=12)

    return ax

def draw_hbars(ax:plt.Axes, labels:List[Text], widths:NDArray, **kwargs)->plt.Axes:
    """
    Draw a horizontal bar chart on the given axes.

    Args:
        ax (plt.Axes): The axes on which to draw the bar chart.
        labels (List[Text]): The labels for the bars.
        widths (NDArray): The widths of the horizontal bars.

    Returns:
        plt.Axes: The modified axes with the bar chart.
    """
    bars = ax.barh(
        labels,
        widths,
        **kwargs
    )
    
    ax.bar_label(bars, fmt="%.2f")  # Annotate the bars with their height
    
    return ax

