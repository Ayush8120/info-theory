"""
Plot function for logistic regression example from class slides
First version: 10/28/2024
This version: 10/30/2024
https://northeastern-datalab.github.io/cs7840/fa24/calendar.html
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm



def create_matrix_plot(matrix, filename, add_text=False, show_colorbar=True, filetype="pdf"):
    """
    Function to create a plot and save it as a PDF or PNG.
    Option to include a colorbar
    Notice that the order is top down although the numbering (the axis direction) is bottom-up!
    """
    n, k = matrix.shape
    fig, ax = plt.subplots(figsize=(k, n))  # Create a figure with a tight width

    # Normalize the colors to range from 0 to 1
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.Blues  # Use a blue colormap

    # Create the matrix visualization
    for i in range(matrix.shape[0]):  # For each data point (rows)
        for j in range(matrix.shape[1]):  # For each class (columns)
            color = cmap(norm(matrix[i, j]))  # Determine the color based on the value
            ax.add_patch(plt.Rectangle((j, n-i-1), 1, 1, facecolor=color, edgecolor='black', linewidth=2))  #direction top-down

            # Optionally, add text inside each square (probability mass rounded to two decimals)
            if add_text:
                text = f'{matrix[i, j]:.2f}'
                ax.text(j + 0.5, n-i - 0.5, text, va='center', ha='center', fontsize=10, color='black')     #direction top-down

    # Draw the outer perimeter of the matrix with doubled line width (linewidth=4)
    ax.add_patch(plt.Rectangle((0, 0), matrix.shape[1], matrix.shape[0], fill=False, edgecolor='black', linewidth=4))

    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(0, matrix.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Adjust the plot area to match the exact width of the matrix
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)

    # Optionally show color bar below the matrix (legend)
    if show_colorbar:
        cbar_ax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.03, ax.get_position().width, 0.02])
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([0, 0.5, 1])  # Set color bar ticks to 0, 0.5, 1
        cbar.ax.tick_params(labelsize=8)  # Make the tick labels smaller

    # Save the plot as a PDF with tight bounding box
    plt.savefig("figures/" + filename + "." + filetype,
                format=filetype,
                dpi=None,
                edgecolor='w',
                orientation='portrait',
                transparent=False,
                bbox_inches='tight',
                pad_inches=0.05)

    # Show then close the plot
    # plt.show()
    # plt.close()

