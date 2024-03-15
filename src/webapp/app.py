import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate c(x, y)
def c(x, y, c1):
    #return c1 * (y - np.log(1-y)) + (1-c1)*(x-np.log10(x))
    return y * (x**c1 - 1)

def colormap_background():
    # Streamlit UI
    st.title("Color Map Visualization")
    c1 = st.slider("Select the value of c1", -1., 1., value=0.)

    # Create a grid of x and y values
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)

    # Calculate c(x, y)
    C = c(X, Y, c1)

    # Plotting
    fig, ax = plt.subplots()
    ax.imshow(C, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax.set_xlabel('Latency')
    ax.set_ylabel('Trainin-free score')
    ax.set_title(f'Reward value for various combinations')
    st.pyplot(fig)

def main():
    colormap_background()

if __name__ == "__main__":
    main()

