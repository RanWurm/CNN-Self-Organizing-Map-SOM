# Self-Organizing Map (SOM) for Digit Recognition

## Overview

This project implements a Self-Organizing Map (SOM), a type of artificial neural network used for reducing dimensions and visualizing high-dimensional data. The project uses SOM to classify and visualize the MNIST dataset of handwritten digits. It demonstrates how SOM can be used to identify similar digits and organize them into a two-dimensional grid based on their features.

## Project Description

### Objective

To develop an SOM that can learn from high-dimensional data and provide visual insights into the clustering and topology of the data. This specific implementation focuses on the MNIST dataset, showcasing how different handwritten digits are recognized and grouped by the SOM.

### Features

- **High-dimensional Data Handling**: Efficiently manages and processes the MNIST dataset, which contains images of handwritten digits, each represented as a 784-dimensional vector.
- **Dimensionality Reduction**: Implements dimensionality reduction to project the high-dimensional data onto a two-dimensional grid.
- **Visual Representation**: Provides a graphical representation of the SOM grid where each neuron's weights are displayed as images, showing which type of digit each neuron represents.
- **Error Calculation**: Computes quantization and topological errors to evaluate the SOM's performance.

### Implementation Details

#### Classes and Methods

- `SelfOrganizingMap`: The main class that implements the SOM, including methods for initializing the map, finding the Best Matching Unit (BMU), updating weights, training, and plotting the map.
- `find_bmu`: Identifies the neuron (node) in the SOM whose weights are most similar to the input vector.
- `update_weights`: Adjusts the weights of the SOM neurons based on their distance from the BMU to better resemble the input data over time.
- `train`: Runs through the dataset multiple times, adjusting the SOM to better reflect the underlying patterns in the data.
- `assign_labels_post_training`: Assigns labels to each neuron based on the most common digit it represents after training.
- `calculate_errors`: Calculates the quantization and topological errors of the SOM after training.
- `plot_weights_with_labels`: Visualizes the SOM grid, showing the dominant digit and the confidence (as a percentage) for each neuron.

### Usage

1. **Data Loading**: Load the dataset containing the digit images and their corresponding labels.
2. **Initialization and Training**:
    ```python
    som = SelfOrganizingMap(10, 10, 784)  # Create a 10x10 SOM for 784-dimensional input.
    som.train(digits_data.values, 12)     # Train the SOM with 12 epochs.
    ```
3. **Label Assignment and Error Calculation**:
    ```python
    som.assign_labels_post_training(digits_data.values, digits_keys.values.flatten())
    quant_error, topo_error = som.calculate_errors(digits_data.values)
    ```
4. **Visualization**:
    ```python
    som.plot_weights_with_labels(quant_error, topo_error)  # Plot the trained SOM with labels.
    ```

### Dependencies

- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation.
- **Matplotlib**: For plotting the SOM grid.

### Running the Project

Ensure you have Python and the necessary libraries installed. Run the script using:

```bash
python som_digit_recognition.py
Testing
The project includes detailed steps to test the SOM using the MNIST dataset, focusing on visual inspection of the grid and error metrics to assess performance.

Contributions
Contributions to extend the functionality, such as adding more interactive visualizations or improving the SOM training algorithm, are welcome. Please fork the repository and submit a pull request with your changes.

License
This project is released under the MIT License.

Contact
For more information or to raise issues, please contact me at [your-email@example.com].
