import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
digits_data_path = 'digits_test.csv'
digits_keys_path = 'digits_keys.csv'
digits_data = pd.read_csv(digits_data_path, header=None)
digits_keys = pd.read_csv(digits_keys_path, header=None)

class SelfOrganizingMap:
    def __init__(self, width, height, input_dim):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.weights = 0.5 + 0.01 * np.random.randn(width * height, input_dim)
        self.digit_counts = [np.zeros(10) for _ in range(width * height)]

    def find_bmu(self, input_vector):
        dists = np.linalg.norm(self.weights - input_vector, axis=1)
        return np.argmin(dists)

    def find_second_bmu(self, input_vector, bmu_idx):
        dists = np.linalg.norm(self.weights - input_vector, axis=1)
        dists[bmu_idx] = np.inf  # Invalidate the first BMU to find the second BMU
        return np.argmin(dists)

    def update_weights(self, bmu_idx, input_vector, iteration, total_iterations):
        learning_rate = 0.1 * (1 - iteration / total_iterations)
        bmu_row, bmu_col = divmod(bmu_idx, self.width)
        for i in range(self.height * self.width):
            row, col = divmod(i, self.width)
            dist = np.sqrt((row - bmu_row) ** 2 + (col - bmu_col) ** 2)
            if dist <= 1:  # Neighborhood function
                influence = np.exp(-dist ** 2 / 2)
                self.weights[i] += learning_rate * influence * (input_vector - self.weights[i])

    def train(self, data, epochs):
        total_iterations = epochs * len(data)
        for i in range(total_iterations):
            input_vector = data[i % len(data)]
            bmu_idx = self.find_bmu(input_vector)
            self.update_weights(bmu_idx, input_vector, i, total_iterations)

    def assign_labels_post_training(self, data, labels):
        for input_vector, label in zip(data, labels):
            bmu_idx = self.find_bmu(input_vector)
            self.digit_counts[bmu_idx][label] += 1

    def calculate_errors(self, data):
        quantization_error = 0
        topological_error = 0
        total_data = len(data)
        for input_vector in data:
            bmu_idx = self.find_bmu(input_vector)
            second_bmu_idx = self.find_second_bmu(input_vector, bmu_idx)
            quantization_error += np.linalg.norm((self.weights[bmu_idx] - input_vector) / 255)

            bmu_row, bmu_col = divmod(bmu_idx, self.width)
            second_bmu_row, second_bmu_col = divmod(second_bmu_idx, self.width)
            if abs(bmu_row - second_bmu_row) > 1 or abs(bmu_col - second_bmu_col) > 1:
                topological_error += 1

        quantization_error /= total_data
        topological_error /= total_data
        return quantization_error, topological_error

    def plot_weights_with_labels(self, quant_error, topo_error):
        dominant_digits = [np.argmax(counts) if np.sum(counts) > 0 else None for counts in self.digit_counts]
        percentages = [(np.max(counts) / np.sum(counts) * 100) if np.sum(counts) > 0 else 0 for counts in self.digit_counts]

        fig, axes = plt.subplots(self.height, self.width, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.weights[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
            if dominant_digits[i] is not None:
                label_text = f'{dominant_digits[i]}\n{percentages[i]:.0f}%'
                ax.text(0.5, -0.1, label_text, color='blue', weight='bold', fontsize=7,
                        transform=ax.transAxes, ha='center', va='top')
        plt.suptitle(f"Quantization Error: {quant_error:.4f}, Topological Error: {topo_error:.4f}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# Initialize and train SOM
som = SelfOrganizingMap(10, 10, 784)
som.train(digits_data.values, 12)

# Assign labels post training to calculate dominant digits
som.assign_labels_post_training(digits_data.values, digits_keys.values.flatten())

quant_error, topo_error = som.calculate_errors(digits_data.values)
# Visualization with dominant digits and percentage shown above each square
som.plot_weights_with_labels(quant_error, topo_error)
