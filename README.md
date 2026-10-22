# Neural Networks from Scratch

**From raw pixels to handwritten digits—with a neural network built in NumPy.**

This project explores how a multilayer perceptron learns to classify images, one forward pass and gradient update at a time. The network's math is implemented by hand: weights, biases, sigmoid activations, backpropagation, and gradient descent.

**Recorded MNIST test accuracy: 95.63% after 30 epochs.**

## What's inside?

- **A NumPy-powered network** with configurable layer sizes.
- **Backpropagation from scratch** so you can follow how each gradient is calculated.
- **A complete MNIST experiment** covering training, evaluation, and digit prediction.
- **Learning curves** for training and test loss and accuracy in the V0 notebook.

Keras loads the dataset; NumPy handles the neural network itself. Matplotlib visualizes the results.

## The experiment at a glance

The [MNIST dataset](https://keras.io/api/datasets/mnist/) contains grayscale images of handwritten digits, with 60,000 training images and 10,000 test images. Each 28 × 28 image becomes 784 input values, normalized by dividing by 255.

```text
  Handwritten digit       Hidden layers       Predicted digit
     28 × 28 pixels
          │
          ▼
     [784 inputs] ────► [32] ────► [16] ────► [10 outputs]
                                                0–9
```

The saved run in [`NeuralNetworkV0.ipynb`](NeuralNetworkV0.ipynb) uses:

| Setting | Value |
| --- | --- |
| Architecture | 784 → 32 → 16 → 10 |
| Activation | Sigmoid |
| Loss metric | Mean squared error |
| Learning rate | 0.005 |
| Batch size | 1 |
| Training epochs | 30 |
| Final test accuracy | **95.63%** |

These are recorded notebook results; new runs may differ because initialization and training order are random.

## Get started

From the project directory, install the notebook dependencies:

```bash
python -m pip install numpy matplotlib tensorflow jupyterlab
```

TensorFlow provides a backend for the notebook's Keras import.

Launch JupyterLab:

```bash
jupyter lab
```

Open **[`NeuralNetworkV0.ipynb`](NeuralNetworkV0.ipynb)** and run the cells in order to:

1. Download and load MNIST through Keras (internet access is needed on the first run).
2. Create the network and train it for 30 epochs.
3. Inspect loss and accuracy curves.
4. Display a test image and compare its predicted digit with the true label.

For a quicker first experiment, reduce `epochs` in the training cell.

### Predict a digit after training

With the trained `n` and test data available in the notebook:

```python
index = np.random.randint(len(x_test))
scores = n.forward(x_test[index])
prediction = int(np.argmax(scores))

plt.imshow(x_test[index], cmap="gray")
plt.title(f"Predicted: {prediction} | Actual: {y_test[index]}")
plt.axis("off")
plt.show()
```

## Explore the code

| File | What to explore |
| --- | --- |
| [`NeuralNetworkV0.ipynb`](NeuralNetworkV0.ipynb) | Full training experiment, saved results, learning curves, and predictions. Start here. |
| [`NeuralNetworkV0.py`](NeuralNetworkV0.py) | Standalone `NeuralNetworkV0` class with forward propagation, gradient calculations, updates, and loss computation. |
| [`NeuralNetwork.ipynb`](NeuralNetwork.ipynb) | An earlier notebook implementation and training experiment. |

## Make it your own

Try changing the hidden-layer widths, learning rate, or batch size in the V0 notebook. Watch the learning curves: how quickly does the loss fall, and does better training accuracy also improve test accuracy?

Small changes make this a hands-on way to connect the equations of a neural network with the behavior of a working classifier.
