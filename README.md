# Hybrid Quantum Generative Adversarial Network (QGAN)

A Python implementation of a hybrid quantum-classical Generative Adversarial Network (QGAN) designed for synthetic tabular data generation. Built using **PennyLane** and **Autograd (NumPy)**, this framework embeds continuous latent noise into a Parameterized Quantum Circuit (PQC) generator while training against a classical discriminator to generate realistic low-dimensional distributions without deep learning framework overhead (such as PyTorch or TensorFlow).

---
```
## Architecture Overview
+------------------------+
                      |   Latent Noise Vector  |
                      +-----------+------------+
                                  |
                                  v
+------------------+      +-----------+------------+      +--------------------+
|  Real Tabular    |      |   Quantum Generator    |      |  Synthetic Tabular |
|     Data         +----->|  (Parameterized Circuit)----->|        Data        |
+--------+---------+      +------------------------+      +---------+----------+
|                                                          |
+------------------------+---------------------------------+
|
v
+-----------+------------+
| Classical Discriminator|
|  (Logistic Classifier) |
+-----------+------------+
|
v
+-----------+------------+
| Real / Synthetic Score |
+------------------------+
```

* **Generator (Quantum):** Uses a $4$-qubit Parameterized Quantum Circuit (PQC) with $4$ rotational layers (`qml.RY`) and cyclic entangling gates (`CNOT`). Data expectation values are evaluated using Pauli-Z measurements ($\langle Z \rangle$), outputting values bounded within $[-1, 1]$.
* **Discriminator (Classical):** A lightweight logistic regression layer built from scratch using PennyLane's Autograd engine.
* **Optimization:** Parameter updates for both models are computed using parameter-shift rules and Adam optimizers (`qml.AdamOptimizer`).

---

## Features

* **Pure PennyLane & NumPy:** Runs natively using standard NumPy arithmetic and PennyLane's Autograd wrapper.
* **Mode Collapse Mitigation:** Avoids spatial collapse through full state-entanglement layers across the qubit array.
* **Built-in Visualizations:** Automatically generates real-time loss tracking curves alongside 2D target vs. generated distribution scatter plots via `matplotlib`.
* **Quantitative Evaluation:** Computes the **Maximum Mean Discrepancy (MMD)** with a Gaussian RBF kernel to evaluate synthetic distribution fidelity.
* **Automated Data Export:** Exports generated synthetic features directly into a formatted `.csv` file upon completion.

---

## Setup & Installation

- Ensure you have Python 3.8+ installed. Install the required dependencies using `pip`:
- `pip install pennylane matplotlib`

## Usage
1. Clone or save the project repository locally:
- `python qgan_pennylane.py`

2. Execution Output:
- Terminal Logs: Displays epoch step losses for both the Generator and Discriminator.
- Visual Figure: Displays a two-panel figure showing loss convergence and 2D feature mapping.
- Dataset Output: Generates quantum_synthetic_data.csv in the root working directory.

## Performance & Results
- The setup achieves equilibrium across 300 training epochs, converging to a balanced minimax loss state.
- Qubit Count: 4 Qubits- Dimension of the synthetic feature space
- Circuit Depth: 4 Layers: Parameterized ansatz layers
- Loss Function: Binary Cross-Entropy: Minimax loss formulation
- MMD Score: ~0.0401: High-fidelity match to target Gaussian distribution

## File Structure
```
├── qgan_pennylane.py           # Main training, visualization, and metric evaluation script
├── quantum_synthetic_data.csv  # Generated output tabular dataset (post-execution)
└── README.md                   # Project documentation
```

### Tech Stack
- PennyLane - Quantum Machine Learning Library
- Matplotlib - Plotting & Visualization
- NumPy - Linear Algebra & Autograd Engine
