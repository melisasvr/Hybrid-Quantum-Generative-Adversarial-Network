import pennylane as qml
from pennylane import numpy as np  # Crucial: Use PennyLane's NumPy for Autograd
import matplotlib.pyplot as plt

# 1. Device and Hyperparameters
n_qubits = 4
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits)

# 2. Quantum Generator 
@qml.qnode(dev, diff_method="parameter-shift")
def quantum_generator(noise, weights):
    # Encode noise into the quantum state
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)
        
    # Parameterized Quantum Circuit (PQC)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
        
    # Output the classical synthetic data vector
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 3. Classical Discriminator
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def discriminator(data, disc_weights, disc_bias):
    # A simple logistic regression layer: sigma(W*x + b)
    return sigmoid(np.dot(data, disc_weights) + disc_bias)

# 4. Binary Cross-Entropy Loss Functions
def discriminator_loss(disc_w, disc_b, gen_w, real_data, noise_data):
    pred_real = discriminator(real_data, disc_w, disc_b)
    loss_real = -np.mean(np.log(pred_real + 1e-8))
    
    fake_data = np.stack([quantum_generator(n, gen_w) for n in noise_data])
    pred_fake = discriminator(fake_data, disc_w, disc_b)
    loss_fake = -np.mean(np.log(1.0 - pred_fake + 1e-8))
    
    return loss_real + loss_fake

def generator_loss(gen_w, disc_w, disc_b, noise_data):
    fake_data = np.stack([quantum_generator(n, gen_w) for n in noise_data])
    pred_fake = discriminator(fake_data, disc_w, disc_b)
    return -np.mean(np.log(pred_fake + 1e-8))

# 5. Initialization
np.random.seed(42)

# Wrapping in np.array() is required to assign requires_grad=True safely
gen_weights = np.array(np.random.uniform(0, 2 * np.pi, size=(n_layers, n_qubits)), requires_grad=True)
disc_weights = np.array(np.random.randn(n_qubits, 1), requires_grad=True)
disc_bias = np.array([0.0], requires_grad=True)

# 6. Built-in PennyLane Optimizers
opt_G = qml.AdamOptimizer(stepsize=0.01)
opt_D = qml.AdamOptimizer(stepsize=0.01)

# --- Training Loop ---
epochs = 300
batch_size = 4

# Centered dummy tabular data
real_dataset = np.random.normal(loc=0.0, scale=0.2, size=(100, n_qubits))

# Tracking for Visuals
D_losses = []
G_losses = []

for epoch in range(epochs):
    # Sample batches
    batch_idx = np.random.randint(0, len(real_dataset), size=batch_size)
    real_batch = real_dataset[batch_idx]
    noise_batch = np.random.uniform(-np.pi, np.pi, size=(batch_size, n_qubits))
    
    # Step Discriminator
    def obj_D(w, b):
        return discriminator_loss(w, b, gen_weights, real_batch, noise_batch)
        
    (disc_weights, disc_bias), cost_D = opt_D.step_and_cost(obj_D, disc_weights, disc_bias)
    
    # Step Generator 
    def obj_G(w):
        return generator_loss(w, disc_weights, disc_bias, noise_batch)
        
    gen_weights, cost_G = opt_G.step_and_cost(obj_G, gen_weights)
    
    # Store metrics
    D_losses.append(cost_D)
    G_losses.append(cost_G)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | D_loss: {cost_D:.4f} | G_loss: {cost_G:.4f}")

# --- 7. Added Visualizations ---
# Generate final fake dataset to compare
test_noise = np.random.uniform(-np.pi, np.pi, size=(100, n_qubits))
fake_dataset = np.stack([quantum_generator(n, gen_weights) for n in test_noise])

# Create side-by-side plots
plt.figure(figsize=(14, 5))

# Subplot 1: Convergence
plt.subplot(1, 2, 1)
plt.plot(D_losses, label="Discriminator Loss", color='blue', linewidth=2)
plt.plot(G_losses, label="Generator Loss", color='orange', linewidth=2)
plt.title("QGAN Training Losses over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Data Distribution (First 2 Features)
plt.subplot(1, 2, 2)
plt.scatter(real_dataset[:, 0], real_dataset[:, 1], alpha=0.6, label="Real Data", marker='o', edgecolors='k')
plt.scatter(fake_dataset[:, 0], fake_dataset[:, 1], alpha=0.6, label="Generated Data", marker='x')
plt.title("Data Distribution (Features 0 vs 1)")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# --- 8. Quantitative Evaluation & Data Export ---
def compute_mmd(x, y, gamma=1.0):
    """Computes Maximum Mean Discrepancy (MMD) using a Gaussian RBF kernel."""
    def rbf_kernel(a, b):
        sq_dist = np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * sq_dist)
    
    K_xx = rbf_kernel(x, x)
    K_yy = rbf_kernel(y, y)
    K_xy = rbf_kernel(x, y)
    
    return np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)

# Calculate fidelity score (Lower score = closer to real data)
mmd_score = compute_mmd(real_dataset, fake_dataset)
print(f"\n--- Final Evaluation ---")
print(f"Maximum Mean Discrepancy (MMD): {mmd_score:.6f}")

# Save synthetic tabular dataset to CSV
np.savetxt("quantum_synthetic_data.csv", fake_dataset, delimiter=",", header="Feature_0,Feature_1,Feature_2,Feature_3", comments="")
print("Synthetic tabular dataset saved successfully to 'quantum_synthetic_data.csv'!")