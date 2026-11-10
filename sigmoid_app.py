# Sigmoid Activation Function Visualiser (Streamlit app)
# Run with:  streamlit run sigmoid_app.py
#
# What this app does:
#   1. Shows the formula of the Sigmoid activation function
#   2. Plots the function and its derivative (gradient)
#   3. Lets you test one neuron:  z = w.x + b  ->  output = f(z)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------
# 1. MATHS: the activation function and its derivative
# ---------------------------------------------------------------
def activation(z):
    """Sigmoid: squashes any input into a value between 0 and 1."""
    return 1 / (1 + np.exp(-z))


def derivative(z):
    """Slope of sigmoid: f(z) * (1 - f(z))."""
    s = activation(z)
    return s * (1 - s)


# ---------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------
def neuron_output(inputs, weights, bias):
    """One neuron: weighted sum plus bias (z), then the activation function."""
    z = float(np.dot(inputs, weights) + bias)
    return z, float(activation(z))


def plot_function(z_min, z_max, z_value, show_derivative=True):
    """Draw the activation function (and optionally its derivative)."""
    z = np.linspace(z_min, z_max, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, activation(z), linewidth=2, label="Sigmoid  f(z)")
    if show_derivative:
        ax.plot(z, derivative(z), "--", label="Derivative  f'(z)")
    ax.scatter([z_value], [activation(z_value)], color="red", zorder=5,
               label="Selected input")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")  # 0.5 decision line
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("z  (weighted sum + bias)")
    ax.set_ylabel("Output")
    ax.set_title("Sigmoid Activation Function")
    ax.grid(True)
    ax.legend()
    return fig


# ---------------------------------------------------------------
# 3. PAGE LAYOUT
# ---------------------------------------------------------------
st.set_page_config(page_title="Sigmoid Activation Function", layout="centered")
st.title("Sigmoid Activation Function")
st.write("Sigmoid maps any input value to an output between 0 and 1, so the output can be read as a probability.")
st.latex(r"f(z) = \frac{1}{1 + e^{-z}}")

# Sidebar controls
st.sidebar.header("Controls")
z_min, z_max = st.sidebar.slider("Range of z (x-axis)", -10.0, 10.0,
                                 (-5.0, 5.0), step=0.5)
z_value = st.sidebar.slider("Pick an input value z", -10.0, 10.0,
                            0.56, step=0.01)
show_derivative = st.sidebar.checkbox("Show derivative (gradient)", value=True)

if z_min >= z_max:
    st.error("The minimum of the range must be smaller than the maximum.")
    st.stop()

tab_graph, tab_neuron, tab_table, tab_notes = st.tabs(
    ["Graph", "Neuron calculator", "Table of values", "Key points"])

# ---- Tab 1: graph ----
with tab_graph:
    fig = plot_function(z_min, z_max, z_value, show_derivative)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    col1.metric("Output f(z)", f"{float(activation(z_value)):.4f}")
    col2.metric("Derivative f'(z)", f"{float(derivative(z_value)):.4f}")

    if activation(z_value) > 0.5:
        st.success("Output is greater than 0.5, so a classifier predicts class 1 (e.g. spam).")
    else:
        st.info("Output is 0.5 or lower, so a classifier predicts class 0 (e.g. not spam).")

# ---- Tab 2: one neuron, step by step ----
with tab_neuron:
    st.write("Change the inputs (x), weights (w) and bias (b). "
             "The neuron computes z = w1*x1 + w2*x2 + ... + b and then applies "
             "the Sigmoid function.")
    default_inputs = [0.8, 0.0]
    default_weights = [0.7, 0.2]
    inputs, weights = [], []
    cols = st.columns(len(default_inputs))
    for i, col in enumerate(cols):
        inputs.append(col.number_input(f"x{i + 1}", value=default_inputs[i],
                                       step=0.1, key=f"x{i}"))
        weights.append(col.number_input(f"w{i + 1}", value=default_weights[i],
                                        step=0.1, key=f"w{i}"))
    bias = st.number_input("Bias b", value=0.0, step=0.1)

    z_neuron, out_neuron = neuron_output(inputs, weights, bias)
    c1, c2 = st.columns(2)
    c1.metric("Weighted sum z", f"{z_neuron:.4f}")
    c2.metric("Neuron output f(z)", f"{out_neuron:.4f}")
    st.caption("Lecture example (output neuron): hidden outputs [0.8, 0] and weights [0.7, 0.2] give z = 0.56, so sigmoid(0.56) is about 0.636. This is greater than 0.5, so the email is classified as spam (1).")

# ---- Tab 3: table of values ----
with tab_table:
    samples = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    st.table({
        "z": samples,
        "f(z)": [round(float(activation(s)), 4) for s in samples],
        "f'(z)": [round(float(derivative(s)), 4) for s in samples],
    })

# ---- Tab 4: key points ----
with tab_notes:
    st.markdown(
        "- **Range:** 0 to 1 (never exactly 0 or 1).\n"
        "- **f(0) = 0.5**, the centre of the S-shaped curve.\n"
        "- **Non-linear**, smooth and easy to differentiate.\n"
        "- **Commonly used in the output layer** for binary classification.\n"
        "- **Weakness:** for very large or very small z the curve is flat, the gradient is almost 0 and learning slows down (vanishing gradient)."
    )
