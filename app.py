import gradio as gr
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Train model directly (no .pkl needed)
iris = load_iris()
X = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

species = ["setosa", "versicolor", "virginica"]

# Prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width):
    try:
        data = np.array([[float(sepal_length), float(sepal_width),
                          float(petal_length), float(petal_width)]])
        
        prediction = model.predict(data)[0]
        
        return f"🌸 Predicted Flower: {species[prediction].capitalize()}"

    except:
        return "⚠️ Please enter valid numeric values!"


# 🎨 CSS Animation
css = """
body {
    background: linear-gradient(-90deg, #1a2a6c, #b21f1f, #fdbb2d);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% { background-position: 20% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gr-button {
    background-color: #ff6b6b !important;
    color: white !important;
    border-radius: 10px !important;
}
"""

# UI
with gr.Blocks(css=css) as demo:

    gr.Markdown("## 🌸 Iris Flower Classifier")

    with gr.Row():
        with gr.Column():
            sepal_length = gr.Number(label="Sepal Length (cm)")
            sepal_width  = gr.Number(label="Sepal Width (cm)")
            petal_length = gr.Number(label="Petal Length (cm)")
            petal_width  = gr.Number(label="Petal Width (cm)")

            btn = gr.Button("Predict 🌿")

        with gr.Column():
            output = gr.Textbox(label="Result")

    btn.click(
        fn=predict,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=output
    )



demo.launch()