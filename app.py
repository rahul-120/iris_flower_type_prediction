import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("iris_knn_model.pkl")

# Flower images (online links)
flower_images = {
    "setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa.jpg",
    "versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# Prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)[0]

    return f"🌸 {prediction.capitalize()}", flower_images[prediction]


# Custom CSS (animation)
css = """
body {
    background: linear-gradient(-45deg, #1a2a6c, #b21f1f, #fdbb2d);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gr-button {
    background-color: #ff6b6b !important;
    color: white !important;
    border-radius: 10px !important;
}
"""

# Interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🌸 Iris Flower Classifier (Interactive)")

    with gr.Row():
        with gr.Column():
            sepal_length = gr.Slider(4, 8, step=0.1, label="Sepal Length")
            sepal_width = gr.Slider(2, 5, step=0.1, label="Sepal Width")
            petal_length = gr.Slider(1, 7, step=0.1, label="Petal Length")
            petal_width = gr.Slider(0.1, 2.5, step=0.1, label="Petal Width")

            btn = gr.Button("Predict 🌿")

        with gr.Column():
            output_text = gr.Textbox(label="Prediction")
            output_image = gr.Image(label="Flower")
            

    btn.click(
        fn=predict,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=[output_text, output_image]
    )

demo.launch()