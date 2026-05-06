import gradio as gr
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_knn_model.pkl")

# Flower images
flower_images = {
    "setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa.jpg",
    "versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# Prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width):
    try:
        data = np.array([[float(sepal_length), float(sepal_width),
                          float(petal_length), float(petal_width)]])
        
        prediction = model.predict(data)[0]

        return f"🌸 Predicted: {prediction.capitalize()}", flower_images[prediction]

    except:
        return "⚠️ Please enter valid numeric values!", None


# 🎨 CSS for animation
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
    font-size: 16px !important;
}

input {
    border-radius: 8px !important;
}
"""

# UI Layout
with gr.Blocks(css=css) as demo:

    gr.Markdown("## 🌸 Iris Flower Classifier (Interactive ML App)")

    with gr.Row():
        with gr.Column():
            sepal_length = gr.Number(label="Sepal Length (cm)")
            sepal_width  = gr.Number(label="Sepal Width (cm)")
            petal_length = gr.Number(label="Petal Length (cm)")
            petal_width  = gr.Number(label="Petal Width (cm)")

            predict_btn = gr.Button("Predict 🌿")

        with gr.Column():
            output_text = gr.Textbox(label="Prediction")
            output_image = gr.Image(label="Flower Image")

    predict_btn.click(
        fn=predict,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=[output_text, output_image]
    )

# Launch app
demo.launch()