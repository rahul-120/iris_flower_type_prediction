import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("iris_knn_model.pkl")

def predict(sepal_length, sepal_width, petal_length, petal_width):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)[0]
    
    return prediction

# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs="text",
    title="🌸 Iris Flower Classifier"
)

interface.launch()