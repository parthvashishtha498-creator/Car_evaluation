import os
import joblib
import pandas as pd
import gradio as gr
# Car Evaluation Prediction System
# Developed By : Parth
# Roll No      : 241504
# College      : Panipat Institute of Engineering & Technology (PIET), Samalkha

# Load trained model
model = joblib.load("car_safety_model.pkl")


# Prediction Function
def predict_car_safety(
    buying_price,
    maintenance_cost,
    number_of_doors,
    number_of_persons,
    lug_boot,
    safety
):

    data = pd.DataFrame({
        "buying price": [buying_price],
        "maintenance cost": [maintenance_cost],
        "number of doors": [number_of_doors],
        "number of persons": [number_of_persons],
        "lug_boot": [lug_boot],
        "safety": [safety]
    })

    prediction = model.predict(data)[0]

    decision_map = {
        0: " Unacceptable",
        1: " Acceptable",
        2: " Good",
        3: " Very Good"
    }

    return decision_map.get(prediction, str(prediction))


# ================================
# Gradio UI
# ================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Car Evaluation Prediction System"
) as demo:

    gr.Markdown(
        """
#  Car Evaluation Prediction System

### Machine Learning Based Car Evaluation

This project predicts the **overall evaluation of a car** based on:

-  Buying Price
-  Maintenance Cost
-  Number of Doors
-  Number of Persons
-  Luggage Boot Capacity
-  Safety Level

###  Developed By

**Name : Parth**

**Roll No : 241504**

**Course : BCA (Data Science)**

**College : Panipat Institute of Engineering & Technology (PIET), Samalkha**

"""
    )

    with gr.Row():

        buying_price = gr.Dropdown(
            choices=[1, 2, 3, 4],
            label=" Buying Price"
        )

        maintenance_cost = gr.Dropdown(
            choices=[1, 2, 3, 4],
            label=" Maintenance Cost"
        )

    with gr.Row():

        number_of_doors = gr.Dropdown(
            choices=[2, 3, 4, 5],
            label=" Number of Doors"
        )

        number_of_persons = gr.Dropdown(
            choices=[2, 4, 5],
            label=" Number of Persons"
        )

    with gr.Row():

        lug_boot = gr.Dropdown(
            choices=[1, 2, 3],
            label=" Luggage Boot"
        )

        safety = gr.Dropdown(
            choices=[1, 2, 3],
            label=" Safety"
        )

    predict_btn = gr.Button(
        " Predict Car Evaluation",
        variant="primary"
    )

    output = gr.Textbox(
        label="Prediction Result"
    )

    predict_btn.click(
        fn=predict_car_safety,
        inputs=[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety
        ],
        outputs=output
    )

    gr.Markdown(
        """
---
### Project Information

**Model:** Machine Learning Classification

**Framework:** Gradio

**Language:** Python

**Developed By:** **Parth (241504)**

**PIET, Samalkha**

© 2026 All Rights Reserved.
"""
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
