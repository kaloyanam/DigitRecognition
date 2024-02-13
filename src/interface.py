import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from load_tests import *
from model import *

# Create a model and train it (or load data when it's already trained)
model, x_train, y_train = create_model()
model = train(model, x_train, y_train, True)
    

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0.3)",
    stroke_width=7,
    stroke_color='#ffff00',
    background_color='000000',
    update_streamlit=True,
    height= 140,
    width = 140, 
    drawing_mode='freedraw',
    key="canvas",
)

# Display prediction
if canvas_result.image_data is not None:
    st.image(load_image(canvas_result.image_data).numpy().reshape(28,28))
    p = predict_image(load_image(canvas_result.image_data), model)
    st.bar_chart(pd.DataFrame(p))
    st.header('My guess is: **' + str(np.argmax(p)) + '**', )