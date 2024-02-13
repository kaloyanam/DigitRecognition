import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
from load_tests import *
from model import *

# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
# )

# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# if drawing_mode == 'point':
#     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ", '#ffffff')
# bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

#realtime_update = st.sidebar.checkbox("Update in realtime", True)
model,x_train,y_train = create_model()
model = train(model,x_train,y_train,True)
    

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0.3)",  # Fixed fill color with some opacity
    stroke_width=8,
    stroke_color='#ffff00',
    background_color='000000',
    update_streamlit=True,
    height= 140,
    width = 140, 
    drawing_mode='freedraw',
    key="canvas",
)

# # Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(load_image(canvas_result.image_data).numpy().reshape(28,28))
    st.bar_chart(pd.DataFrame(predict_image(load_image(canvas_result.image_data),model,x_train,y_train)))
    #load_image(canvas_result.image_data)
    #print(canvas_result.image_data)
        
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)