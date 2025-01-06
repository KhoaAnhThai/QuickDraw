import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from torchvision import transforms
from model import QuickDraw
import torch
from dictionary import reversed_dict
import torch.nn.functional as F


new_model = QuickDraw(num_class=29)

state_dict = torch.load("QuickDraw.pth", map_location=torch.device("cpu"))
new_model.load_state_dict(state_dict)

canvas_result = st_canvas(
    fill_color="black",  # Màu nền của canvas
    stroke_color="white",  # Màu nét vẽ
    stroke_width=2,  # Độ dày nét vẽ
    background_color="black",  # Màu nền canvas
    width=500,  # Chiều rộng canvas
    height=500,  # Chiều cao canvas
    drawing_mode="freedraw",  # Chế độ vẽ tự do
    key="canvas"
)

if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype(np.uint8))
        
    image = image.resize((28, 28)).convert('L')
        
    transform = transforms.ToTensor()
    img_tensor = transform(image)
        
    new_model.eval()
        
    img_tensor[img_tensor != 0] = 1
    
    output = new_model(img_tensor)  
    top_prob, top_class = torch.max(output, 1)
    
    class_name = reversed_dict[top_class.item()]
    
    if torch.all(img_tensor == 0):
        pass
    else: 
        st.write(f"Class: {class_name}")
    
    canvas_result.image_data = None 
