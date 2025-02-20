import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

def rgb_to_hsi(image):
    image = image.astype('float32') / 255.0
    R, G, B = image[..., 0], image[..., 1], image[..., 2]
    
    I = (R + G + B) / 3
    
    min_val = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-7) * min_val)
    
    H = np.arccos(((R - G) + (R - B)) / (2 * np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-7))
    H = np.degrees(H)
    H[B > G] = 360 - H[B > G]
    
    return H, S * 100, I * 255

def mark_pixels(image, H, S, I, hue_range, sat_range, int_range):
    marked_image = image.copy()
    mask = (H >= hue_range[0]) & (H <= hue_range[1]) & \
           (S >= sat_range[0]) & (S <= sat_range[1]) & \
           (I >= int_range[0]) & (I <= int_range[1])
    marked_image[mask] = [0, 255, 0]
    return marked_image

def open_image():
    global image_rgb, H, S, I
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, S, I = rgb_to_hsi(image_rgb)
        display_image(image_rgb)

def display_image(image):
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)
    panel.configure(image=image_tk)
    panel.image = image_tk

def process_image():
    hue_range = (hue_min.get(), hue_max.get())
    sat_range = (sat_min.get(), sat_max.get())
    int_range = (int_min.get(), int_max.get())
    marked_image = mark_pixels(image_rgb, H, S, I, hue_range, sat_range, int_range)
    display_image(marked_image)

root = Tk()
root.title("HSI Image Processing")

# Panel for displaying the image
panel = Label(root)
panel.grid(row=0, column=0, columnspan=6)

# Buttons
open_btn = Button(root, text="Open Image", command=open_image)
open_btn.grid(row=1, column=0, padx=10, pady=10)

process_btn = Button(root, text="Process Image", command=process_image)
process_btn.grid(row=1, column=1, padx=10, pady=10)

# Sliders for Hue, Saturation, and Intensity ranges
hue_min = Scale(root, from_=0, to=360, orient=HORIZONTAL, label="Hue Min")
hue_min.grid(row=2, column=0)
hue_max = Scale(root, from_=0, to=360, orient=HORIZONTAL, label="Hue Max")
hue_max.grid(row=2, column=1)
sat_min = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Saturation Min")
sat_min.grid(row=3, column=0)
sat_max = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Saturation Max")
sat_max.grid(row=3, column=1)
int_min = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Intensity Min")
int_min.grid(row=4, column=0)
int_max = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Intensity Max")
int_max.grid(row=4, column=1)

root.mainloop()

