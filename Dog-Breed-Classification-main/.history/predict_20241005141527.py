import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk
import cv2
import os

# สร้างหน้าต่างหลักของ tkinter
root = tk.Tk()
root.title("YOLOv8 Prediction")
root.geometry("800x600")

# โหลดโมเดล YOLO
model = YOLO('runs/detect/train/weights/best.onnx')

# ฟังก์ชันสำหรับเลือกไฟล์ภาพ
def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    if file_path:
        # แสดงเส้นทางของภาพที่เลือก
        label_path.config(text=file_path)

        # ทำนายผลลัพธ์จาก YOLO และแสดงภาพ
        predict_image(file_path)

# ฟังก์ชันสำหรับโหลดและแสดง confusion matrix
def load_confusion_matrix():
    confusion_matrix_path = "runs/detect/train/confusion_matrix.png"  # พาธของ confusion matrix
    result_path = "runs/detect/train/results.png"
    confusion_image = Image.open(confusion_matrix_path)
    result_image = Image.open(result_path)

    # ปรับขนาดภาพ confusion matrix (กำหนดขนาดที่ต้องการ เช่น 400x300)
    confusion_image = confusion_image.resize((300, 200))
    result_image = result_image.resize((300, 200))

    confusion_image_tk = ImageTk.PhotoImage(confusion_image)
    result_image_tk = ImageTk.PhotoImage(result_image)
    
    # ป้ายสำหรับแสดง confusion matrix
    label_confusion.config(image=confusion_image_tk)
    label_confusion.image = confusion_image_tk
    label_result.config(image=result_image_tk)
    label_result.image = result_image_tk


# ฟังก์ชันสำหรับการทำนายภาพ
# def predict_image(file_path):
#     try:
#         # ทำนายด้วย YOLO และบันทึกภาพผลลัพธ์
#         results = model.predict(file_path, save=True, conf=0.6)
#         # ตรวจสอบผลลัพธ์การทำนาย
       
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         label_path.config(text=f"Error during prediction: {e}")

# ฟังก์ชันสำหรับการทำนายภาพ
def predict_image(file_path):
    try:
        # ทำนายด้วย YOLO และบันทึกภาพผลลัพธ์
        results = model.predict(file_path, save=True, conf=0.6)

        # ตรวจสอบผลลัพธ์การทำนายและพาธของไฟล์ที่บันทึก
        print(results)  # พิมพ์ข้อมูลของผลลัพธ์เพื่อตรวจสอบพาธ
        # results[0].save_dir ควรบอกพาธที่บันทึกไฟล์

        # สมมติว่า YOLO บันทึกผลลัพธ์ที่พาธนี้ (พิมพ์ออกมาเพื่อดูจริง ๆ ว่าอยู่ที่ไหน)
        prediction_folder = results[0].save_dir
        print(f"Prediction saved at: {prediction_folder}")

        # ดึงชื่อไฟล์จากพาธต้นฉบับ
        prediction_image_path = os.path.join(prediction_folder, os.path.basename(file_path))

        # โหลดภาพที่ทำนาย
        predicted_image = Image.open(prediction_image_path)
        
        # ปรับขนาดภาพเพื่อแสดงใน Label (กำหนดขนาดที่ต้องการ เช่น 600x400)
        predicted_image = predicted_image.resize((500, 400))

        # แปลงภาพเป็นรูปแบบที่ tkinter เข้าใจ
        predicted_image_tk = ImageTk.PhotoImage(predicted_image)
        
        # แสดงภาพที่ทำนายใน Label
        label_image.config(image=predicted_image_tk)
        label_image.image = predicted_image_tk

    except Exception as e:
        print(f"Error during prediction: {e}")
        label_path.config(text=f"Error during prediction: {e}")




# เพิ่มป้ายสำหรับแสดง confusion matrix ใน UI
frame_images = tk.Frame(root)
frame_images.pack(pady=10)

# สร้างป้ายสำหรับแสดง confusion matrix และ result image
label_confusion = tk.Label(frame_images)
label_confusion.pack(side='left', padx=10)

label_result = tk.Label(frame_images)
label_result.pack(side='left', padx=10)

# ปุ่มสำหรับเลือกภาพ
btn_open = tk.Button(root, text="Open Image", command=open_image)
btn_open.pack(pady=20)

# ป้ายแสดงเส้นทางของภาพที่เลือก
label_path = tk.Label(root, text="No image selected")
label_path.pack(pady=10)
load_confusion_matrix()
# ป้ายสำหรับแสดงภาพที่ทำนาย
label_image = tk.Label(root)
label_image.pack(pady=10)

# เริ่มโปรแกรม tkinter
root.mainloop()
