import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('picture1.jpg', cv2.IMREAD_GRAYSCALE)  

# Robert 
robert_x = cv2.filter2D(img, -1, np.array([[1, 0], [0, -1]]))  
robert_y = cv2.filter2D(img, -1, np.array([[0, 1], [-1, 0]]))  
robert_xy = cv2.bitwise_or(robert_x, robert_y) 

# Prewitt 
prewitt_x = cv2.filter2D(img, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))  
prewitt_y = cv2.filter2D(img, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))  
prewitt_xy = cv2.bitwise_or(prewitt_x, prewitt_y)  

# Sobel 
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # ใช้ Sobel เพื่อหาขอบภาพในแนวนอน
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # ใช้ Sobel เพื่อหาขอบภาพในแนวตั้ง
sobel_xy = cv2.bitwise_or(np.uint8(np.absolute(sobel_x)), np.uint8(np.absolute(sobel_y)))  # รวมขอบภาพแนวนอนและแนวตั้งด้วย bitwise_or และแปลงผลลัพธ์เป็น uint8

# Canny 
canny = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 50, 100)  # ใช้ GaussianBlur เพื่อลด noise และใช้ Canny เพื่อหาขอบภาพ

# แสดงผลภาพ
titles = ["ORIGINAL", "ROBERT", "PREWITT", "SOBEL", "CANNY"]  # กำหนดชื่อหัวข้อของแต่ละภาพ
images = [img, robert_xy, prewitt_xy, sobel_xy, canny]  # เก็บภาพต้นฉบับและภาพที่ผ่านการหาขอบ

plt.figure(figsize=(15, 8))  # กำหนดขนาดของภาพทั้งหมดที่จะแสดง
for i in range(5):
    plt.subplot(2, 3, i + 1)  # สร้าง subplot สำหรับแต่ละภาพ
    plt.imshow(images[i], cmap='gray')  # แสดงภาพในโทนสีเทา
    plt.title(titles[i])  # กำหนดหัวข้อของแต่ละภาพ
    plt.axis('off')  # ซ่อนแกนของกราฟ

plt.tight_layout()  # ปรับ layout ให้เหมาะสม
plt.show()  # แสดงผลภาพทั้งหมด
