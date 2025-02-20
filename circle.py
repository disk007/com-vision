import matplotlib.pyplot as plt
import numpy as np
import cv2 

image = cv2.imread('./orange.jpg')  # อ่านภาพ
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # แปลงเป็นภาพขาวดำ
blurred_img = cv2.GaussianBlur(img, (9, 9), 0) 
canny_edges = cv2.Canny(blurred_img, 70, 150,apertureSize=3)


def hough_circle_detection(original_image):
    # ทำ Gaussian Blur อีกครั้งเพื่อช่วยตรวจจับวงกลม
    blurred = cv2.GaussianBlur(original_image, (9, 9), 0)
    canny_edges = cv2.Canny(blurred, 70, 150,apertureSize=3)
    
    # ใช้ HoughCircles เพื่อค้นหาวงกลม
    circles = cv2.HoughCircles(canny_edges, 
                               cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=100, param2=20, minRadius=20, maxRadius=100)
    num_circles = 0
    
    # แปลงภาพให้เป็นสีเพื่อแสดงผลวงกลม
    circle_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    # ถ้าพบวงกลม ให้ทำการวาดวงกลมที่ตรวจพบลงในภาพ
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        num_circles = len(circles)
        for (x, y, r) in circles:
            # วาดขอบวงกลม
            cv2.circle(circle_image, (x, y), r, (0, 255, 0), 2)
            # วาดจุดศูนย์กลางวงกลม
            cv2.circle(circle_image, (x, y), 2, (0, 0, 255), 3)
    
    return circle_image,num_circles

# เรียกใช้ฟังก์ชันตรวจจับวงกลม
circle_image,num_circle = hough_circle_detection(img)

# แสดงผลภาพด้วย matplotlib
titles = ["CANNY","CIRCLES"]
images = [canny_edges,circle_image]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) if i > 0 else images[i], cmap='gray')  # แสดงผลภาพ
    plt.title(titles[i])
    plt.axis('off')

plt.subplot(2, 3, 2)
plt.text(30, 50, f"circle = {num_circle}", fontsize=15, color='white', backgroundcolor='black')

plt.tight_layout()
plt.show()
