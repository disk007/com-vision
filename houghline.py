import matplotlib.pyplot as plt
import numpy as np
import cv2 

image = cv2.imread('./line4.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurred_img = cv2.GaussianBlur(img, (17, 17), 0) #ทำการเบลอภาพเพื่อช่วยลด noise
canny_edges = cv2.Canny(blurred_img, 70, 150,apertureSize=3) #หาขอบในภาพ



def hough_line_detection(edges, original_image):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=120, maxLineGap=20)  #ตรวจจับเส้น
    line_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR) 
    num_lines = 0 
    if lines is not None:
        num_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2) 
    return line_image, num_lines


line_image, num_lines = hough_line_detection(canny_edges, img)


titles = ["CANNY", "LINES"]
images = [canny_edges, line_image]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) if i > 0 else images[i], cmap='gray') 
    plt.title(titles[i])
    plt.axis('off')

plt.subplot(2, 3, 2)
plt.text(30, 50, f"line = {num_lines}", fontsize=15, color='white', backgroundcolor='black')

plt.tight_layout()
plt.show()
