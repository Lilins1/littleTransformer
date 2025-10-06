import cv2
import os
from datetime import datetime

# 指定保存目录
save_dir = "pic/captured"
os.makedirs(save_dir, exist_ok=True)

# 打开摄像头（0表示默认摄像头，如果有多个可以尝试1, 2...）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

# 读取一帧图像
ret, frame = cap.read()
if ret:
    filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"✅ 照片已保存到: {filepath}")
else:
    print("⚠️ 无法读取图像")

cap.release()
cv2.destroyAllWindows()
