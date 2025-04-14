import cv2
import matplotlib.pyplot as plt

# 打开视频文件
video = cv2.VideoCapture("./car_test.mp4")
# 读取一帧
ret, frame = video.read()


# plt.imshow(frame)
# plt.show()

# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.show()


video = cv2.VideoCapture("./car_test.mp4")
num = 0         # 计数器
save_step = 10  # 间隔帧
while True:
    ret, frame = video.read()
    if not ret:
        break
    num += 1
    if num % save_step == 0:
        cv2.imwrite("./demo_images/" + str(num) + ".jpg", frame)