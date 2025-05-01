import cv2
import numpy

# 加载预训练的Haar级联分类器来检测眼睛
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 启动摄像头
#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("test.mp4")

threshold_value=50
max_value=255
threshold_type=cv2.THRESH_BINARY_INV
while True:
    # 读取一帧
    ret, frame = cap.read()
    ret, original=cap.read()
    height, width, _ = frame.shape
    a=200
    b=200
    start_x = max(0, width // 2 - int(a/2))
    start_y = max(0, height // 2 - int(b/2))
    end_x = min(start_x + int(2*a), width)
    end_y = min(start_y + int(2*b), height)

    # 截取屏幕中央的100x100像素部分
    frame = frame[start_y:end_y, start_x:end_x]#cv2.resize(frame[start_y:end_y, start_x:end_x],(a*2,b*2))

    frame = cv2.flip(frame, 1)  # 或者使用 cv2.FLIP_HORIZONTAL
    if not ret:
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, threshold_value, max_value, threshold_type)

    # 创建一个结构元素，这里使用3x3的正方形
    kernel = numpy.ones((3, 3), numpy.uint8)

    #binary_image = cv2.erode(binary_image, None, iterations=2)#侵蚀
    #binary_image = cv2.dilate(binary_image, None, iterations=3)#膨胀

    # 检测眼睛
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes)>=2:
        try:
            more=-5
            to_x=200
            to_y=200
            if eyes[0][0]>eyes[1][0]:
                left_eye=cv2.resize(frame[eyes[0][1]-more:eyes[0][1]+eyes[0][3]+more, eyes[0][0]-more:eyes[0][0]+eyes[0][2]+more], (to_x, to_y))
                left_eye_binary = cv2.resize(binary_image[eyes[0][1]-more:eyes[0][1] + eyes[0][3]+more, eyes[0][0]-more:eyes[0][0] + eyes[0][2]+more],(to_x, to_y))
                right_eye=cv2.resize(frame[eyes[1][1]-more:eyes[1][1]+eyes[1][3]+more, eyes[1][0]-more:eyes[1][0]+eyes[1][2]+more], (to_x, to_y))
                right_eye_binary = cv2.resize(binary_image[eyes[1][1]-more:eyes[1][1] + eyes[1][3]+more, eyes[1][0]-more:eyes[1][0] + eyes[1][2]+more],(to_x, to_y))
            else:
                left_eye = cv2.resize(frame[eyes[1][1]-more:eyes[1][1] + eyes[1][3]+more, eyes[1][0]-more:eyes[1][0] + eyes[1][2]]+more,(to_x, to_y))
                left_eye_binary = cv2.resize(binary_image[eyes[1][1]-more:eyes[1][1] + eyes[1][3]+more, eyes[1][0]-more:eyes[1][0] + eyes[1][2]+more],(to_x, to_y))
                right_eye = cv2.resize(frame[eyes[0][1]-more:eyes[0][1] + eyes[0][3]+more, eyes[0][0]-more:eyes[0][0] + eyes[0][2]]+more,(to_x, to_y))
                right_eye_binary = cv2.resize(binary_image[eyes[0][1]-more:eyes[0][1] + eyes[0][3]+more, eyes[0][0]-more:eyes[0][0] + eyes[0][2]]+more,(to_x, to_y))

            l = max(cv2.findContours(left_eye_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)
            r = max(cv2.findContours(right_eye_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)
            cv2.drawContours(left_eye, l, -1, (0, 255, 0), 4)
            cv2.drawContours(right_eye, r, -1, (0, 255, 0), 4)

            if len(l) > 0:
                lx, ly, lw, lh = cv2.boundingRect(l)
                lcx=int(lx+lw/2)
                lcy=int(ly+lh/2)
                cv2.circle(left_eye, (lcx, lcy), 20, (255, 0, 0), 3)
                cv2.rectangle(left_eye, (lx, ly), (lx + lw, ly + lh), (0, 0, 255), 2)
                cv2.line(left_eye, (0, lcy), (to_x, lcy), (255, 255, 0), 2)
                cv2.line(left_eye, (lcx, 0), (lcx, to_y), (255, 255, 0), 2)

            if len(r) > 0:
                rx, ry, rw, rh = cv2.boundingRect(r)
                rcx=int(rx+rw/2)
                rcy=int(ry+rh/2)
                cv2.circle(right_eye, (rcx, rcy), 20, (255, 0, 0), 3)
                cv2.rectangle(right_eye, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
                cv2.line(right_eye, (0, rcy), (to_x, rcy), (255, 255, 0), 2)
                cv2.line(right_eye, (rcx, 0), (rcx, to_y), (255, 255, 0), 2)

            cv2.imshow('left_eye', left_eye)
            cv2.imshow('right_eye', right_eye)
            cv2.imshow('left_eye_binary', left_eye_binary)
            cv2.imshow('right_eye_binary', right_eye_binary)
        except Exception as e:
            print(f"Error {e}")
            pass

    else:
        print("Eyes?")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
    # 遍历每个检测到的眼睛区域
    for (ex, ey, ew, eh) in eyes:
        # 提取眼睛区域
        eyeROI = gray[ey:ey+eh, ex:ex+ew]

        # 这里需要您自己的瞳孔检测逻辑或模型
        # 例如，使用形态学操作、边缘检测等方法来定位瞳孔

        # 假设您已经定位到了瞳孔的位置
        # 瞳孔位置假设为pupil_center
        pupil_center = (ex + ew // 2, ey + eh // 2)

        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

        # 在原始图像上标记瞳孔位置
        cv2.line(frame, (0, pupil_center[1]), (1000, pupil_center[1]), (0, 255, 0), 3)
        cv2.line(frame, (pupil_center[0], 0), (pupil_center[0], 1000), (0, 255, 0), 3)

    # 显示结果

    cv2.imshow('original', original)
    cv2.imshow('binary_image', binary_image)
    cv2.imshow('gray', gray)
    cv2.imshow('Pupil', frame)

    # 等待按键，如果按下'q'则退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if cv2.waitKey(1) & 0xFF == ord('w'):
    #     n+=1
    #     if(n<0):n=0
    #     print(f"侵蚀：{n}")
    # if cv2.waitKey(1) & 0xFF == ord('s'):
    #     n-=1
    #     if(n<0):n=0
    #     print(f"侵蚀：{n}")
    # if cv2.waitKey(1) & 0xFF == ord('e'):
    #     m+=1
    #     if(m<0):m=0
    #     print(f"膨胀：{m}")
    # if cv2.waitKey(1) & 0xFF == ord('d'):
    #     m-=1
    #     if(m<0):m=0
    #     print(f"膨胀：{m}")
    #threshold_value=(threshold_value+1)%255
    #print(threshold_value)

# 释放摄像头
cap.release()
cv2.destroyAllWindows()