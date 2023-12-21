import numpy as np
import cv2
from collections import deque


def setValues(x):
    pass


cv2.namedWindow("Color_detectors", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color_detectors", 600, 400)

cv2.createTrackbar("Upper Hue", "Color_detectors", 180, 180, setValues)
cv2.createTrackbar("Upper Saturation", "Color_detectors", 232, 255, setValues)
cv2.createTrackbar("Upper Value", "Color_detectors", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color_detectors", 159, 180, setValues)
cv2.createTrackbar("Lower Saturation", "Color_detectors", 102, 255, setValues)
cv2.createTrackbar("Lower Value", "Color_detectors", 12, 255, setValues)


bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]


blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0


kernel = np.ones((5, 5), np.uint8)

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
colorIndex = -1  


eraser_mode = False


paintWindow = np.zeros((465, 840, 3), dtype=np.uint8) + 255


for i, color in enumerate(colors):
    y_coord = 78 + i * 73
    paintWindow = cv2.rectangle(paintWindow, (11, y_coord), (81, y_coord + 45), color, -1)


paintWindow = cv2.rectangle(paintWindow, (11, 15), (81, 58), (0, 0, 0), 1)
paintWindow = cv2.rectangle(paintWindow, (11, 390), (81, 445), (200, 200, 200), 1)


cv2.putText(paintWindow, "CLEAR", (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "ERASER", (22, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(0)

points = [bpoints, gpoints, rpoints, ypoints]


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   
    u_hue = cv2.getTrackbarPos("Upper Hue", "Color_detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color_detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color_detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color_detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color_detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color_detectors")
    Upper_hsv = np.array([u_hue, u_saturation, u_value])
    Lower_hsv = np.array([l_hue, l_saturation, l_value])

    frame = cv2.rectangle(frame, (11, 15), (81, 58), (122, 122, 122), -1)
    for i, color in enumerate(colors):
        y_coord = 78 + i * 73
        frame = cv2.rectangle(frame, (11, y_coord), (81, y_coord + 45), color, -1)

    frame = cv2.rectangle(frame, (11, 390), (81, 445), (200, 200, 200), -1)

    cv2.putText(frame, "CLEAR", (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "ERASER", (22, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

  
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

   
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        M = cv2.moments(cnt)
        if M['m00'] != 0:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        
        if center is not None and center[0] <= 81:
            if 15 <= center[1] <= 58: 
                for points_list in points:
                    points_list.append(deque(maxlen=512))
                blue_index = green_index = red_index = yellow_index = 0
                paintWindow[:, 81:, :] = 255
            elif 390 <= center[1] <= 445:  
                eraser_mode = not eraser_mode
                status = "ERASER ON" if eraser_mode else "ERASER OFF"
                cv2.putText(frame, status, (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                for i in range(len(colors)):
                    y_coord = 78 + i * 73
                    if y_coord <= center[1] <= y_coord + 45:
                        colorIndex = i
                        break

        elif center is not None and eraser_mode:
        
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(len(points[i][j])):
                        if points[i][j][k] is not None:
                            point = tuple(map(int, points[i][j][k]))
                            distance = np.linalg.norm(np.array(point) - np.array(center))
                            if distance < 20:  
                                points[i][j][k] = None
                                paintWindow = cv2.circle(paintWindow, point, 20, (255, 255, 255), -1)
        elif center is not None and colorIndex >= 0:
           
            points[colorIndex][blue_index].appendleft(center)

    else:
        for points_list in points:
            points_list.append(deque(maxlen=512))
        blue_index = green_index = red_index = yellow_index = 0

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is not None and points[i][j][k] is not None:
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("mask", Mask)

   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
