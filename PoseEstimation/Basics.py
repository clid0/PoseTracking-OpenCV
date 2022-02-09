import cv2
import mediapipe as mp
import time
mpDraw = mp.solutions.drawing_utils  # printing을 하기 위한
mpPose = mp.solutions.pose
'''
Pose init
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
'''
pose = mpPose.Pose()
# cap = cv2.VideoCapture('video/1.mp4')
cap = cv2.VideoCapture(0)
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR -> RGB
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        # 랜드마크 점 찍고, line들을 잇는 실시간 점과 선
        mpDraw.draw_landmarks(img, results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)
        # 랜드마크마다 좌표들이 잡힐텐데 그것들을 잡아줌
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)   # id : 랜드마크 point, lm : 랜드마크들의 정규화된 좌표
            cx, cy = int(lm.x * w), int(lm.y * h)   # 픽셀에 맞춘 좌표
            # 랜드마크에 원그리기
            cv2.circle(img, (cx, cy), 1, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)  # put fps
    cv2.imshow("Image", img)
    cv2.waitKey(1)
