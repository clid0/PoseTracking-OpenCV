import cv2
import mediapipe as mp
import time
import math


class poseDetector():
    def __init__(self, mode=False, complexity=1, smooth=True, enableSeg=False, smoothSeg=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode, self.complexity, self.smooth, self.enableSeg, self.smoothSeg, self.detectionCon, self.trackCon)

    # Pose 찾기(존재하는 사람 찾기와 비슷)
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # pose가 존재하면
        if self.results.pose_landmarks:
            if draw:
                # 이미지에 landmarks 표시
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        # landmarks표시된 이미지를 반환
        return img

    # 관절 찾기
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList


def main():
    # cap = cv2.VideoCapture('video/2.mp4')
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector(enableSeg=False, smoothSeg=False)
    while True:
        success, img = cap.read()
        # 사이즈가 너무 크니 줄이고
        img = cv2.resize(img, (720, 480))
        img = detector.findPose(img)    # 포즈 찾은 img
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # 13번째 point를 출력
            print(lmList[13])
            # 13번째 point를 15사이즈로 키워서 출력
            cv2.circle(img, (lmList[13][1], lmList[13]
                       [2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
