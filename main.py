import cv2
import time
import pyautogui
from math import sqrt, pow
import winsound
from mediapipe.python.solutions import hands as mp_hands

pyautogui.FAILSAFE = False
sw, sh = pyautogui.size()
ls = lambda lm: (lm.x * sw, lm.y * sh, lm.z * sw)
left_click_down = False
track = False
hand_folded = False


def distance(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2))


def move_mouse(dx, dy, t):
    dxs = - dx * t
    dys = dy * t

    if abs(dx) < 3:
        dxs = 0
    elif abs(dx) < 10:
        dxs = - dx
    elif abs(dx) < 30:
        dxs /= 2

    if abs(dy) < 3:
        dys = 0
    elif abs(dy) < 10:
        dys = dy
    elif abs(dy) < 30:
        dys /= 2

    pyautogui.moveRel(dxs, dys, duration=0)


def left_click(d):
    if d <= 60:
        if not left_click_down:
            pyautogui.mouseDown(button='left')
            return True

    elif left_click_down:
        pyautogui.click()
        pyautogui.mouseUp(button='left')
        return False
    return left_click_down


def right_click(d):
    if d <= 60:
        pyautogui.rightClick()


videoCap = cv2.VideoCapture(0)
lastFrameTime = 0

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5, )

index_tip_pos = pyautogui.position()
while videoCap.isOpened():
    success, img = videoCap.read()
    if success:
        # cv2.flipND(img, 1, img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        thisFrameTime = time.time()
        fps = 1 / (thisFrameTime - lastFrameTime)
        lastFrameTime = thisFrameTime
        cv2.putText(img, f'FPS:{int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        recHands = hands.process(imgRGB)
        h, w, c = img.shape
        if recHands.multi_hand_landmarks:
            hand = recHands.multi_hand_landmarks[0]
            thumb_tip = hand.landmark[4]
            index_tip = hand.landmark[8]
            middle_tip = hand.landmark[12]
            pinky_tip = hand.landmark[20]
            cmc = hand.landmark[1]
            for point in (thumb_tip, index_tip, middle_tip, pinky_tip, cmc):
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(img, (x, y), 3, (255, 0, 255), cv2.FILLED)

            if distance(ls(middle_tip), ls(cmc)) < 80:
                hand_folded = True
                if hand_folded:
                    if track:
                        track = False
                        winsound.Beep(2000, 100)
                        winsound.Beep(1000, 90)
                    else:
                        track = True
                        winsound.Beep(1000, 90)
                        winsound.Beep(2000, 100)
            else:
                hand_folded = False

            if track:
                click_d = distance(ls(middle_tip), ls(thumb_tip))
                r_click_d = distance(ls(pinky_tip), ls(thumb_tip))
                left_click_down = left_click(click_d)
                right_click(r_click_d)

                x, y = int(index_tip.x * sw), int(index_tip.y * sh)
                dx, dy = x - index_tip_pos[0], y - index_tip_pos[1]
                move_mouse(dx, dy, fps)
                index_tip_pos = (x, y)

        # cv2.imshow("CamOutput", img)
        # cv2.waitKey(1)
