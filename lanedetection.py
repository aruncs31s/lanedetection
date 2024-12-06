import cv2
import numpy as np

import utlis
from MotorModule import Motor  # Ensure this import is correct

# Initialize motor control with GPIO pins
motor = Motor(2, 3, 4, 10, 22, 27)  # Adjust these pins based on your setup

# Curve tracking parameters
curveList = []
avgVal = 10


def getLaneCurve(img, display=2):
    imgCopy = img.copy()
    imgResult = img.copy()

    # Step 1: Thresholding
    imgThres = utlis.thresholding(img)

    # Step 2: Warping Image
    hT, wT, c = img.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)

    # Step 3: Histogram and Curve Calculation
    middlePoint, imgHist = utlis.getHistogram(
        imgWarp, display=True, minPer=0.5, region=4
    )
    curveAveragePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint

    # Step 4: Averaging Curve
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    # Step 5: Overlay for Display
    if display != 0:
        imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0 : hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(
            imgResult,
            str(curve),
            (wT // 2 - 80, 85),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (255, 0, 255),
            3,
        )
        cv2.line(
            imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5
        )
        cv2.line(
            imgResult,
            ((wT // 2 + (curve * 3)), midY - 25),
            (wT // 2 + (curve * 3), midY + 25),
            (0, 255, 0),
            5,
        )
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(
                imgResult,
                (w * x + int(curve // 50), midY - 10),
                (w * x + int(curve // 50), midY + 10),
                (0, 0, 255),
                2,
            )

    # Display Options
    if display == 2:
        imgStacked = utlis.stackImages(
            0.7, ([img, imgWarpPoints, imgWarp], [imgHist, imgLaneColor, imgResult])
        )
        cv2.imshow("ImageStack", imgStacked)
    elif display == 1:
        cv2.imshow("Result", imgResult)

    # Normalization
    curve = curve / 100
    if curve > 1:
        curve = 1
    if curve < -1:
        curve = -1

    return curve


def main():
    # Start video capture using Iriun Webcam (replace 0 with 1 if needed)
    cap = cv2.VideoCapture(0)
    initialTrackBarVals = [102, 80, 20, 214]
    utlis.initializeTrackbars(initialTrackBarVals)
    frameCounter = 0

    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (480, 240))
        curve = getLaneCurve(img, display=2)

        # Control motor based on curve value
        sensitivity = 1.3  # SENSITIVITY for turning
        maxSpeed = 0.3  # MAX SPEED for the bot

        # Limit the curve value to a maximum range
        if curve > maxSpeed:
            curve = maxSpeed
        elif curve < -maxSpeed:
            curve = -maxSpeed

        # Adjust sensitivity for sharper turns
        if curve > 0:
            sensitivity = 1.7
            if curve < 0.05:  # Ignore small values
                curve = 0
        else:
            if curve > -0.08:  # Ignore small values
                curve = 0

        # Move motor based on curve and sensitivity
        motor.move(0.20, -curve * sensitivity, 0.05)

        # Print the curve value for debugging
        print("Curve:", curve)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    motor.stop()  # Stop the motor when exiting


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted. Stopping the motor.")
        motor.stop()
