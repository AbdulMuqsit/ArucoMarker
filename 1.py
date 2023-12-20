import cv2 as cv
import numpy as np
from cv2 import aruco

def moving_average(matrix):
    x = matrix.mean(axis=1)
    matrix[:, 2] = x
    return matrix

calib_data_path = "calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 8  # centimeters

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Use predefined dictionary directly for marker detection parameters
param_markers = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

cap = cv.VideoCapture(0)
all_angles = np.zeros(shape=(3, 3))
frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect markers using aruco.detectMarkers directly
    corners, ids, reject = aruco.detectMarkers(gray_frame, marker_dict)

    if ids is not None:
        rVec, tVec, _ = cv.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)

        if frame_no <= 2:
            all_angles[:, frame_no] = rVec[0, 0]
        else:
            all_angles[:, 0] = all_angles[:, 1]
            all_angles[:, 1] = all_angles[:, 2]
            all_angles[:, 2] = rVec[0, 0]
            all_angles = moving_average(all_angles)

        print(frame_no)
        frame_no += 1

        total_markers = range(0, ids.size)
        for i, (ids, corners) in enumerate(zip(ids, corners)):
            cv.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)

            distance = np.sqrt(tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2)

            point = aruco.drawAxis(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4)

            cv.putText(frame, f"ThetaX:{round(all_angles[:, 2][0] * 180/np.pi, 0)} ThetaY: {round(all_angles[:, 2][1] * 180/np.pi, 0)} ThetaZ: {round(all_angles[:, 2][2] * 180/np.pi, 0)}",
                       (10, 30), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2, cv.LINE_AA)

            print(rVec)
            print(all_angles)

    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
