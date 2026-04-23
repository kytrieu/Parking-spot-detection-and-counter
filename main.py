import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask = "mask/mask_1920_1080.png"
video_path = "data/parking_1920_1080.mp4"

mask = cv2.imread(mask, 0)

cap = cv2.VideoCapture(video_path)

connect_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connect_components)
spots_status = [None for _ in spots]
diffs = [None for _ in spots]
previous_frame = None

frame_number = 0
ret = True
step = 30

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if frame_number % step == 0 and previous_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        print([diffs[j] for j in np.argsort(diffs)][::-1])
        plt.figure()
        plt.hist([diffs[j] / np.amax(diffs) for j in np.argsort(diffs)][::-1])
        if frame_number == 300:
            plt.show()
    if frame_number % step == 0:
        if previous_frame is None:
            array_ = range(len(spots))
        else:
            array_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_idx in array_:
            spot = spots[spot_idx]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            spot_status = empty_or_not(spot_crop)

            spots_status[spot_idx] = spot_status

    if frame_number % step == 0:
        previous_frame = frame.copy()

    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        x1, y1, w, h = spots[spot_idx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.putText(frame, "Available spots: {} / {}".format(str(sum(spots_status)), str(len(spots_status))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

