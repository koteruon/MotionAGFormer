import csv

import cv2

# 初始化
video_path = "demo/video/fhs_left_01.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

annotations = []
start_frame = None
current_frame = 0

# 總影片幀數
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def save_annotations(annotations):
    with open("annotations.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["start_frame", "end_frame"])
        writer.writerows(annotations)
    print("Annotations saved to annotations.csv")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video.")
        break

    # 畫上目前 frame 數量和總共 frame 數量
    frame_number_text = f"Frame: {current_frame}/{total_frames}"
    cv2.putText(
        frame,
        frame_number_text,
        (frame.shape[1] - 300, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 根據 start_frame 和 end_frame 顯示相應的文字
    if start_frame is not None:
        cv2.putText(frame, "Record", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    elif start_frame is None and len(annotations) > 0 and annotations[-1][1] == current_frame:
        cv2.putText(frame, "Standing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    key = cv2.waitKey(0) & 0xFF  # 等待按鍵

    if key == ord("q"):  # 按下 'q' 鍵退出
        break
    elif key == ord(" "):  # 按下空白鍵記錄
        if start_frame is None:
            start_frame = current_frame
            print(f"Start frame: {start_frame}")
        else:
            end_frame = current_frame
            annotations.append((start_frame, end_frame))
            print(f"End frame: {end_frame}")
            start_frame = None
    elif key == 81:  # 左鍵
        current_frame = max(current_frame - 1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    elif key == 83:  # 右鍵
        current_frame = min(current_frame + 1, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

cap.release()
cv2.destroyAllWindows()

# 儲存註釋
if annotations:
    save_annotations(annotations)
else:
    print("No annotations were made.")
