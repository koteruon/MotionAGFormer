import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# 定義全局變量
points = []
table_width = 2.74 / 2  # 桌子的寬度 (米)
table_height = 1.525  # 桌子的長度 (米)


# 鼠標點擊事件處理函數
def click_event(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Frame", frame)
        if len(points) == 4:  # 左上、右上、右下、左下
            print("四個角點已經選擇完成")
            cv2.destroyAllWindows()
            calculate_pose()


# 計算旋轉四元數
def calculate_pose():
    global points
    if len(points) != 4:
        print("請選擇四個角點")
        return

    # 轉換點到NumPy數組
    image_points = np.array(points, dtype=np.float32)

    # 假設的相機內參數
    focal_length = 800
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    # 桌子在3D世界中的角點坐標
    world_points = np.array(
        [[0, 0, 0], [table_width, 0, 0], [table_width, table_height, 0], [0, table_height, 0]], dtype=np.float32
    )

    # 使用PnP求解旋轉和位移
    success, rotation_vector, translation_vector = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)

    # 將旋轉向量轉換為旋轉矩陣
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 將旋轉矩陣轉換為四元數
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # 返回 [x, y, z, w]
    print("Quaternion:", quat)


# 讀取影片
video_path = "demo/video/bhpull/bhpull_01.mp4"
cap = cv2.VideoCapture(video_path)

# 獲取影片總幀數
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 計算中間幀
middle_frame_index = total_frames // 2

# 設置到中間幀
cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

# 讀取中間幀
ret, frame = cap.read()

if not ret:
    print("不能讀取中間幀")
    cap.release()
    cv2.destroyAllWindows()
else:
    # 顯示中間幀並等待用戶點擊
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 釋放資源
cap.release()
