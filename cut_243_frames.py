import cv2

input_path = "demo/video/stroke_postures/backhand_chop_01.avi"  # 輸入影片路徑
output_path = "demo/video/backhand_chop_01.avi"  # 輸出影片路徑
num_frames = int(243 * 1.5)  # 你想要的總幀數

# 開啟影片
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("無法開啟影片")
    exit()

# 取得影片的參數
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定影片輸出格式
fourcc = cv2.VideoWriter_fourcc(*"FFV1")  # 使用無失真壓縮 FFV1 編碼
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

count = 0
while count < num_frames:
    ret, frame = cap.read()
    if not ret:
        print("影片讀取結束或發生錯誤")
        break
    out.write(frame)
    count += 1

cap.release()
out.release()
print(f"已儲存 {count} 個 frame 到 {output_path}")
