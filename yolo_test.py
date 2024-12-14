import cv2
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="XQYfh3L44kZKAAaDu3qq"
)

# 載入影片
video_path = "static/1214.mp4"
cap = cv2.VideoCapture(video_path)

# 設定輸出影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 保存圖片並送入模型進行推論
    cv2.imwrite("frame.jpg", frame)
    result = CLIENT.infer("frame.jpg", model_id="best-basketball-shot/2")

    # 繪製預測框
    for prediction in result['predictions']:
        x, y = int(prediction['x']), int(prediction['y'])   
        w, h = int(prediction['width']), int(prediction['height'])
        label = f"{prediction['class']} ({prediction['confidence']:.2f})"

        # 繪製矩形框
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 顯示影片畫面
    cv2.imshow("Video Detection", frame)
    out.write(frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()

