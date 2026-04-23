import cv2

# Thay 'video_bai_do_xe.mp4' bằng tên file video của bạn
videoPath = 'data/9010418-uhd_3840_2160_30fps'
cap = cv2.VideoCapture( videoPath + '.mp4')

# Đọc frame đầu tiên
success, frame = cap.read()

if success:
    # Lưu frame ra thành 1 file ảnh
    cv2.imwrite( videoPath +'.jpg', frame)
    
    # In ra kích thước để kiểm tra cho chắc (ví dụ: 1080, 1920, 3)
    print("Đã lưu ảnh thành công. Kích thước (Cao, Rộng, Kênh màu):", frame.shape)
else:
    print("Không thể đọc được video, kiểm tra lại đường dẫn!")

cap.release()