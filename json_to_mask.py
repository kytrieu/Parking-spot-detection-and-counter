import json
import cv2
import numpy as np

# --- CẤU HÌNH ---
json_file_path = 'data/9010418-uhd_3840_2160_30fps.json'  
output_mask_path = 'mask/9010418-uhd_3840_2160_30fps.png' 

# Cài đặt độ phân giải mong muốn (Ví dụ: 1920x1080)
# Nếu bạn muốn giữ nguyên độ phân giải gốc, hãy để target_width = None
target_width = None
target_height = None

def create_advanced_mask():
    # 1. Đọc dữ liệu từ file JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    orig_h = data['imageHeight']
    orig_w = data['imageWidth']

    # 2. Xác định độ phân giải cuối cùng để tạo ảnh đen
    w = target_width if target_width else orig_w
    h = target_height if target_height else orig_h

    # Tính tỷ lệ phóng to/thu nhỏ (Scale)
    scale_x = w / orig_w
    scale_y = h / orig_h

    # 3. Tạo nền đen với độ phân giải mới
    mask = np.zeros((h, w), dtype=np.uint8)

    # 4. Duyệt qua từng ô đỗ xe
    for shape in data['shapes']:
        shape_type = shape.get('shape_type', 'polygon')
        pts = shape['points']

        # XỬ LÝ LỖI RECTANGLE: Biến 2 điểm thành 4 điểm
        if shape_type == 'rectangle':
            pt1, pt2 = pts[0], pts[1]
            xmin, ymin = pt1[0], pt1[1]
            xmax, ymax = pt2[0], pt2[1]
            # Tạo lại thành 4 góc khép kín
            pts = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

        # NHÂN TỶ LỆ (SCALE) CHO TỌA ĐỘ MỚI
        scaled_pts = []
        for p in pts:
            new_x = int(p[0] * scale_x)
            new_y = int(p[1] * scale_y)
            scaled_pts.append([new_x, new_y])
        
        points_array = np.array(scaled_pts, dtype=np.int32)
        
        # 5. TÔ KÍN MÀU TRẮNG VÀO BÊN TRONG
        cv2.fillPoly(mask, [points_array], color=255)

    # 6. Lưu ảnh
    cv2.imwrite(output_mask_path, mask)
    print(f"Thành công! Ảnh mask sắc nét kích thước {w}x{h} đã được lưu tại: {output_mask_path}")

if __name__ == '__main__':
    create_advanced_mask()