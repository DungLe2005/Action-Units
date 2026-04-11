import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import mediapipe as mp
import logging

# --- CẤU HÌNH MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
# -------------------------

def detect_face(img):
    """
    Phát hiện khuôn mặt sử dụng MediaPipe Face Mesh.
    Trả về bounding box và facial landmarks (mắt) để align.
    """
    results = face_mesh_detector.process(img)
    
    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img.shape
    
    # Tính bounding box bao quanh các landmarks
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bbox = [x1, y1, x2, y2]
    
    # MediaPipe refine_landmarks=True cung cấp điểm 473 (trung tâm mắt phải người / bên trái ảnh) 
    # và điểm 468 (trung tâm mắt trái người / bên phải ảnh)
    if len(landmarks) >= 474:
        left_eye_img = (int(landmarks[473].x * w), int(landmarks[473].y * h))
        right_eye_img = (int(landmarks[468].x * w), int(landmarks[468].y * h))
    else:
        # Lấy trung bình vùng mắt nếu thiếu refine_landmarks
        left_eye_indices = [362, 263, 384, 385, 386, 387, 388, 466, 390, 373, 374, 380, 381, 382]
        right_eye_indices = [33, 133, 157, 158, 159, 160, 161, 246, 163, 144, 145, 153, 154, 155]
        
        left_x = int(np.mean([landmarks[i].x * w for i in left_eye_indices]))
        left_y = int(np.mean([landmarks[i].y * h for i in left_eye_indices]))
        left_eye_img = (left_x, left_y)
        
        right_x = int(np.mean([landmarks[i].x * w for i in right_eye_indices]))
        right_y = int(np.mean([landmarks[i].y * h for i in right_eye_indices]))
        right_eye_img = (right_x, right_y)
        
    landmarks_dict = {
        'left_eye': left_eye_img,   # Mắt nằm ở nửa bên trái của tấm ảnh
        'right_eye': right_eye_img  # Mắt nằm ở nửa bên phải của tấm ảnh
    }

    return bbox, landmarks_dict

def get_rotation_matrix(left_eye, right_eye):
    """
    Tính ma trận xoay để đưa 2 mắt về phương ngang.
    """
    # Tính góc nghiêng giữa 2 mắt
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Tính điểm trung tâm của 2 mắt
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    # Tạo ma trận quay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return M, angle

def align_face(img, landmarks):
    """
    Xoay ảnh để hai mắt nằm ngang.
    """
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    # Lấy ma trận xoay
    M, _ = get_rotation_matrix(left_eye, right_eye)
    
    # Kích thước ảnh gốc
    h, w = img.shape[:2]
    
    # Xoay toàn bộ ảnh
    aligned_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    #Xoay ngược lại 180 độ
    aligned_img = cv2.rotate(aligned_img, cv2.ROTATE_180)

    return aligned_img, M

def crop_face(img, bbox, margin=0.2):
    """
    Cắt khuôn mặt với viền margin.
    bbox: [x1, y1, x2, y2]
    margin: tỷ lệ mở rộng thêm
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    
    face_w = x2 - x1
    face_h = y2 - y1
    
    # Nới rộng theo margin (20%)
    margin_w = int(face_w * margin)
    margin_h = int(face_h * margin)
    
    new_x1 = max(0, x1 - margin_w)
    new_y1 = max(0, y1 - margin_h)
    new_x2 = min(w, x2 + margin_w)
    new_y2 = min(h, y2 + margin_h)
    
    # Làm cho khung hình cắt là hình vuông 
    # (vì model CLIP nhận input 224x224, cắt vuông trước khi resize để tránh padding/bóp méo nhiều)
    crop_w = new_x2 - new_x1
    crop_h = new_y2 - new_y1
    
    if crop_w > crop_h:
        diff = crop_w - crop_h
        new_y1 = max(0, new_y1 - diff // 2)
        new_y2 = min(h, new_y2 + diff // 2)
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        new_x1 = max(0, new_x1 - diff // 2)
        new_x2 = min(w, new_x2 + diff // 2)
        
    cropped_img = img[new_y1:new_y2, new_x1:new_x2]
    return cropped_img

def process_image(image_path, output_path, margin=0.2, target_size=(224, 224), debug=False):
    """
    Chạy chu trình đầy đủ trên một bức ảnh: Đọc -> Detect -> Align -> Crop -> Resize -> Lưu.
    """
    # Đọc ảnh
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "Failed to read image"
    
    # OpenCV mặc định đọc ảnh dưới format BGR, nhưng RetinaFace / hiển thị chuẩn thường dùng RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Phát hiện mặt trên ảnh gốc
    bbox, landmarks = detect_face(img_rgb)
    if bbox is None:
        return False, "No face detected in original image"
    
    # 2. Align ảnh xoay ngang mắt
    aligned_img, M = align_face(img_rgb, landmarks)
    
    # 3. Sau khi xoay tổng thể, vị trí bounding box bị lệch. 
    # Chúng ta chạy Detect lần 2 trên ảnh SẠCH ĐÃ ĐƯỢC XOAY NGANG để lấy bbox chính xác.
    # Tuy mất thêm tẹo chi phí tính toán nhưng cho ra kết quả bám mặt chuẩn nhất.
    new_bbox, _ = detect_face(aligned_img)
    if new_bbox is None:
        # Fallback: nếu lần 2 không tìm thấy (hiếm khi xảy ra), ta dùng ảnh align nhưng tự xoay bbox cũ
        # Hoặc đơn giản là fallback xài luôn crop margin trên ảnh bbox cũ chưa align.
        # Nhưng để an toàn ta báo lỗi skip ảnh này.
        return False, "No face detected after alignment"

    # 4. Crop khuôn mặt (kèm margin 20% và ưu tiên hình vuông)
    cropped_img = crop_face(aligned_img, new_bbox, margin=margin)
    if cropped_img.size == 0:
         return False, "Cropped image is empty"
        
    # 5. Resize về 224x224 cho CLIP training
    final_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_CUBIC)
    
    # (Optional) Debug Mode: Show Visualization
    if debug:
        show_debug_visualization(img_rgb, bbox, aligned_img, new_bbox, final_img)
    
    # Lưu ra thư mục output (chuyển lại về BGR để lưu theo OpenCV chuẩn)
    # Tạo các thư mục con còn thiếu
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), final_img_bgr)
    
    return True, "Success"

def show_debug_visualization(img_rgb, orig_bbox, aligned_img, aligned_bbox, final_img):
    """
    Hỗ trợ hiển thị trực quan nếu cần kiểm tra luồng crop.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ảnh gốc có bbox
    ax_orig = axs[0]
    ax_orig.imshow(img_rgb)
    x1, y1, x2, y2 = orig_bbox
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
    ax_orig.add_patch(rect)
    ax_orig.set_title("1. Original + Bounding Box")
    ax_orig.axis('off')
    
    # Ảnh đã được xoay có bbox mới
    ax_align = axs[1]
    ax_align.imshow(aligned_img)
    x1, y1, x2, y2 = aligned_bbox
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='green', linewidth=2)
    ax_align.add_patch(rect)
    ax_align.set_title("2. Aligned Face")
    ax_align.axis('off')
    
    # Kết quả cuốii cùng
    ax_final = axs[2]
    ax_final.imshow(final_img)
    ax_final.set_title("3. Cropped & Resized (224x224)")
    ax_final.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_dataset(input_dir, output_dir, margin=0.2, debug=False, log_file="no_face_detected.txt", labels_file=None):
    """
    Càn quét toàn bộ dataset, đọc theo cấu trúc fold, và lưu song song.
    Cập nhật luôn file labels.csv loại bỏ các ảnh không có khuôn mặt (nếu được cung cấp).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Lấy toàn bộ danh sách các file ảnh (jpg, jpeg, png)
    image_extensions = {".jpg", ".jpeg", ".png"}
    all_images = [p for p in input_path.rglob("*") if p.suffix.lower() in image_extensions]
    
    if len(all_images) == 0:
        print(f"Không tìm thấy file ảnh nào trong thư mục {input_dir}")
        return

    print(f"Tìm thấy {len(all_images)} ảnh. Bắt đầu tiền xử lý...")
    
    failed_images = []
    successful_rel_paths = set()
    
    # Hiển thị progress bar với tqdm
    for img_p in tqdm(all_images, desc="Processing Faces"):
        # Lấy đường dẫn tương đối để bảo lưu cấu trúc gốc
        rel_path = img_p.relative_to(input_path)
        out_p = output_path / rel_path
        
        # Gọi hàm xử lý
        success, msg = process_image(str(img_p), str(out_p), margin=margin, debug=debug)
        
        if success:
            # Lưu lại tên file theo chuẩn dấu gạch chéo để đối chiếu CSV (vd: SN009/A7.../046.jpg)
            successful_rel_paths.add(rel_path.as_posix())
        else:
            failed_images.append(f"{str(img_p)} - Reason: {msg}")
            
    # Xử lý cập nhật file CSV nếu người dùng truyền vào
    if labels_file and os.path.exists(labels_file):
        print(f"\nĐang đồng bộ hóa file nhãn: {labels_file}...")
        try:
            df = pd.read_csv(labels_file)
            image_col = df.columns[0] # Giả thiết cột đầu tiên luôn là đường dẫn ảnh
            original_len = len(df)
            
            # Lọc dataframe chỉ giữ lại những dòng có "đường dẫn tương đối" nằm trong tập xử lý thành công
            df['is_valid'] = df[image_col].apply(lambda x: Path(x).as_posix() in successful_rel_paths)
            df_filtered = df[df['is_valid']].drop(columns=['is_valid'])
            
            new_labels_path = Path(output_dir).parent / Path(labels_file).name # Lưu ra thư mục cha của output_dir
            df_filtered.to_csv(new_labels_path, index=False)
            
            print(f"[Thành công] Đã tạo file CSV mới tại: {new_labels_path}")
            print(f"-> Giữ lại {len(df_filtered)}/{original_len} dòng hợp lệ (loại bỏ {original_len - len(df_filtered)} ảnh mồ côi).")
        except Exception as e:
            print(f"[Lỗi] Không thể cập nhật file CSV: {e}")
            
    # Ghi log file những ảnh hỏng / không nhận diện được mặt
    if failed_images:
        with open(log_file, "w", encoding='utf-8') as f:
            for item in failed_images:
                f.write(f"{item}\n")
        print(f"\n[Cảnh báo] Có {len(failed_images)} ảnh không thể xử lý.")
        print(f"Chi tiết đã được lưu trong file: {log_file}")
    else:
        print("\n[Hoàn tất] Toàn bộ dữ liệu đã được tiền xử lý thành công!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing cho CLIP: Face Detection, Alignment & Cropping")
    parser.add_argument("--input", type=str, required=True, help="Thư mục chứa dataset gốc (Images)")
    parser.add_argument("--output", type=str, required=True, help="Thư mục xuất ảnh sau khi xử lý")
    parser.add_argument("--labels", type=str, default=None, help="(Optional) Đường dẫn tới file labels.csv. Script sẽ tự động xóa các dòng thừa và lưu file mới.")
    parser.add_argument("--margin", type=float, default=0.2, help="Lề an toàn quanh khuôn mặt, mặc định: 0.2 (20%)")
    parser.add_argument("--debug", action="store_true", help="Nếu bật, sẽ mở cửa sổ visualize các bước cho từng ảnh (phù hợp test file nhỏ)")
    parser.add_argument("--log", type=str, default="no_face_detected.txt", help="Tên file log ghi lại các ảnh bị lỗi/bỏ qua")
    
    args = parser.parse_args()
    
    # Để sử dụng được RetinaFace tối ưu và giảm cảnh báo
    import warnings
    warnings.filterwarnings("ignore")
    
    process_dataset(args.input, args.output, margin=args.margin, debug=args.debug, log_file=args.log, labels_file=args.labels)
