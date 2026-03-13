# Hướng Dẫn Chuyển Đổi Facial Action Units (AUs) thành Text Prompts

Tài liệu này hướng dẫn cách sử dụng các dự đoán Action Unit (dạng nhị phân 0/1) từ mô hình CLIP đã huấn luyện để tạo ra các mô tả (prompts) bằng ngôn ngữ tự nhiên. Các mô tả này có thể được sử dụng cho các mô hình tạo ảnh (như Stable Diffusion, Midjourney, v.v.) nhằm hướng dẫn chi tiết cách tạo ra hoặc chỉnh sửa biểu cảm trên khuôn mặt.

## 1. Bản Đồ Action Units (AUs) sang Ngôn Ngữ Tự Nhiên

Mỗi AU tương ứng với một chuyển động cơ mặt cụ thể. Để tạo prompt hiệu quả, chúng ta cần ánh xạ mỗi AU đang "active" (giá trị dự đoán = 1) thành một cụm từ mô tả hành động đó.

Dưới đây là bảng ánh xạ phổ biến (bạn có thể điều chỉnh tùy theo 12 AUs mà dataset của bạn đang sử dụng):

| AU Mã Số | Tên Tiếng Anh        | Mô Tả Prompt (Tiếng Anh - Thường dùng cho AI tạo ảnh)       | Mô tả tiếng Việt (Tham khảo)            |
| -------- | -------------------- | ----------------------------------------------------------- | --------------------------------------- |
| AU 1     | Inner Brow Raiser    | "raised inner eyebrows"                                     | Nhướng phần trong của lông mày          |
| AU 2     | Outer Brow Raiser    | "raised outer eyebrows"                                     | Nhướng phần ngoài của lông mày          |
| AU 4     | Brow Lowerer         | "lowered and pulled together eyebrows", "frowning eyebrows" | Cau mày, hạ lông mày                    |
| AU 6     | Cheek Raiser         | "raised cheeks", "squinting eyes slightly"                  | Nâng gò má (thường đi kèm nụ cười thật) |
| AU 7     | Lid Tightener        | "tightened eyelids", "narrowed eyes"                        | Siết chặt mí mắt                        |
| AU 9     | Nose Wrinkler        | "wrinkled nose"                                             | Nhăn mũi                                |
| AU 10    | Upper Lip Raiser     | "raised upper lip"                                          | Nâng môi trên                           |
| AU 12    | Lip Corner Puller    | "smiling", "pulled up lip corners"                          | Kéo khóe miệng (cười)                   |
| AU 15    | Lip Corner Depressor | "sad mouth", "pulled down lip corners"                      | Trễ khóe miệng xuống                    |
| AU 17    | Chin Raiser          | "pushed up chin", "pouting"                                 | Đẩy cằm lên, bĩu môi                    |
| AU 20    | Lip Stretcher        | "stretched lips horizontally"                               | Kéo căng môi ngang                      |
| AU 24    | Lip Presser          | "pressed lips together tightly"                             | Bặm môi                                 |
| AU 25    | Lips Part            | "parted lips", "slightly open mouth"                        | Mở hé môi                               |
| AU 26    | Jaw Drop             | "dropped jaw", "open mouth"                                 | Há miệng                                |

## 2. Cách Xây Dựng Prompt Từ Kết Quả Dự Đoán

Ý tưởng cốt lõi là:

1. Xác định các AU có giá trị `1` (Active).
2. Lấy cụm từ tiếng Anh tương ứng với các AU đó.
3. Ghép các cụm từ đó vào một "khung" (template) prompt có sẵn.

### Template Prompt Mẫu

```text
A close-up portrait of a person with {cụm từ AU 1}, {cụm từ AU 2}, and {cụm từ AU 3}, highly detailed facial features.
```

Ví dụ, nếu mô hình dự đoán AU 6 và AU 12 là `1` (các AU khác là `0`):

- **AU 6:** "raised cheeks"
- **AU 12:** "smiling"
- **Kết quả Prompt:** `"A close-up portrait of a person with raised cheeks and smiling, highly detailed facial features."`

## 3. Mã Nguồn Mẫu (Python)

Bạn có thể thêm đoạn code này vào file `inference.py` để mô hình tự động sinh ra prompt sau khi dự đoán.

```python
# Danh sách từ điển ánh xạ AU index sang text prompt (Cần sửa lại theo đúng thứ tự 12 classes của bạn!)
# Giả sử đây là 12 AUs trong dataset của bạn:
AU_PROMPT_MAPPING = {
    0: "raised inner eyebrows",       # Vị dụ: AU 1
    1: "raised outer eyebrows",       # Ví dụ: AU 2
    2: "frowning eyebrows",           # Ví dụ: AU 4
    3: "raised cheeks",               # Ví dụ: AU 6
    4: "tightened eyelids",           # Ví dụ: AU 7
    5: "wrinkled nose",               # Ví dụ: AU 9
    6: "raised upper lip",            # Ví dụ: AU 10
    7: "smiling",                     # Ví dụ: AU 12
    8: "pulled down lip corners",     # Ví dụ: AU 15
    9: "pressed lips together",       # Ví dụ: AU 24
    10: "parted lips",                # Ví dụ: AU 25
    11: "dropped jaw"                 # Ví dụ: AU 26
}

def generate_prompt_from_binaries(binary_preds, base_prompt="A portrait of a face showing"):
    """
    Sinh ra text prompt từ mảng nhị phân dự đoán Action Units.

    Args:
        binary_preds (list hoặc numpy array): Mảng chứa [0, 1, 0...] kích thước bằng số classes.
        base_prompt (str): Cụm từ mở đầu cho prompt.
    """
    active_descriptions = []

    # Duyệt qua các dự đoán, nếu == 1 thì lấy mô tả tương ứng
    for i, is_active in enumerate(binary_preds):
        if is_active == 1:
            # Lấy mô tả nếu index tồn tại trong mapping
            description = AU_PROMPT_MAPPING.get(i)
            if description:
                active_descriptions.append(description)

    if not active_descriptions:
        return base_prompt + " a neutral expression."

    # Nối các mô tả lại với nhau bằng dấu phẩy và chữ 'and'
    if len(active_descriptions) == 1:
        features_str = active_descriptions[0]
    else:
        features_str = ", ".join(active_descriptions[:-1]) + ", and " + active_descriptions[-1]

    final_prompt = f"{base_prompt} {features_str}."
    return final_prompt

# --- Cách sử dụng trong inference.py ---
# Giả sử `binary_preds` là output từ method `infer.predict()`
# binary_preds = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] # Chẳng hạn AU6 và AU12 active
#
# prompt = generate_prompt_from_binaries(binary_preds)
# print("Generated Prompt:", prompt)
# Output: Generated Prompt: A portrait of a face showing raised cheeks, and smiling.
```

## 4. Ứng Dụng Hướng Dẫn Chỉnh Sửa Ảnh (Image Modification)

Nếu mục tiêu của bạn là **chỉnh sửa một bức ảnh hiện có** (ví dụ thông qua Inpainting hoặc InstructPix2Pix), bạn có thể thay đổi `base_prompt` thành các dạng mệnh lệnh:

- **Chỉnh sửa (Instruct):** `make the person have {danh_sách_AUs}`
  - _Ví dụ:_ `"Make the person have wrinkled nose and raised upper lip."\*
- **Thêm cảm xúc:** `add {danh_sách_AUs} to the face`

### Ghi chú Quan trọng:

Bạn **cực kỳ cần thiết phải kiểm tra lại config** của file dữ liệu (thường nằm ở `labels.csv` hoặc tài liệu mô tả dataset) để biết chính xác `AU_0` đến `AU_11` của bạn tương ứng với mã số AU nào trong hệ thống FACS (Facial Action Coding System). Nếu ánh xạ sai, prompt được tạo ra sẽ không khớp với đặc điểm khuôn mặt thực tế. Mảng `AU_PROMPT_MAPPING` trong code bên trên là một ví dụ mẫu cần tùy chỉnh!
