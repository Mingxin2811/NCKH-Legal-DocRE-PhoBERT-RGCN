# Chạy notebook trên Google Colab

Thư mục này chỉ giữ những mục cần cho người chạy:

```text
google_colab/
├── NCKH_PhoBERT_RGCN_Training.ipynb  # Notebook độc lập, 11 bước
├── README_COLAB.md                 # Hướng dẫn này
├── requirements_colab.txt          # Tham khảo các thư viện
└── dataset/                       # Có thể đặt dữ liệu ở đây, hoặc nơi khác trên Drive
```

**Chỉ cần upload notebook và dataset. Không cần file Python bên ngoài.**
Mã phát triển, tests, dữ liệu mẫu và bản sao lưu được giữ tại `../colab_support/`,
không cần upload thư mục đó lên Colab. Dataset gốc trên máy không bị sửa hay di chuyển.

## Chạy từng bước

1. Mở **bản notebook mới** từ máy trong Colab. Nếu đang mở bản cũ trên Drive, hãy thay
   bằng file `.ipynb` đã cập nhật này và khởi động lại phiên để tránh dùng mã cũ trong RAM.
2. Bật GPU ở **Runtime → Change runtime type**.
3. Chạy **Bước 1** để kết nối Google Drive, sau đó **Bước 2** để cài thư viện.
4. Ở **Bước 3**, chỉ sửa hai đường dẫn cho đúng vị trí Drive của bạn:

   ```python
   DATASET_DIR = Path('/content/drive/MyDrive/NCKH_train/Dataset')
   OUTPUT_DIR = Path('/content/drive/MyDrive/NCKH_Results/raw_doc_run_42')
   ```

   Tên `Dataset`/`dataset` phân biệt hoa thường. Thư mục dữ liệu phải đã giải nén và
   có các file `*_ket_qua.json` trong các thư mục con. Không đặt đường dẫn tới file ZIP.
   JSON cần có `full_text`, `entities` và `relations`, giống dữ liệu trong `dataset/` của dự án.
5. Chạy **Bước 4 → 5 → 6** để nạp schema, kiểm tra dữ liệu và định nghĩa model.
   Bước 5 hiện trực tiếp lỗi dữ liệu kèm tên file nếu có; không gọi script bằng subprocess.
6. Ở đầu cell **Bước 7**, giữ hoặc sửa cấu hình:

   ```python
   NER_EPOCHS = 20
   RELATION_EPOCHS = 30
   ENCODER = 'vinai/phobert-base-v2'
   ENCODER_BATCH = 8
   VARIANTS = 'semantic,rgcn'
   ```

   Chạy cell và đợi hoàn tất. Trình tự: embedding → NER → RE baseline → RE RGCN.
   Nếu hết GPU RAM, giảm `ENCODER_BATCH` xuống 2. Early stopping có thể dừng trước số epoch tối đa.
7. Chạy **Bước 8 → 10** để xem biểu đồ, báo cáo test và bảng so sánh thực đo.
8. Ở **Bước 11**, sửa `NEW_TEXT` hoặc đặt đường dẫn file UTF-8 `.txt` vào `NEW_TEXT_FILE`.
   Cell tự nhận diện thực thể và quan hệ, lưu toàn bộ vào `prediction.json`.
9. Cell **Tùy chọn: Đóng gói & Tải Kết quả** tạo ZIP trên Drive, gồm báo cáo/hình/checkpoint,
   bỏ cache embedding. Đặt `DOWNLOAD_ZIP = True` nếu muốn tải về máy ngay.

Chạy lần lượt từ trên xuống, đợi cell trước xong rồi chạy cell sau. Cần ít nhất 3 thân
văn bản khác nhau để chia train/val/test; 2 văn bản mẫu chỉ đủ kiểm tra định dạng.

## Nếu có lỗi

- **`Không tìm thấy dataset` / `Không có *_ket_qua.json`**: kiểm tra đường dẫn ở Bước 3,
  Drive đã mount chưa, dữ liệu đã giải nén chưa.
- **`Cannot read dataset JSON ...` / `Missing full_text ...` / `Unknown ...`**: tên file
  gây lỗi xuất hiện ngay trong traceback. Sửa file/schema đó; không tự đổi nhãn lạ thành `THUOC`.
- **`Dataset/seed changed`**: dùng `OUTPUT_DIR` mới để tạo split mới, giữ nguyên kết quả cũ.
- **`Chạy xong Bước 7 ...`**: mô hình/báo cáo chưa được tạo; chạy train trước khi vẽ/test.
- Nếu vẫn thấy **`CalledProcessError` gọi `build_document_graph.py`**, bạn đang mở notebook cũ.
  Notebook mới không có lệnh đó và không cần `PROJECT_DIR`.

Traceback `CalledProcessError` cũ chỉ cho biết script ngoài thất bại, không cho biết nguyên nhân
bên trong. Bản này bỏ lớp gọi script để hiển thị trực tiếp lỗi gốc tại cell.

## Đọc kết quả đúng

- `predicted_entities_end_to_end`: kết quả văn bản mới; `gold_entities`: RE khi đã biết thực thể đúng.
- NER được đo exact-span/type sau khi chuẩn hóa khoảng trắng quanh thực thể.
- RE dùng đa nhãn. Không quan hệ = không nhãn dương nào vượt threshold.
- `accuracy_candidates_exact_set` là Accuracy trên cặp ứng viên, cần so với
  `no_relation_baseline_accuracy`; nhiều cặp âm có thể khiến Accuracy cao.
- `precision_macro_22` và `f1_macro_22` luôn tính 22 loại quan hệ dương; lớp không có mẫu
  có điểm 0. `f1_macro_supported` chỉ là chỉ số phụ trên lớp đang có mẫu.
- Ma trận ở Bước 9 hiển thị TP/FP/FN từng loại, phù hợp bài toán đa nhãn.

PhoBERT được đóng băng, chỉ train head NER/RE và RGCN. NER start/end hỗ trợ chồng lấn khác
loại, còn giới hạn với lồng nhau cùng loại hoặc ranh giới giữa lexical unit; các gold này
vẫn được tính khi đánh giá. Candidate RE dùng 8 hàng xóm mỗi phía và 12 thực thể đầu làm
anchor; gold ngoài ứng viên vẫn tính vào FN. Checkpoint/threshold chỉ chọn bằng validation.

Phụ lục `THÔNG TIN THUỘC TÍNH & LƯỢC ĐỒ TỪ FILE JSON` tự sinh chứa đáp án bị loại khỏi tác vụ;
cạnh GNN chỉ dựa vào vị trí, không dùng quan hệ đúng. Điểm dựa trên nhãn cung cấp, chưa phải
test gold được người xác nhận; dữ liệu thiếu nhãn cần được rà soát thủ công.

Trên bộ 174 văn bản cục bộ: sau lọc còn 40.803 thực thể và 39.126 quan hệ; chỉ 6/22 loại
quan hệ có mẫu trong thân. Vì vậy Macro-F1 cố định 22 lớp có trần **27,27%** trên toàn bộ
nhãn đang có (tập test có thể thấp hơn). Muốn 60% đủ 22 lớp cần bổ sung dữ liệu thực sự có
bằng chứng trong thân. Không giữ phần phụ lục lộ đáp án để nâng điểm.

Kết quả/checkpoint/cache nằm trong `OUTPUT_DIR` trên Drive. Khi chạy lại, cache được tái sử dụng,
nhưng các head train từ đầu, chưa tiếp tục epoch bị ngắt. Checkpoint thuộc pipeline cũ phải train lại.
