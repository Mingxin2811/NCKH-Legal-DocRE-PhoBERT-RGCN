# PhoBERT-base-v2 + Sliding Window + RGCN trên Google Colab

Notebook độc lập: chỉ upload `NCKH_PhoBERT_RGCN_Training.ipynb`, không cần các file Python
trong `colab_support/`. Dataset và kết quả nằm trên Drive.

## Chuẩn bị và chạy 11 bước

Mở notebook mới, chọn **Runtime → Change runtime type → GPU**. Dataset cần giải nén,
có `*_ket_qua.json` chứa `full_text`, `entities`, `relations`. CSV là bản đối chiếu,
không được đếm thêm thành tài liệu. URL thư mục Drive không phải đường dẫn dữ liệu;
hãy thêm thư mục được chia sẻ vào My Drive hoặc sao chép/giải nén vào nơi truy cập được.

1. **Kết nối Drive:** chạy Bước 1, cấp quyền cho Colab.
2. **Cài thư viện:** chạy Bước 2; nếu Colab yêu cầu restart thì restart, kết nối lại Drive.
3. **Đường dẫn:** sửa `DATASET_DIR` và đặt `OUTPUT_DIR` mới. Giữ
   `TRAIN_NEW_MODEL = True`, `TRAIN_SEED = 42`.

   ```python
   DATASET_DIR = Path('/content/drive/MyDrive/NCKH_train/Dataset')
   OUTPUT_DIR = Path('/content/drive/MyDrive/NCKH_Results/phobert_rgcn_v2_run_42')
   ```

   Tên phân biệt hoa/thường. Không trỏ tới ZIP. Mỗi lần train dùng thư mục mới;
   chương trình từ chối ghi đè một run đã có `run_config.json`.
4. **Schema:** nạp 40 loại thực thể và 22 loại quan hệ.
5. **Kiểm tra dữ liệu:** xem số văn bản và số mẫu theo nhãn. Chia khoảng **80/10/10** theo
   nhóm thân trùng chính xác sau chuẩn hóa; không nhân bản khi thiếu 3 thân khác nhau.
   Kiểm tra thêm phiên bản gần trùng và các lớp hiếm. Audit kỹ thuật không chứng nhận
   nhãn đúng ngữ nghĩa. Dataset gốc không bị sửa.
6. **Kiến trúc:** PhoBERT-base-v2 đóng băng, cửa sổ 256 (254 nội dung + BOS/EOS), stride
   128; NER start/end, RGCN 2 tầng, gated fusion và 22 đầu ra sigmoid.
7. **Train:** NER tối đa 20 epoch, RE 30, patience 5. Train cả semantic và rgcn. Thiếu
   VRAM thì giảm `ENCODER_BATCH` xuống 2. NER chọn theo exact-span F1; RE chọn theo
   micro-F1 validation với thực thể NER dự đoán. Ngưỡng 0.2/0.35/0.5/0.65/0.8 chỉ chọn
   bằng validation. Mặc định tối đa 5 cặp âm/cặp dương; ứng viên gồm hàng xóm, neo và
   span bao chứa theo offset. Quan hệ ngoài ứng viên vẫn tính FN khi đánh giá.
8. **Đường học:** NER và RE vẽ riêng loss và F1 validation; hình lưu PNG.
9. **Test chính:** NER P/R/F1 exact-span/type; RE micro-P/R/F1 và Macro-F1 22 nhãn.
   `predicted_entities_end_to_end` là kết quả chính; `gold_entities` chỉ đo riêng RE.
10. **Phân tích lỗi:** F1 theo loại thực thể và TP/FP/FN theo quan hệ. Bảng chi tiết lưu
    CSV; bật `SHOW_DETAILED_TABLES` nếu muốn in thêm.
11. **DOCX mới:** để `INPUT_FILE = None`, chạy và chọn đúng một DOCX/TXT từ máy; hoặc đặt
    đường dẫn DOCX trên Drive. Hệ thống tự nhận diện, xuất bảng và đồ thị HTML tương tác.

Cuối notebook chạy **Đóng gói & Tải Kết quả**: ZIP gồm checkpoint, bảng, hình và kết quả
DOCX; bỏ cache embedding. Bật `DOWNLOAD_ZIP = True` để tải về máy.

## Chỉ kiểm tra DOCX vào lần mở Colab sau

1. Chạy Bước 1–2.
2. Bước 3: đặt `TRAIN_NEW_MODEL = False`, trỏ `OUTPUT_DIR` đến run **v2** đã hoàn tất.
3. Chạy Bước 4–7: chỉ nạp định nghĩa, không đọc dataset và không train lại.
4. Chạy Bước 11, chọn DOCX. Có thể chạy Bước 8–10 để xem lại báo cáo.

Checkpoint `raw-doc-v1` phải giữ riêng; v2 cần train mới. Cache khác checkpoint; notebook
chưa hỗ trợ khôi phục optimizer/epoch sau khi Colab ngắt phiên.

## Tệp kết quả

```text
OUTPUT_DIR/
├── data_audit.json, split_manifest.json, split_summary.json
├── split_relation_support.json, run_config.json, environment.json
├── run_status.json                   # complete khi train/test hoàn tất
├── checkpoints/                      # ner.pt, relation_semantic.pt, relation_rgcn.pt
├── ner_history.json, relation_*_history.json
├── ner_test.json, test_semantic.json, test_rgcn.json
├── report_ner.csv, report_relations.csv # bảng chính, tỷ lệ theo %
├── entities_per_type.csv, relations_*.csv # bảng chi tiết, tỷ lệ 0–1
├── comparison.csv                    # toàn bộ chỉ số thô, tỷ lệ 0–1
├── REPORT.md
├── figures/                          # 6 biểu đồ PNG
└── predictions/<tên_docx>_<thời_gian>/
    ├── input_text.txt                # offset tính trên tệp này
    ├── source_info.json              # nguồn, thời gian, kiểm tra trùng thân
    ├── prediction.json               # đầy đủ kết quả
    ├── entities.csv, relations.csv
    ├── prediction.html               # văn bản tô màu, đồ thị tương tác
    └── relation_graph.png            # bản xem gọn
```

## Đọc DOCX và xem đồ thị

- Đọc thân văn bản và bảng theo thứ tự; không OCR ảnh, không lấy header/footer hoặc
  chú thích. DOCX ảnh quét cần OCR trước. Chưa hỗ trợ trực tiếp PDF hoặc `.doc`.
- Offset theo `input_text.txt`, không theo trang Word.
- HTML: lọc loại quan hệ/điểm, đổi số nút/cạnh, kéo nút, phóng to và chọn nút để xem
  đoạn nguồn. Có nút tải SVG. Nếu iframe không hiện trong Colab, tải HTML về mở bằng
  trình duyệt. HTML hoạt động offline.
- Cạnh vẽ là quan hệ **dự đoán**, không phải cạnh vị trí đầu vào RGCN. Điểm sigmoid
  chưa phải xác suất đúng đã hiệu chỉnh. Không có kết quả thì không tạo cạnh giả.
- PNG tối đa 25 nút/35 cạnh. HTML mặc định 45 nút/80 cạnh, có thể đổi. Với đồ thị rất
  lớn, HTML xét tối đa 20.000 cạnh điểm cao nhất; JSON/CSV luôn giữ đầy đủ kết quả.
- Không có đáp án được người kiểm tra thì không chấm được P/R/F1 cho DOCX. Chương trình
  báo nếu trùng thân chuẩn hóa với manifest; vẫn cần tự kiểm tra bản gần trùng.
- Thời gian suy luận gồm mã hóa + NER + RE; không gồm đọc DOCX, nạp model và vẽ hình.
- Nếu NER dự đoán quá 10.000 thực thể hoặc đồ thị vượt 500.000 cạnh vị trí, chương trình
  dừng và báo rõ. Kiểm tra chất lượng NER, ngưỡng hoặc thử tài liệu ngắn hơn; không cắt
  ngầm kết quả để tiếp tục.

## Giới hạn thực nghiệm

Dữ liệu cục bộ 174 văn bản sau lọc chỉ có 6/22 quan hệ có mẫu; phải đọc support thực tế
của bộ Drive. Macro-F1 cố định 22 lớp không thể đặt mục tiêu 60% với phân bố đó. Cần rà
soát nhãn tự động, đặc biệt các nhãn chỉ có ở phụ lục tự sinh và nhãn sai ngữ nghĩa.
NER còn giới hạn ghép ranh giới/lồng nhau cùng loại. Hai tầng RGCN không bảo đảm trao đổi
toàn bộ tài liệu. Semantic so với RGCN đánh giá cả truyền tin và gated fusion.

Gặp `Dataset/seed changed`, `Old pipeline manifest` hoặc `Output already contains...`:
dùng thư mục kết quả mới, giữ nguyên run cũ. File sai offset/ID/mã nhãn cần sửa từ nguồn;
chương trình không âm thầm biến nhãn lạ thành THUOC.
