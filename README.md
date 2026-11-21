# Trích xuất thực thể và quan hệ pháp luật với PhoBERT và RGCN

Hệ thống xử lý văn bản pháp luật tiếng Việt bằng **PhoBERT-base-v2 + Sliding Window +
NER start/end + RGCN + Gated Fusion**, với 40 loại thực thể và 22 quan hệ đa nhãn.

[Mở trên Google Colab](https://colab.research.google.com/github/Mingxin2811/NCKH-Legal-DocRE-PhoBERT-RGCN/blob/main/google_colab/NCKH_PhoBERT_RGCN_Training.ipynb)
· [Hướng dẫn 11 bước](google_colab/README_COLAB.md)

## Chạy trên Google Colab

1. Mở notebook, bật GPU và kết nối Drive.
2. Cài thư viện theo Bước 2.
3. Bước 3: sửa `DATASET_DIR` tới dữ liệu đã giải nén; đặt `OUTPUT_DIR` mới và giữ
   `TRAIN_NEW_MODEL = True`.
4. Chạy Bước 4–7 để kiểm tra dữ liệu, chia tập và huấn luyện.
5. Chạy Bước 8–10 để xuất bảng và sáu biểu đồ đánh giá.
6. Bước 11: tải một DOCX/TXT lên để nhận diện và xem đồ thị quan hệ tương tác.
7. Chạy cell cuối để tải ZIP gồm checkpoint, bảng, hình và kết quả dự đoán.

Notebook chạy độc lập, không cần script Python bên ngoài. Dữ liệu và trọng số lưu trên
Drive. JSON phải có `full_text`, `entities`, `relations`; CSV là bản đối chiếu, không
được tính thêm thành tài liệu huấn luyện.

Chỉ kiểm tra DOCX ở phiên sau: đặt `TRAIN_NEW_MODEL = False`, chọn thư mục kết quả v2
đã lưu, chạy Bước 1–7 để nạp định nghĩa rồi chạy Bước 11. Không huấn luyện lại.

## Cấu trúc mã nguồn

| Đường dẫn | Vai trò |
|---|---|
| `google_colab/` | Notebook độc lập, hướng dẫn và thư viện |
| `colab_support/data_pipeline.py` | Kiểm tra dữ liệu, chia tập, tạo cạnh và ứng viên |
| `colab_support/context_encoder.py` | PhoBERT, cửa sổ BPE và ánh xạ vị trí |
| `colab_support/model_gnn.py` | BoundaryNER và LegalRGCN |
| `colab_support/training.py` | Huấn luyện, chọn checkpoint, đánh giá |
| `colab_support/document_io.py` | Đọc DOCX/TXT/JSON |
| `colab_support/prediction_view.py` | Xuất CSV/JSON, tô thực thể, đồ thị HTML/PNG |
| `colab_support/reporting.py` | Bảng và biểu đồ thực nghiệm |
| `colab_support/test_thu_cong.py` | Nạp checkpoint và suy luận |
| `colab_support/rebuild_notebook.py` | Sinh notebook từ mã nguồn |
| `colab_support/tests/` | Kiểm thử chức năng và luồng notebook |
| `auto_label_pipeline.py` | Công cụ gán nhãn dữ liệu nguồn, tùy chọn |
| `labels.py` | Schema nhãn và ánh xạ |

`docs/` và `results/` có từ lần công bố trước được giữ làm tài liệu lịch sử. Kết quả đó
không đại diện cho một lần huấn luyện pipeline v2. Bộ train BIO–CRF và GNN cũ được thay
bằng mã trong `colab_support/`; checkpoint v1 cần huấn luyện lại.

## Kiến trúc và đánh giá

- PhoBERT-base-v2 đóng băng; mean pooling tạo biểu diễn ngữ nghĩa.
- Cửa sổ 254 token nội dung + BOS/EOS, bước dịch 128 token nội dung.
- NER: 40 đầu ra start và 40 đầu ra end, không dùng CRF.
- Hai tầng RGCN, bốn loại cạnh vị trí; quan hệ chuẩn không làm cạnh đầu vào.
- Gated fusion và bộ phân loại 22 sigmoid độc lập.
- Chia khoảng 80/10/10 theo nhóm thân trùng chính xác đã chuẩn hóa, lưu seed/manifest;
  cần rà soát thêm phiên bản gần trùng và các lớp hiếm.
- NER chọn checkpoint bằng validation exact-span/type F1; RE chọn bằng micro-F1
  validation với thực thể NER dự đoán. Test không tham gia chọn mô hình/ngưỡng.
- Báo cáo chính: NER P/R/F1; RE micro-P/R/F1 và Macro-F1 22 nhãn. Báo riêng thực thể
  chuẩn và thực thể dự đoán; xem thêm support, candidate recall, lỗi theo nhãn.

Nhãn tự động cần được rà soát. Lớp vắng mẫu có F1=0 trong Macro-F1 22 lớp. Điểm sigmoid
chưa phải xác suất đúng đã hiệu chỉnh. DOCX không có đáp án chỉ cho dự đoán, không tự
chấm P/R/F1. Không cam kết chỉ số đạt một mức định trước.

## Chạy trên máy

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s colab_support/tests -p "test_*.py" -v
python colab_support/rebuild_notebook.py
python colab_support/training.py --dataset /path/to/dataset --output /path/to/new_run
python colab_support/test_thu_cong.py --model-dir /path/to/new_run --input-file /path/to/new.docx --output prediction.json
```

[Hướng dẫn mã nguồn](colab_support/README.md) mô tả thêm phép thử bằng encoder thật.
Chạy một epoch trên dữ liệu nhỏ chỉ xác minh chương trình, không đo chất lượng nghiên cứu.

## Nguồn nền tảng

- [PhoBERT của VinAI](https://github.com/VinAIResearch/PhoBERT)
- [RGCNConv của PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html)

Đề tài kế thừa các mô hình và thư viện trên; không tự nhận đã tiền huấn luyện PhoBERT
hoặc phát hành checkpoint LegalPhoBERT chuyên luật.


## Kết quả thực nghiệm PhoBERT RGCN v2 seed 42

Lần chạy `phobert_rgcn_v2_run_42` dùng 376 tài liệu (263 train, 56 validation, 57 test). NER exact-span/type đạt F1 64,97%. RE RGCN đạt micro-F1 52,91% với thực thể chuẩn và 19,02% khi chạy đầu cuối; macro-F1 22 nhãn đầu cuối là 1,82% (semantic: 2,32%). Đây là một seed; test có 8/22 nhãn quan hệ có mẫu.

- [Báo cáo đã cập nhật](docs/27_BC_Chinh_Sua.docx)
- [Metrics, checkpoint và hình của lần chạy](results/phobert_rgcn_v2_run_42/)
- [Đối chiếu từng phần đã điền với file nguồn](results/phobert_rgcn_v2_run_42/DOI_CHIEU_KET_QUA_RUN42.md)

Các phần kiểm tra thủ công, kiểm tra DOCX mới, benchmark suy luận và nhật ký thu thập chưa có bằng chứng trong gói kết quả vẫn giữ trạng thái chờ trong báo cáo.
