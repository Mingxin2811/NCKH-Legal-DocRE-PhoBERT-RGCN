# Kết quả thực đo

NER: đúng đồng thời ranh giới và loại. RE: quan hệ dương có hướng; kết quả chính dùng thực thể NER dự đoán.

Macro-F1 lấy đủ 22 lớp, lớp không có mẫu có F1=0. Đọc cùng support; micro-F1 có thể bị THUOC chi phối.

Loss là loss huấn luyện có trọng số, không phải phần trăm chính xác. Các biểu đồ không chứng minh khả năng khái quát nếu nhãn test chưa được người kiểm tra.

| task                |   precision_pct |   recall_pct |   f1_pct |   tp |   fp |   fn |
|:--------------------|----------------:|-------------:|---------:|-----:|-----:|-----:|
| NER exact-span/type |              61 |         69.5 |    64.97 | 1855 | 1186 |  814 |

| model    | setting                       |   precision_micro_pct |   recall_micro_pct |   f1_micro_pct |   f1_macro_22_pct |   candidate_recall_pct |   supported_relations |
|:---------|:------------------------------|----------------------:|-------------------:|---------------:|------------------:|-----------------------:|----------------------:|
| semantic | gold_entities                 |                 49.75 |              50.49 |          50.12 |             12    |                  85.47 |                     8 |
| semantic | predicted_entities_end_to_end |                 15.42 |              19.61 |          17.26 |              2.32 |                  31.53 |                     8 |
| rgcn     | gold_entities                 |                 51.55 |              54.35 |          52.91 |             17.1  |                  85.47 |                     8 |
| rgcn     | predicted_entities_end_to_end |                 17.79 |              20.45 |          19.02 |              1.82 |                  31.53 |                     8 |