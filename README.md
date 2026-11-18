# Nghien cuu Khoa hoc: PhoBERT + RGCN cho Trich xuat Quan he Phap ly (DocRE)

De tai: "Ung dung Graph Neural Networks (GNN) ket hop Transformer trong trich xuat quan he thuc the cap do tai lieu tren van ban quy pham phap luat Viet Nam".

## 1. Tong quan kien truc
- **Giai doan 1 (Ngay nghia & NER):** PhoBERT-base-v2 voi co che Cua so truot (Sliding Window 256/128) va mo hinh Boundary NER (80 dau ra sigmoid).
- **Giai doan 2 (Do thi & DocRE):** Xay dung do thi khach quan tu vi tri thuc the, mang 2 tang RGCN ket hop Gated Fusion va bo phan loai da nhan 22 quan he phap ly.

## 2. Cau truc thu muc
- `auto_label_pipeline.py`: Pipeline gan nhan tu dong (Distant Supervision + Regex).
- `build_document_graph.py`: Xay dung do thi tai lieu va khong gian dac trung nut.
- `labels.py`: Dinh nghia 40 nhan thuc the va 22 quan he phap ly.
- `google_colab/`: Notebook huan luyen doc lap tren Google Colab GPU.
- `docs/`: Bao cao tong ket va tai lieu dac ta.
- `results/`: Bieu do huan luyen va ket qua thuc nghiem.

## 3. Huong dan chay
Tham khao file `google_colab/README_COLAB.md` de thuc thi notebook tren Google Colab hoac chay script tren may ca nhan theo huong dan.
