# Đối chiếu cập nhật báo cáo từ lần chạy seed 42

Nguồn: phobert_rgcn_v2_run_42_results.zip. Mọi phần trăm được tính từ số gốc trước khi làm tròn.

| Vị trí / trường | Nội dung đã điền | File và trường nguồn |
|---|---|---|
| F1_NER_KET_LUAN | 64,97 | ner_test.json: f1 |
| F1_RE_KET_LUAN | 19,02 | test_rgcn.json: predicted_entities_end_to_end.f1_micro_positive |
| MACRO_RE_KET_LUAN | 1,82 | test_rgcn.json: predicted_entities_end_to_end.f1_macro_22 |
| CHENH_LECH_DIEM_PHAN_TRAM | 1,76 | comparison.csv: RGCN E2E trừ semantic E2E, tính trước làm tròn |
| N_TRAIN | 263 | split_summary.json: counts |
| N_VAL | 56 | split_summary.json: counts |
| N_TEST | 57 | split_summary.json: counts |
| N_ALL | 376 | split_summary.json: counts |
| TY_LE_TRAIN | 69,95 | split_summary.json: counts; tỷ lệ tính lại trên 376 tài liệu |
| TY_LE_VAL | 14,89 | split_summary.json: counts; tỷ lệ tính lại trên 376 tài liệu |
| TY_LE_TEST | 15,16 | split_summary.json: counts; tỷ lệ tính lại trên 376 tài liệu |
| E_ALL | 53482 | data_audit.json: totals.retained_entities |
| R_TRAIN | 34982 | split_relation_support.json: train (tổng số bộ ba) |
| R_VAL | 12692 | split_relation_support.json: val (tổng số bộ ba) |
| R_TEST | 2147 | split_relation_support.json: test (tổng số bộ ba) |
| R_ALL | 49821 | data_audit.json: totals.retained_relations |
| C_VAL | 24 | selected_checkpoints.json: ner.validation.per_type; đếm lớp có gold |
| C_TEST | 22 | ner_test.json: per_type; đếm lớp có gold |
| C_ALL | 29 | data_audit.json: entity_counts (số CODE có mẫu) |
| L_TRAIN | 12 | split_relation_support.json: train (số lớp có mẫu) |
| L_VAL | 8 | split_relation_support.json: val (số lớp có mẫu) |
| L_TEST | 8 | split_relation_support.json: test (số lớp có mẫu) |
| L_ALL | 12 | data_audit.json: relation_counts (số CODE có mẫu) |
| RUN_ID | phobert_rgcn_v2_run_42 | run_config.json: output |
| DATA_VERSION | raw-doc-v2 | data_audit.json: version |
| SPLIT_ID | split_manifest.json; seed 42 | split_manifest.json; split_summary.json |
| THIET_BI_THUC_TE | Tesla T4 (CUDA) | run_status.json: device/device_name |
| PHIEN_BAN_MOI_TRUONG | PyTorch 2.14.1+cu130; Transformers 5.22.1; PyG 2.8.1 | environment.json; không có phiên bản Python |
| SO_LAN_LAP | 1 seed (42) | run_config.json: seed; gói chỉ có một lần chạy |
| CHECKPOINT_NER | checkpoints/ner.pt (epoch 18; ngưỡng 0,80) | selected_checkpoints.json: ner |
| CHECKPOINT_RE | checkpoints/relation_rgcn.pt (epoch 15; ngưỡng 0,80) | selected_checkpoints.json: relation_rgcn |
| EPOCH_NER_TOT_NHAT | 18 | selected_checkpoints.json: ner.epoch |
| EPOCH_NER_DA_CHAY | 20 | ner_history.json |
| NGUONG_NER | 0,80 | selected_checkpoints.json: ner.threshold |
| LOSS_NER | 0,002204 | ner_history.json: epoch 18, loss |
| F1_NER_VAL | 72,50 | selected_checkpoints.json: ner.validation.f1 |
| P_NER_TEST | 61,00 | ner_test.json: precision |
| R_NER_TEST | 69,50 | ner_test.json: recall |
| F1_NER_TEST | 64,97 | ner_test.json: f1 |
| SUPPORT_NER_TEST | 2669 | ner_test.json: tp + fn |
| G_META | 407 | ner_test.json: per_type; cộng nhóm TEN_VAN_BAN,SO_HIEU,LOAI_VAN_BAN,NGANH,NGAY_BAN_HANH,LINH_VUC,NGAY_CO_HIEU_LUC,NGAY_HET_HIEU_LUC,TINH_TRANG_HIEU_LUC,CO_QUAN_BAN_HANH,CHUC_DANH,NGUOI_KY |
| TP_META | 298 | ner_test.json: per_type; cộng nhóm TEN_VAN_BAN,SO_HIEU,LOAI_VAN_BAN,NGANH,NGAY_BAN_HANH,LINH_VUC,NGAY_CO_HIEU_LUC,NGAY_HET_HIEU_LUC,TINH_TRANG_HIEU_LUC,CO_QUAN_BAN_HANH,CHUC_DANH,NGUOI_KY |
| FP_META | 284 | ner_test.json: per_type; cộng nhóm TEN_VAN_BAN,SO_HIEU,LOAI_VAN_BAN,NGANH,NGAY_BAN_HANH,LINH_VUC,NGAY_CO_HIEU_LUC,NGAY_HET_HIEU_LUC,TINH_TRANG_HIEU_LUC,CO_QUAN_BAN_HANH,CHUC_DANH,NGUOI_KY |
| FN_META | 109 | ner_test.json: per_type; cộng nhóm TEN_VAN_BAN,SO_HIEU,LOAI_VAN_BAN,NGANH,NGAY_BAN_HANH,LINH_VUC,NGAY_CO_HIEU_LUC,NGAY_HET_HIEU_LUC,TINH_TRANG_HIEU_LUC,CO_QUAN_BAN_HANH,CHUC_DANH,NGUOI_KY |
| F1_META | 60,26 | ner_test.json: per_type; cộng nhóm TEN_VAN_BAN,SO_HIEU,LOAI_VAN_BAN,NGANH,NGAY_BAN_HANH,LINH_VUC,NGAY_CO_HIEU_LUC,NGAY_HET_HIEU_LUC,TINH_TRANG_HIEU_LUC,CO_QUAN_BAN_HANH,CHUC_DANH,NGUOI_KY |
| G_STRUCT | 2153 | ner_test.json: per_type; cộng nhóm PHAN,CHUONG,MUC,TIEU_MUC,DIEU,KHOAN,DIEM |
| TP_STRUCT | 1527 | ner_test.json: per_type; cộng nhóm PHAN,CHUONG,MUC,TIEU_MUC,DIEU,KHOAN,DIEM |
| FP_STRUCT | 770 | ner_test.json: per_type; cộng nhóm PHAN,CHUONG,MUC,TIEU_MUC,DIEU,KHOAN,DIEM |
| FN_STRUCT | 626 | ner_test.json: per_type; cộng nhóm PHAN,CHUONG,MUC,TIEU_MUC,DIEU,KHOAN,DIEM |
| F1_STRUCT | 68,63 | ner_test.json: per_type; cộng nhóm PHAN,CHUONG,MUC,TIEU_MUC,DIEU,KHOAN,DIEM |
| G_REF | 109 | ner_test.json: per_type; cộng nhóm VB_DUOC_HUONG_DAN_AP_DUNG,VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN,VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO,VB_DUOC_DINH_CHINH,VB_DUOC_DAN_CHIEU,CAN_CU_BAN_HANH,VB_DUOC_GIAI_THICH,VB_BI_DINH_CHI_THI_HANH,VB_BI_TAM_NGUNG_GIA_HAN,VB_HUONG_DAN_AP_DUNG,VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN,VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO,VB_HOP_NHAT,VB_DINH_CHINH,VB_DAN_CHIEU,VB_AP_DUNG,VB_GIAI_THICH,VB_DINH_CHI_THI_HANH,VB_QUY_DINH_TAM_NGUNG_GIA_HAN |
| TP_REF | 30 | ner_test.json: per_type; cộng nhóm VB_DUOC_HUONG_DAN_AP_DUNG,VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN,VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO,VB_DUOC_DINH_CHINH,VB_DUOC_DAN_CHIEU,CAN_CU_BAN_HANH,VB_DUOC_GIAI_THICH,VB_BI_DINH_CHI_THI_HANH,VB_BI_TAM_NGUNG_GIA_HAN,VB_HUONG_DAN_AP_DUNG,VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN,VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO,VB_HOP_NHAT,VB_DINH_CHINH,VB_DAN_CHIEU,VB_AP_DUNG,VB_GIAI_THICH,VB_DINH_CHI_THI_HANH,VB_QUY_DINH_TAM_NGUNG_GIA_HAN |
| FP_REF | 132 | ner_test.json: per_type; cộng nhóm VB_DUOC_HUONG_DAN_AP_DUNG,VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN,VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO,VB_DUOC_DINH_CHINH,VB_DUOC_DAN_CHIEU,CAN_CU_BAN_HANH,VB_DUOC_GIAI_THICH,VB_BI_DINH_CHI_THI_HANH,VB_BI_TAM_NGUNG_GIA_HAN,VB_HUONG_DAN_AP_DUNG,VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN,VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO,VB_HOP_NHAT,VB_DINH_CHINH,VB_DAN_CHIEU,VB_AP_DUNG,VB_GIAI_THICH,VB_DINH_CHI_THI_HANH,VB_QUY_DINH_TAM_NGUNG_GIA_HAN |
| FN_REF | 79 | ner_test.json: per_type; cộng nhóm VB_DUOC_HUONG_DAN_AP_DUNG,VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN,VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO,VB_DUOC_DINH_CHINH,VB_DUOC_DAN_CHIEU,CAN_CU_BAN_HANH,VB_DUOC_GIAI_THICH,VB_BI_DINH_CHI_THI_HANH,VB_BI_TAM_NGUNG_GIA_HAN,VB_HUONG_DAN_AP_DUNG,VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN,VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO,VB_HOP_NHAT,VB_DINH_CHINH,VB_DAN_CHIEU,VB_AP_DUNG,VB_GIAI_THICH,VB_DINH_CHI_THI_HANH,VB_QUY_DINH_TAM_NGUNG_GIA_HAN |
| F1_REF | 22,14 | ner_test.json: per_type; cộng nhóm VB_DUOC_HUONG_DAN_AP_DUNG,VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN,VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO,VB_DUOC_DINH_CHINH,VB_DUOC_DAN_CHIEU,CAN_CU_BAN_HANH,VB_DUOC_GIAI_THICH,VB_BI_DINH_CHI_THI_HANH,VB_BI_TAM_NGUNG_GIA_HAN,VB_HUONG_DAN_AP_DUNG,VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH,VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN,VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO,VB_HOP_NHAT,VB_DINH_CHINH,VB_DAN_CHIEU,VB_AP_DUNG,VB_GIAI_THICH,VB_DINH_CHI_THI_HANH,VB_QUY_DINH_TAM_NGUNG_GIA_HAN |
| N_NER_BO_SOT | 814 | ner_test.json: fn (tổng lỗi exact-span, chưa tách nguyên nhân) |
| N_NER_THUA | 1186 | ner_test.json: fp (tổng lỗi exact-span, chưa tách nguyên nhân) |
| P_SEM_G | 49,75 | test_semantic.json: gold_entities.precision_micro_positive; comparison.csv |
| R_SEM_G | 50,49 | test_semantic.json: gold_entities.recall_micro_positive; comparison.csv |
| F1_SEM_G | 50,12 | test_semantic.json: gold_entities.f1_micro_positive; comparison.csv |
| MACRO_SEM_G | 12,00 | test_semantic.json: gold_entities.f1_macro_22; comparison.csv |
| P_GNN_G | 51,55 | test_rgcn.json: gold_entities.precision_micro_positive; comparison.csv |
| R_GNN_G | 54,35 | test_rgcn.json: gold_entities.recall_micro_positive; comparison.csv |
| F1_GNN_G | 52,91 | test_rgcn.json: gold_entities.f1_micro_positive; comparison.csv |
| MACRO_GNN_G | 17,10 | test_rgcn.json: gold_entities.f1_macro_22; comparison.csv |
| P_SEM_E2E | 15,42 | test_semantic.json: predicted_entities_end_to_end.precision_micro_positive; comparison.csv |
| R_SEM_E2E | 19,61 | test_semantic.json: predicted_entities_end_to_end.recall_micro_positive; comparison.csv |
| F1_SEM_E2E | 17,26 | test_semantic.json: predicted_entities_end_to_end.f1_micro_positive; comparison.csv |
| MACRO_SEM_E2E | 2,32 | test_semantic.json: predicted_entities_end_to_end.f1_macro_22; comparison.csv |
| P_GNN_E2E | 17,79 | test_rgcn.json: predicted_entities_end_to_end.precision_micro_positive; comparison.csv |
| R_GNN_E2E | 20,45 | test_rgcn.json: predicted_entities_end_to_end.recall_micro_positive; comparison.csv |
| F1_GNN_E2E | 19,02 | test_rgcn.json: predicted_entities_end_to_end.f1_micro_positive; comparison.csv |
| MACRO_GNN_E2E | 1,82 | test_rgcn.json: predicted_entities_end_to_end.f1_macro_22; comparison.csv |
| EPOCH_SEM | 30 | selected_checkpoints.json: relation_semantic; relation_semantic_history.json: epoch 30 |
| EPOCH_GNN | 15 | selected_checkpoints.json: relation_rgcn; relation_rgcn_history.json: epoch 15 |
| TAU_SEM | 0,8 | selected_checkpoints.json: relation_semantic; relation_semantic_history.json: epoch 30 |
| TAU_GNN | 0,8 | selected_checkpoints.json: relation_rgcn; relation_rgcn_history.json: epoch 15 |
| F1_VAL_SEM | 29,69 | selected_checkpoints.json: relation_semantic; relation_semantic_history.json: epoch 30 |
| F1_VAL_GNN | 30,88 | selected_checkpoints.json: relation_rgcn; relation_rgcn_history.json: epoch 15 |
| LOSS_SEM | 0,003634 | selected_checkpoints.json: relation_semantic; relation_semantic_history.json: epoch 30 |
| LOSS_GNN | 0,003668 | selected_checkpoints.json: relation_rgcn; relation_rgcn_history.json: epoch 15 |
| CAND_RECALL_G | 85,47 | test_rgcn.json: gold_entities.candidate_recall |
| MISSED_CAND_G | 312 | test_rgcn.json: gold_entities; tổng support × (1 - candidate_recall); kiểm tra khớp số nguyên |
| CAND_RECALL_E2E | 31,53 | test_rgcn.json: predicted_entities_end_to_end.candidate_recall |
| MISSED_CAND_E2E | 1470 | test_rgcn.json: predicted_entities_end_to_end; tổng support × (1 - candidate_recall); kiểm tra khớp số nguyên |
| Table index 33, row 1, col 1 | 59 | ner_test.json: per_type.TEN_VAN_BAN; entities_per_type.csv |
| Table index 33, row 1, col 2 | 36 | ner_test.json: per_type.TEN_VAN_BAN; entities_per_type.csv |
| Table index 33, row 1, col 3 | 34 | ner_test.json: per_type.TEN_VAN_BAN; entities_per_type.csv |
| Table index 33, row 1, col 4 | 23 | ner_test.json: per_type.TEN_VAN_BAN; entities_per_type.csv |
| Table index 33, row 1, col 5 | 51,43 | ner_test.json: per_type.TEN_VAN_BAN; entities_per_type.csv |
| Table index 33, row 1, col 6 | 61,02 | ner_test.json: per_type.TEN_VAN_BAN; entities_per_type.csv |
| Table index 33, row 1, col 7 | 55,81 | ner_test.json: per_type.TEN_VAN_BAN; entities_per_type.csv |
| Table index 33, row 2, col 1 | 57 | ner_test.json: per_type.SO_HIEU; entities_per_type.csv |
| Table index 33, row 2, col 2 | 53 | ner_test.json: per_type.SO_HIEU; entities_per_type.csv |
| Table index 33, row 2, col 3 | 44 | ner_test.json: per_type.SO_HIEU; entities_per_type.csv |
| Table index 33, row 2, col 4 | 4 | ner_test.json: per_type.SO_HIEU; entities_per_type.csv |
| Table index 33, row 2, col 5 | 54,64 | ner_test.json: per_type.SO_HIEU; entities_per_type.csv |
| Table index 33, row 2, col 6 | 92,98 | ner_test.json: per_type.SO_HIEU; entities_per_type.csv |
| Table index 33, row 2, col 7 | 68,83 | ner_test.json: per_type.SO_HIEU; entities_per_type.csv |
| Table index 33, row 3, col 1 | 57 | ner_test.json: per_type.LOAI_VAN_BAN; entities_per_type.csv |
| Table index 33, row 3, col 2 | 45 | ner_test.json: per_type.LOAI_VAN_BAN; entities_per_type.csv |
| Table index 33, row 3, col 3 | 46 | ner_test.json: per_type.LOAI_VAN_BAN; entities_per_type.csv |
| Table index 33, row 3, col 4 | 12 | ner_test.json: per_type.LOAI_VAN_BAN; entities_per_type.csv |
| Table index 33, row 3, col 5 | 49,45 | ner_test.json: per_type.LOAI_VAN_BAN; entities_per_type.csv |
| Table index 33, row 3, col 6 | 78,95 | ner_test.json: per_type.LOAI_VAN_BAN; entities_per_type.csv |
| Table index 33, row 3, col 7 | 60,81 | ner_test.json: per_type.LOAI_VAN_BAN; entities_per_type.csv |
| Table index 33, row 4, col 1 | 2 | ner_test.json: per_type.NGANH; entities_per_type.csv |
| Table index 33, row 4, col 2 | 0 | ner_test.json: per_type.NGANH; entities_per_type.csv |
| Table index 33, row 4, col 3 | 3 | ner_test.json: per_type.NGANH; entities_per_type.csv |
| Table index 33, row 4, col 4 | 2 | ner_test.json: per_type.NGANH; entities_per_type.csv |
| Table index 33, row 4, col 5 | 0,00 | ner_test.json: per_type.NGANH; entities_per_type.csv |
| Table index 33, row 4, col 6 | 0,00 | ner_test.json: per_type.NGANH; entities_per_type.csv |
| Table index 33, row 4, col 7 | 0,00 | ner_test.json: per_type.NGANH; entities_per_type.csv |
| Table index 33, row 5, col 1 | 54 | ner_test.json: per_type.NGAY_BAN_HANH; entities_per_type.csv |
| Table index 33, row 5, col 2 | 48 | ner_test.json: per_type.NGAY_BAN_HANH; entities_per_type.csv |
| Table index 33, row 5, col 3 | 10 | ner_test.json: per_type.NGAY_BAN_HANH; entities_per_type.csv |
| Table index 33, row 5, col 4 | 6 | ner_test.json: per_type.NGAY_BAN_HANH; entities_per_type.csv |
| Table index 33, row 5, col 5 | 82,76 | ner_test.json: per_type.NGAY_BAN_HANH; entities_per_type.csv |
| Table index 33, row 5, col 6 | 88,89 | ner_test.json: per_type.NGAY_BAN_HANH; entities_per_type.csv |
| Table index 33, row 5, col 7 | 85,71 | ner_test.json: per_type.NGAY_BAN_HANH; entities_per_type.csv |
| Table index 33, row 6, col 1 | 0 | ner_test.json: per_type.LINH_VUC; entities_per_type.csv |
| Table index 33, row 6, col 2 | 0 | ner_test.json: per_type.LINH_VUC; entities_per_type.csv |
| Table index 33, row 6, col 3 | 0 | ner_test.json: per_type.LINH_VUC; entities_per_type.csv |
| Table index 33, row 6, col 4 | 0 | ner_test.json: per_type.LINH_VUC; entities_per_type.csv |
| Table index 33, row 6, col 5 | 0,00 | ner_test.json: per_type.LINH_VUC; entities_per_type.csv |
| Table index 33, row 6, col 6 | 0,00 | ner_test.json: per_type.LINH_VUC; entities_per_type.csv |
| Table index 33, row 6, col 7 | 0,00 | ner_test.json: per_type.LINH_VUC; entities_per_type.csv |
| Table index 33, row 7, col 1 | 27 | ner_test.json: per_type.NGAY_CO_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 7, col 2 | 15 | ner_test.json: per_type.NGAY_CO_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 7, col 3 | 52 | ner_test.json: per_type.NGAY_CO_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 7, col 4 | 12 | ner_test.json: per_type.NGAY_CO_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 7, col 5 | 22,39 | ner_test.json: per_type.NGAY_CO_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 7, col 6 | 55,56 | ner_test.json: per_type.NGAY_CO_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 7, col 7 | 31,91 | ner_test.json: per_type.NGAY_CO_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 8, col 1 | 1 | ner_test.json: per_type.NGAY_HET_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 8, col 2 | 0 | ner_test.json: per_type.NGAY_HET_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 8, col 3 | 0 | ner_test.json: per_type.NGAY_HET_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 8, col 4 | 1 | ner_test.json: per_type.NGAY_HET_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 8, col 5 | 0,00 | ner_test.json: per_type.NGAY_HET_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 8, col 6 | 0,00 | ner_test.json: per_type.NGAY_HET_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 8, col 7 | 0,00 | ner_test.json: per_type.NGAY_HET_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 9, col 1 | 0 | ner_test.json: per_type.TINH_TRANG_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 9, col 2 | 0 | ner_test.json: per_type.TINH_TRANG_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 9, col 3 | 0 | ner_test.json: per_type.TINH_TRANG_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 9, col 4 | 0 | ner_test.json: per_type.TINH_TRANG_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 9, col 5 | 0,00 | ner_test.json: per_type.TINH_TRANG_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 9, col 6 | 0,00 | ner_test.json: per_type.TINH_TRANG_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 9, col 7 | 0,00 | ner_test.json: per_type.TINH_TRANG_HIEU_LUC; entities_per_type.csv |
| Table index 33, row 10, col 1 | 51 | ner_test.json: per_type.CO_QUAN_BAN_HANH; entities_per_type.csv |
| Table index 33, row 10, col 2 | 40 | ner_test.json: per_type.CO_QUAN_BAN_HANH; entities_per_type.csv |
| Table index 33, row 10, col 3 | 35 | ner_test.json: per_type.CO_QUAN_BAN_HANH; entities_per_type.csv |
| Table index 33, row 10, col 4 | 11 | ner_test.json: per_type.CO_QUAN_BAN_HANH; entities_per_type.csv |
| Table index 33, row 10, col 5 | 53,33 | ner_test.json: per_type.CO_QUAN_BAN_HANH; entities_per_type.csv |
| Table index 33, row 10, col 6 | 78,43 | ner_test.json: per_type.CO_QUAN_BAN_HANH; entities_per_type.csv |
| Table index 33, row 10, col 7 | 63,49 | ner_test.json: per_type.CO_QUAN_BAN_HANH; entities_per_type.csv |
| Table index 33, row 11, col 1 | 50 | ner_test.json: per_type.CHUC_DANH; entities_per_type.csv |
| Table index 33, row 11, col 2 | 27 | ner_test.json: per_type.CHUC_DANH; entities_per_type.csv |
| Table index 33, row 11, col 3 | 45 | ner_test.json: per_type.CHUC_DANH; entities_per_type.csv |
| Table index 33, row 11, col 4 | 23 | ner_test.json: per_type.CHUC_DANH; entities_per_type.csv |
| Table index 33, row 11, col 5 | 37,50 | ner_test.json: per_type.CHUC_DANH; entities_per_type.csv |
| Table index 33, row 11, col 6 | 54,00 | ner_test.json: per_type.CHUC_DANH; entities_per_type.csv |
| Table index 33, row 11, col 7 | 44,26 | ner_test.json: per_type.CHUC_DANH; entities_per_type.csv |
| Table index 33, row 12, col 1 | 49 | ner_test.json: per_type.NGUOI_KY; entities_per_type.csv |
| Table index 33, row 12, col 2 | 34 | ner_test.json: per_type.NGUOI_KY; entities_per_type.csv |
| Table index 33, row 12, col 3 | 15 | ner_test.json: per_type.NGUOI_KY; entities_per_type.csv |
| Table index 33, row 12, col 4 | 15 | ner_test.json: per_type.NGUOI_KY; entities_per_type.csv |
| Table index 33, row 12, col 5 | 69,39 | ner_test.json: per_type.NGUOI_KY; entities_per_type.csv |
| Table index 33, row 12, col 6 | 69,39 | ner_test.json: per_type.NGUOI_KY; entities_per_type.csv |
| Table index 33, row 12, col 7 | 69,39 | ner_test.json: per_type.NGUOI_KY; entities_per_type.csv |
| Table index 33, row 13, col 1 | 0 | ner_test.json: per_type.VB_DUOC_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 13, col 2 | 0 | ner_test.json: per_type.VB_DUOC_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 13, col 3 | 0 | ner_test.json: per_type.VB_DUOC_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 13, col 4 | 0 | ner_test.json: per_type.VB_DUOC_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 13, col 5 | 0,00 | ner_test.json: per_type.VB_DUOC_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 13, col 6 | 0,00 | ner_test.json: per_type.VB_DUOC_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 13, col 7 | 0,00 | ner_test.json: per_type.VB_DUOC_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 14, col 1 | 6 | ner_test.json: per_type.VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 14, col 2 | 1 | ner_test.json: per_type.VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 14, col 3 | 0 | ner_test.json: per_type.VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 14, col 4 | 5 | ner_test.json: per_type.VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 14, col 5 | 100,00 | ner_test.json: per_type.VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 14, col 6 | 16,67 | ner_test.json: per_type.VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 14, col 7 | 28,57 | ner_test.json: per_type.VB_DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 15, col 1 | 14 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 15, col 2 | 1 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 15, col 3 | 6 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 15, col 4 | 13 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 15, col 5 | 14,29 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 15, col 6 | 7,14 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 15, col 7 | 9,52 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 16, col 1 | 7 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 16, col 2 | 0 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 16, col 3 | 0 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 16, col 4 | 7 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 16, col 5 | 0,00 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 16, col 6 | 0,00 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 16, col 7 | 0,00 | ner_test.json: per_type.VB_BI_TAC_DONG_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 17, col 1 | 0 | ner_test.json: per_type.VB_DUOC_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 17, col 2 | 0 | ner_test.json: per_type.VB_DUOC_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 17, col 3 | 0 | ner_test.json: per_type.VB_DUOC_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 17, col 4 | 0 | ner_test.json: per_type.VB_DUOC_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 17, col 5 | 0,00 | ner_test.json: per_type.VB_DUOC_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 17, col 6 | 0,00 | ner_test.json: per_type.VB_DUOC_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 17, col 7 | 0,00 | ner_test.json: per_type.VB_DUOC_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 18, col 1 | 10 | ner_test.json: per_type.VB_DUOC_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 18, col 2 | 0 | ner_test.json: per_type.VB_DUOC_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 18, col 3 | 13 | ner_test.json: per_type.VB_DUOC_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 18, col 4 | 10 | ner_test.json: per_type.VB_DUOC_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 18, col 5 | 0,00 | ner_test.json: per_type.VB_DUOC_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 18, col 6 | 0,00 | ner_test.json: per_type.VB_DUOC_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 18, col 7 | 0,00 | ner_test.json: per_type.VB_DUOC_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 19, col 1 | 70 | ner_test.json: per_type.CAN_CU_BAN_HANH; entities_per_type.csv |
| Table index 33, row 19, col 2 | 28 | ner_test.json: per_type.CAN_CU_BAN_HANH; entities_per_type.csv |
| Table index 33, row 19, col 3 | 113 | ner_test.json: per_type.CAN_CU_BAN_HANH; entities_per_type.csv |
| Table index 33, row 19, col 4 | 42 | ner_test.json: per_type.CAN_CU_BAN_HANH; entities_per_type.csv |
| Table index 33, row 19, col 5 | 19,86 | ner_test.json: per_type.CAN_CU_BAN_HANH; entities_per_type.csv |
| Table index 33, row 19, col 6 | 40,00 | ner_test.json: per_type.CAN_CU_BAN_HANH; entities_per_type.csv |
| Table index 33, row 19, col 7 | 26,54 | ner_test.json: per_type.CAN_CU_BAN_HANH; entities_per_type.csv |
| Table index 33, row 20, col 1 | 0 | ner_test.json: per_type.VB_DUOC_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 20, col 2 | 0 | ner_test.json: per_type.VB_DUOC_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 20, col 3 | 0 | ner_test.json: per_type.VB_DUOC_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 20, col 4 | 0 | ner_test.json: per_type.VB_DUOC_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 20, col 5 | 0,00 | ner_test.json: per_type.VB_DUOC_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 20, col 6 | 0,00 | ner_test.json: per_type.VB_DUOC_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 20, col 7 | 0,00 | ner_test.json: per_type.VB_DUOC_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 21, col 1 | 0 | ner_test.json: per_type.VB_BI_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 21, col 2 | 0 | ner_test.json: per_type.VB_BI_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 21, col 3 | 0 | ner_test.json: per_type.VB_BI_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 21, col 4 | 0 | ner_test.json: per_type.VB_BI_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 21, col 5 | 0,00 | ner_test.json: per_type.VB_BI_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 21, col 6 | 0,00 | ner_test.json: per_type.VB_BI_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 21, col 7 | 0,00 | ner_test.json: per_type.VB_BI_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 22, col 1 | 0 | ner_test.json: per_type.VB_BI_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 22, col 2 | 0 | ner_test.json: per_type.VB_BI_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 22, col 3 | 0 | ner_test.json: per_type.VB_BI_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 22, col 4 | 0 | ner_test.json: per_type.VB_BI_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 22, col 5 | 0,00 | ner_test.json: per_type.VB_BI_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 22, col 6 | 0,00 | ner_test.json: per_type.VB_BI_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 22, col 7 | 0,00 | ner_test.json: per_type.VB_BI_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 23, col 1 | 0 | ner_test.json: per_type.VB_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 23, col 2 | 0 | ner_test.json: per_type.VB_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 23, col 3 | 0 | ner_test.json: per_type.VB_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 23, col 4 | 0 | ner_test.json: per_type.VB_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 23, col 5 | 0,00 | ner_test.json: per_type.VB_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 23, col 6 | 0,00 | ner_test.json: per_type.VB_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 23, col 7 | 0,00 | ner_test.json: per_type.VB_HUONG_DAN_AP_DUNG; entities_per_type.csv |
| Table index 33, row 24, col 1 | 1 | ner_test.json: per_type.VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 24, col 2 | 0 | ner_test.json: per_type.VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 24, col 3 | 0 | ner_test.json: per_type.VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 24, col 4 | 1 | ner_test.json: per_type.VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 24, col 5 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 24, col 6 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 24, col 7 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; entities_per_type.csv |
| Table index 33, row 25, col 1 | 0 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 25, col 2 | 0 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 25, col 3 | 0 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 25, col 4 | 0 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 25, col 5 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 25, col 6 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 25, col 7 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_MOT_PHAN; entities_per_type.csv |
| Table index 33, row 26, col 1 | 0 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 26, col 2 | 0 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 26, col 3 | 0 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 26, col 4 | 0 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 26, col 5 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 26, col 6 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 26, col 7 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_HET_HIEU_LUC_TOAN_BO; entities_per_type.csv |
| Table index 33, row 27, col 1 | 0 | ner_test.json: per_type.VB_HOP_NHAT; entities_per_type.csv |
| Table index 33, row 27, col 2 | 0 | ner_test.json: per_type.VB_HOP_NHAT; entities_per_type.csv |
| Table index 33, row 27, col 3 | 0 | ner_test.json: per_type.VB_HOP_NHAT; entities_per_type.csv |
| Table index 33, row 27, col 4 | 0 | ner_test.json: per_type.VB_HOP_NHAT; entities_per_type.csv |
| Table index 33, row 27, col 5 | 0,00 | ner_test.json: per_type.VB_HOP_NHAT; entities_per_type.csv |
| Table index 33, row 27, col 6 | 0,00 | ner_test.json: per_type.VB_HOP_NHAT; entities_per_type.csv |
| Table index 33, row 27, col 7 | 0,00 | ner_test.json: per_type.VB_HOP_NHAT; entities_per_type.csv |
| Table index 33, row 28, col 1 | 0 | ner_test.json: per_type.VB_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 28, col 2 | 0 | ner_test.json: per_type.VB_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 28, col 3 | 0 | ner_test.json: per_type.VB_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 28, col 4 | 0 | ner_test.json: per_type.VB_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 28, col 5 | 0,00 | ner_test.json: per_type.VB_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 28, col 6 | 0,00 | ner_test.json: per_type.VB_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 28, col 7 | 0,00 | ner_test.json: per_type.VB_DINH_CHINH; entities_per_type.csv |
| Table index 33, row 29, col 1 | 1 | ner_test.json: per_type.VB_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 29, col 2 | 0 | ner_test.json: per_type.VB_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 29, col 3 | 0 | ner_test.json: per_type.VB_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 29, col 4 | 1 | ner_test.json: per_type.VB_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 29, col 5 | 0,00 | ner_test.json: per_type.VB_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 29, col 6 | 0,00 | ner_test.json: per_type.VB_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 29, col 7 | 0,00 | ner_test.json: per_type.VB_DAN_CHIEU; entities_per_type.csv |
| Table index 33, row 30, col 1 | 0 | ner_test.json: per_type.VB_AP_DUNG; entities_per_type.csv |
| Table index 33, row 30, col 2 | 0 | ner_test.json: per_type.VB_AP_DUNG; entities_per_type.csv |
| Table index 33, row 30, col 3 | 0 | ner_test.json: per_type.VB_AP_DUNG; entities_per_type.csv |
| Table index 33, row 30, col 4 | 0 | ner_test.json: per_type.VB_AP_DUNG; entities_per_type.csv |
| Table index 33, row 30, col 5 | 0,00 | ner_test.json: per_type.VB_AP_DUNG; entities_per_type.csv |
| Table index 33, row 30, col 6 | 0,00 | ner_test.json: per_type.VB_AP_DUNG; entities_per_type.csv |
| Table index 33, row 30, col 7 | 0,00 | ner_test.json: per_type.VB_AP_DUNG; entities_per_type.csv |
| Table index 33, row 31, col 1 | 0 | ner_test.json: per_type.VB_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 31, col 2 | 0 | ner_test.json: per_type.VB_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 31, col 3 | 0 | ner_test.json: per_type.VB_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 31, col 4 | 0 | ner_test.json: per_type.VB_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 31, col 5 | 0,00 | ner_test.json: per_type.VB_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 31, col 6 | 0,00 | ner_test.json: per_type.VB_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 31, col 7 | 0,00 | ner_test.json: per_type.VB_GIAI_THICH; entities_per_type.csv |
| Table index 33, row 32, col 1 | 0 | ner_test.json: per_type.VB_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 32, col 2 | 0 | ner_test.json: per_type.VB_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 32, col 3 | 0 | ner_test.json: per_type.VB_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 32, col 4 | 0 | ner_test.json: per_type.VB_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 32, col 5 | 0,00 | ner_test.json: per_type.VB_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 32, col 6 | 0,00 | ner_test.json: per_type.VB_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 32, col 7 | 0,00 | ner_test.json: per_type.VB_DINH_CHI_THI_HANH; entities_per_type.csv |
| Table index 33, row 33, col 1 | 0 | ner_test.json: per_type.VB_QUY_DINH_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 33, col 2 | 0 | ner_test.json: per_type.VB_QUY_DINH_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 33, col 3 | 0 | ner_test.json: per_type.VB_QUY_DINH_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 33, col 4 | 0 | ner_test.json: per_type.VB_QUY_DINH_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 33, col 5 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 33, col 6 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 33, col 7 | 0,00 | ner_test.json: per_type.VB_QUY_DINH_TAM_NGUNG_GIA_HAN; entities_per_type.csv |
| Table index 33, row 34, col 1 | 0 | ner_test.json: per_type.PHAN; entities_per_type.csv |
| Table index 33, row 34, col 2 | 0 | ner_test.json: per_type.PHAN; entities_per_type.csv |
| Table index 33, row 34, col 3 | 0 | ner_test.json: per_type.PHAN; entities_per_type.csv |
| Table index 33, row 34, col 4 | 0 | ner_test.json: per_type.PHAN; entities_per_type.csv |
| Table index 33, row 34, col 5 | 0,00 | ner_test.json: per_type.PHAN; entities_per_type.csv |
| Table index 33, row 34, col 6 | 0,00 | ner_test.json: per_type.PHAN; entities_per_type.csv |
| Table index 33, row 34, col 7 | 0,00 | ner_test.json: per_type.PHAN; entities_per_type.csv |
| Table index 33, row 35, col 1 | 73 | ner_test.json: per_type.CHUONG; entities_per_type.csv |
| Table index 33, row 35, col 2 | 50 | ner_test.json: per_type.CHUONG; entities_per_type.csv |
| Table index 33, row 35, col 3 | 24 | ner_test.json: per_type.CHUONG; entities_per_type.csv |
| Table index 33, row 35, col 4 | 23 | ner_test.json: per_type.CHUONG; entities_per_type.csv |
| Table index 33, row 35, col 5 | 67,57 | ner_test.json: per_type.CHUONG; entities_per_type.csv |
| Table index 33, row 35, col 6 | 68,49 | ner_test.json: per_type.CHUONG; entities_per_type.csv |
| Table index 33, row 35, col 7 | 68,03 | ner_test.json: per_type.CHUONG; entities_per_type.csv |
| Table index 33, row 36, col 1 | 24 | ner_test.json: per_type.MUC; entities_per_type.csv |
| Table index 33, row 36, col 2 | 12 | ner_test.json: per_type.MUC; entities_per_type.csv |
| Table index 33, row 36, col 3 | 44 | ner_test.json: per_type.MUC; entities_per_type.csv |
| Table index 33, row 36, col 4 | 12 | ner_test.json: per_type.MUC; entities_per_type.csv |
| Table index 33, row 36, col 5 | 21,43 | ner_test.json: per_type.MUC; entities_per_type.csv |
| Table index 33, row 36, col 6 | 50,00 | ner_test.json: per_type.MUC; entities_per_type.csv |
| Table index 33, row 36, col 7 | 30,00 | ner_test.json: per_type.MUC; entities_per_type.csv |
| Table index 33, row 37, col 1 | 0 | ner_test.json: per_type.TIEU_MUC; entities_per_type.csv |
| Table index 33, row 37, col 2 | 0 | ner_test.json: per_type.TIEU_MUC; entities_per_type.csv |
| Table index 33, row 37, col 3 | 0 | ner_test.json: per_type.TIEU_MUC; entities_per_type.csv |
| Table index 33, row 37, col 4 | 0 | ner_test.json: per_type.TIEU_MUC; entities_per_type.csv |
| Table index 33, row 37, col 5 | 0,00 | ner_test.json: per_type.TIEU_MUC; entities_per_type.csv |
| Table index 33, row 37, col 6 | 0,00 | ner_test.json: per_type.TIEU_MUC; entities_per_type.csv |
| Table index 33, row 37, col 7 | 0,00 | ner_test.json: per_type.TIEU_MUC; entities_per_type.csv |
| Table index 33, row 38, col 1 | 611 | ner_test.json: per_type.DIEU; entities_per_type.csv |
| Table index 33, row 38, col 2 | 203 | ner_test.json: per_type.DIEU; entities_per_type.csv |
| Table index 33, row 38, col 3 | 428 | ner_test.json: per_type.DIEU; entities_per_type.csv |
| Table index 33, row 38, col 4 | 408 | ner_test.json: per_type.DIEU; entities_per_type.csv |
| Table index 33, row 38, col 5 | 32,17 | ner_test.json: per_type.DIEU; entities_per_type.csv |
| Table index 33, row 38, col 6 | 33,22 | ner_test.json: per_type.DIEU; entities_per_type.csv |
| Table index 33, row 38, col 7 | 32,69 | ner_test.json: per_type.DIEU; entities_per_type.csv |
| Table index 33, row 39, col 1 | 1034 | ner_test.json: per_type.KHOAN; entities_per_type.csv |
| Table index 33, row 39, col 2 | 903 | ner_test.json: per_type.KHOAN; entities_per_type.csv |
| Table index 33, row 39, col 3 | 200 | ner_test.json: per_type.KHOAN; entities_per_type.csv |
| Table index 33, row 39, col 4 | 131 | ner_test.json: per_type.KHOAN; entities_per_type.csv |
| Table index 33, row 39, col 5 | 81,87 | ner_test.json: per_type.KHOAN; entities_per_type.csv |
| Table index 33, row 39, col 6 | 87,33 | ner_test.json: per_type.KHOAN; entities_per_type.csv |
| Table index 33, row 39, col 7 | 84,51 | ner_test.json: per_type.KHOAN; entities_per_type.csv |
| Table index 33, row 40, col 1 | 411 | ner_test.json: per_type.DIEM; entities_per_type.csv |
| Table index 33, row 40, col 2 | 359 | ner_test.json: per_type.DIEM; entities_per_type.csv |
| Table index 33, row 40, col 3 | 74 | ner_test.json: per_type.DIEM; entities_per_type.csv |
| Table index 33, row 40, col 4 | 52 | ner_test.json: per_type.DIEM; entities_per_type.csv |
| Table index 33, row 40, col 5 | 82,91 | ner_test.json: per_type.DIEM; entities_per_type.csv |
| Table index 33, row 40, col 6 | 87,35 | ner_test.json: per_type.DIEM; entities_per_type.csv |
| Table index 33, row 40, col 7 | 85,07 | ner_test.json: per_type.DIEM; entities_per_type.csv |
| Table index 34, row 1, col 1 | 2043 | test_rgcn.json: per_class.THUOC; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 1, col 2 | 51,49 | test_rgcn.json: per_class.THUOC; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 1, col 3 | 53,30 | test_rgcn.json: per_class.THUOC; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 1, col 4 | 52,38 | test_rgcn.json: per_class.THUOC; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 1, col 5 | 18,65 | test_rgcn.json: per_class.THUOC; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 1, col 6 | 20,66 | test_rgcn.json: per_class.THUOC; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 1, col 7 | 19,60 | test_rgcn.json: per_class.THUOC; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 2, col 1 | 70 | test_rgcn.json: per_class.CAN_CU_BAN_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 2, col 2 | 62,92 | test_rgcn.json: per_class.CAN_CU_BAN_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 2, col 3 | 80,00 | test_rgcn.json: per_class.CAN_CU_BAN_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 2, col 4 | 70,44 | test_rgcn.json: per_class.CAN_CU_BAN_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 2, col 5 | 12,50 | test_rgcn.json: per_class.CAN_CU_BAN_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 2, col 6 | 22,86 | test_rgcn.json: per_class.CAN_CU_BAN_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 2, col 7 | 16,16 | test_rgcn.json: per_class.CAN_CU_BAN_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 3, col 1 | 0 | test_rgcn.json: per_class.HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 3, col 2 | 0,00 | test_rgcn.json: per_class.HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 3, col 3 | 0,00 | test_rgcn.json: per_class.HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 3, col 4 | 0,00 | test_rgcn.json: per_class.HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 3, col 5 | 0,00 | test_rgcn.json: per_class.HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 3, col 6 | 0,00 | test_rgcn.json: per_class.HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 3, col 7 | 0,00 | test_rgcn.json: per_class.HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 4, col 1 | 0 | test_rgcn.json: per_class.DUOC_HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 4, col 2 | 0,00 | test_rgcn.json: per_class.DUOC_HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 4, col 3 | 0,00 | test_rgcn.json: per_class.DUOC_HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 4, col 4 | 0,00 | test_rgcn.json: per_class.DUOC_HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 4, col 5 | 0,00 | test_rgcn.json: per_class.DUOC_HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 4, col 6 | 0,00 | test_rgcn.json: per_class.DUOC_HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 4, col 7 | 0,00 | test_rgcn.json: per_class.DUOC_HUONG_DAN_AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 5, col 1 | 1 | test_rgcn.json: per_class.QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 5, col 2 | 33,33 | test_rgcn.json: per_class.QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 5, col 3 | 100,00 | test_rgcn.json: per_class.QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 5, col 4 | 50,00 | test_rgcn.json: per_class.QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 5, col 5 | 0,00 | test_rgcn.json: per_class.QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 5, col 6 | 0,00 | test_rgcn.json: per_class.QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 5, col 7 | 0,00 | test_rgcn.json: per_class.QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 6, col 1 | 5 | test_rgcn.json: per_class.DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 6, col 2 | 23,53 | test_rgcn.json: per_class.DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 6, col 3 | 80,00 | test_rgcn.json: per_class.DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 6, col 4 | 36,36 | test_rgcn.json: per_class.DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 6, col 5 | 0,00 | test_rgcn.json: per_class.DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 6, col 6 | 0,00 | test_rgcn.json: per_class.DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 6, col 7 | 0,00 | test_rgcn.json: per_class.DUOC_QUY_DINH_CHI_TIET_HUONG_DAN_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 7, col 1 | 0 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 7, col 2 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 7, col 3 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 7, col 4 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 7, col 5 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 7, col 6 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 7, col 7 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 8, col 1 | 14 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 8, col 2 | 66,67 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 8, col 3 | 71,43 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 8, col 4 | 68,97 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 8, col 5 | 3,03 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 8, col 6 | 7,14 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 8, col 7 | 4,26 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_MOT_PHAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 9, col 1 | 0 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 9, col 2 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 9, col 3 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 9, col 4 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 9, col 5 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 9, col 6 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 9, col 7 | 0,00 | test_rgcn.json: per_class.QUY_DINH_HET_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 10, col 1 | 7 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 10, col 2 | 71,43 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 10, col 3 | 71,43 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 10, col 4 | 71,43 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 10, col 5 | 0,00 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 10, col 6 | 0,00 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 10, col 7 | 0,00 | test_rgcn.json: per_class.BI_TAC_DONG_HIEU_LUC_TOAN_BO; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 11, col 1 | 0 | test_rgcn.json: per_class.DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 11, col 2 | 0,00 | test_rgcn.json: per_class.DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 11, col 3 | 0,00 | test_rgcn.json: per_class.DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 11, col 4 | 0,00 | test_rgcn.json: per_class.DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 11, col 5 | 0,00 | test_rgcn.json: per_class.DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 11, col 6 | 0,00 | test_rgcn.json: per_class.DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 11, col 7 | 0,00 | test_rgcn.json: per_class.DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 12, col 1 | 0 | test_rgcn.json: per_class.DUOC_DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 12, col 2 | 0,00 | test_rgcn.json: per_class.DUOC_DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 12, col 3 | 0,00 | test_rgcn.json: per_class.DUOC_DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 12, col 4 | 0,00 | test_rgcn.json: per_class.DUOC_DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 12, col 5 | 0,00 | test_rgcn.json: per_class.DUOC_DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 12, col 6 | 0,00 | test_rgcn.json: per_class.DUOC_DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 12, col 7 | 0,00 | test_rgcn.json: per_class.DUOC_DINH_CHINH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 13, col 1 | 2 | test_rgcn.json: per_class.DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 13, col 2 | 0,00 | test_rgcn.json: per_class.DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 13, col 3 | 0,00 | test_rgcn.json: per_class.DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 13, col 4 | 0,00 | test_rgcn.json: per_class.DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 13, col 5 | 0,00 | test_rgcn.json: per_class.DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 13, col 6 | 0,00 | test_rgcn.json: per_class.DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 13, col 7 | 0,00 | test_rgcn.json: per_class.DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 14, col 1 | 5 | test_rgcn.json: per_class.DUOC_DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 14, col 2 | 20,00 | test_rgcn.json: per_class.DUOC_DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 14, col 3 | 40,00 | test_rgcn.json: per_class.DUOC_DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 14, col 4 | 26,67 | test_rgcn.json: per_class.DUOC_DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 14, col 5 | 0,00 | test_rgcn.json: per_class.DUOC_DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 14, col 6 | 0,00 | test_rgcn.json: per_class.DUOC_DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 14, col 7 | 0,00 | test_rgcn.json: per_class.DUOC_DAN_CHIEU; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 15, col 1 | 0 | test_rgcn.json: per_class.AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 15, col 2 | 0,00 | test_rgcn.json: per_class.AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 15, col 3 | 0,00 | test_rgcn.json: per_class.AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 15, col 4 | 0,00 | test_rgcn.json: per_class.AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 15, col 5 | 0,00 | test_rgcn.json: per_class.AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 15, col 6 | 0,00 | test_rgcn.json: per_class.AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 15, col 7 | 0,00 | test_rgcn.json: per_class.AP_DUNG; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 16, col 1 | 0 | test_rgcn.json: per_class.GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 16, col 2 | 0,00 | test_rgcn.json: per_class.GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 16, col 3 | 0,00 | test_rgcn.json: per_class.GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 16, col 4 | 0,00 | test_rgcn.json: per_class.GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 16, col 5 | 0,00 | test_rgcn.json: per_class.GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 16, col 6 | 0,00 | test_rgcn.json: per_class.GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 16, col 7 | 0,00 | test_rgcn.json: per_class.GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 17, col 1 | 0 | test_rgcn.json: per_class.DUOC_GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 17, col 2 | 0,00 | test_rgcn.json: per_class.DUOC_GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 17, col 3 | 0,00 | test_rgcn.json: per_class.DUOC_GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 17, col 4 | 0,00 | test_rgcn.json: per_class.DUOC_GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 17, col 5 | 0,00 | test_rgcn.json: per_class.DUOC_GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 17, col 6 | 0,00 | test_rgcn.json: per_class.DUOC_GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 17, col 7 | 0,00 | test_rgcn.json: per_class.DUOC_GIAI_THICH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 18, col 1 | 0 | test_rgcn.json: per_class.HOP_NHAT; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 18, col 2 | 0,00 | test_rgcn.json: per_class.HOP_NHAT; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 18, col 3 | 0,00 | test_rgcn.json: per_class.HOP_NHAT; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 18, col 4 | 0,00 | test_rgcn.json: per_class.HOP_NHAT; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 18, col 5 | 0,00 | test_rgcn.json: per_class.HOP_NHAT; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 18, col 6 | 0,00 | test_rgcn.json: per_class.HOP_NHAT; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 18, col 7 | 0,00 | test_rgcn.json: per_class.HOP_NHAT; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 19, col 1 | 0 | test_rgcn.json: per_class.DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 19, col 2 | 0,00 | test_rgcn.json: per_class.DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 19, col 3 | 0,00 | test_rgcn.json: per_class.DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 19, col 4 | 0,00 | test_rgcn.json: per_class.DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 19, col 5 | 0,00 | test_rgcn.json: per_class.DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 19, col 6 | 0,00 | test_rgcn.json: per_class.DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 19, col 7 | 0,00 | test_rgcn.json: per_class.DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 20, col 1 | 0 | test_rgcn.json: per_class.BI_DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 20, col 2 | 0,00 | test_rgcn.json: per_class.BI_DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 20, col 3 | 0,00 | test_rgcn.json: per_class.BI_DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 20, col 4 | 0,00 | test_rgcn.json: per_class.BI_DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 20, col 5 | 0,00 | test_rgcn.json: per_class.BI_DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 20, col 6 | 0,00 | test_rgcn.json: per_class.BI_DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 20, col 7 | 0,00 | test_rgcn.json: per_class.BI_DINH_CHI_THI_HANH; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 21, col 1 | 0 | test_rgcn.json: per_class.QUY_DINH_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 21, col 2 | 0,00 | test_rgcn.json: per_class.QUY_DINH_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 21, col 3 | 0,00 | test_rgcn.json: per_class.QUY_DINH_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 21, col 4 | 0,00 | test_rgcn.json: per_class.QUY_DINH_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 21, col 5 | 0,00 | test_rgcn.json: per_class.QUY_DINH_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 21, col 6 | 0,00 | test_rgcn.json: per_class.QUY_DINH_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 21, col 7 | 0,00 | test_rgcn.json: per_class.QUY_DINH_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 22, col 1 | 0 | test_rgcn.json: per_class.BI_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 22, col 2 | 0,00 | test_rgcn.json: per_class.BI_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 22, col 3 | 0,00 | test_rgcn.json: per_class.BI_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 22, col 4 | 0,00 | test_rgcn.json: per_class.BI_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 22, col 5 | 0,00 | test_rgcn.json: per_class.BI_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 22, col 6 | 0,00 | test_rgcn.json: per_class.BI_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Table index 34, row 22, col 7 | 0,00 | test_rgcn.json: per_class.BI_TAM_NGUNG_GIA_HAN; relations_rgcn_{gold_entities,predicted_entities_end_to_end}.csv |
| Paragraph 261 | Các kết quả thực nghiệm được báo cáo theo lần chạy phobert_rgcn_v2_run_42, với seed 42. Checkpoint và ngưỡng được chọn trên validation; các bảng test phân biệt thực thể chuẩn và thực thể NER dự đoán. | run_config.json; selected_checkpoints.json |
| Paragraph 440 | Audit ghi nhận 376 tài liệu, trong đó 375 có phần tổng hợp cuối văn bản. Sau khi loại 10.652 thực thể ở phần tổng hợp và 8.980 quan hệ có đầu mút ngoài phần thân, dữ liệu còn 53.482 thực thể và 49.821 quan hệ; 978 ranh giới thực thể được cắt khoảng trắng. | data_audit.json: totals |
| Paragraph 591 | Thực nghiệm hoàn tất trên Tesla T4 với seed 42. Các bảng và hình lấy từ cùng gói phobert_rgcn_v2_run_42_results.zip; số liệu phần trăm được làm tròn đến hai chữ số thập phân. | run_status.json; run_config.json |
| Paragraph 622 | NER chạy 20 epoch; checkpoint tốt nhất ở epoch 18, ngưỡng 0,80, F1 validation 72,50%. Loss train tại checkpoint là 0,002204. Trên test, mô hình đạt Precision 61,00%, Recall 69,50%, F1 64,97%, với TP = 1.855, FP = 1.186 và FN = 814. | ner_history.json; selected_checkpoints.json; ner_test.json |
| Paragraph 630 | Đánh giá exact-span/type ghi nhận 814 thực thể tham chiếu không khớp dự đoán và 1.186 dự đoán không khớp tham chiếu. Các số này là FN và FP tổng hợp: sai loại hoặc sai ranh giới cũng có thể tạo đồng thời một FP và một FN. Chưa thể phân rã số đếm thành các nguyên nhân loại trừ nhau từ số liệu tổng hợp. | ner_test.json: tp/fp/fn; không có dự đoán chi tiết trong ZIP |
| Table index 24, row 1, col 0 | Thực thể tham chiếu không khớp (FN) | ner_test.json: fn; không tách riêng nguyên nhân |
| Table index 24, row 2, col 0 | Dự đoán không khớp (FP) | ner_test.json: fp; không tách riêng nguyên nhân |
| Paragraph 637 | Ở chế độ thực thể chuẩn, RGCN đạt micro-F1 52,91%, cao hơn semantic 50,12% khoảng 2,80 điểm phần trăm. Ở chế độ toàn quy trình, micro-F1 tăng từ 17,26% lên 19,02% (+1,76 điểm phần trăm), trong khi macro-F1 trên 22 nhãn giảm từ 2,32% xuống 1,82%. Đây là kết quả của một seed, chưa phản ánh độ biến thiên giữa các lần chạy. | comparison.csv; chênh lệch tính từ số chưa làm tròn |
| Paragraph 645 | Semantic chọn epoch 30/30, RGCN chọn epoch 15/20; cả hai dùng ngưỡng 0,80. Micro-F1 validation toàn quy trình tương ứng là 29,69% và 30,88%. RGCN dừng sau 20 epoch, còn checkpoint được đánh giá là epoch 15; ngưỡng chọn trên validation được giữ nguyên cho test. | selected_checkpoints.json; relation_semantic_history.json; relation_rgcn_history.json; run_config.json |
| Paragraph 649 | Tập test có 2147 bộ ba tham chiếu và 8/22 nhãn có mẫu. Candidate recall đạt 85,47% với thực thể chuẩn, giảm còn 31,53% với thực thể NER dự đoán. Tổng số cặp ứng viên tương ứng là 85.746 và 98.784; số bộ ba ngoài ứng viên là 312 và 1470. | test_rgcn.json; split_relation_support.json |
| Paragraph 650 | THUOC có 2.043/2147 bộ ba test (95,16%), chi phối micro-F1. Với thực thể chuẩn, RGCN đạt F1 THUOC 52,38% và CAN_CU_BAN_HANH 70,44% (70 mẫu). Ở chế độ đầu cuối, chất lượng giảm và macro-F1 chỉ đạt 1,82%; do đó chưa thể kết luận cải thiện đồng đều trên các nhãn. | test_rgcn.json: per_class; split_relation_support.json: test |
| Paragraph 675 | Nhật ký run_status.json ghi tổng thời gian lần chạy là 2.339,57 giây (khoảng 38,99 phút), trạng thái complete, thiết bị CUDA Tesla T4. Đây là thời gian tổng được pipeline ghi nhận, không phải thời gian suy luận cho một DOCX và không thay thế phép đo từng giai đoạn trong Bảng 6.15. | run_status.json: elapsed_seconds, status, device, device_name |
| Paragraph 679 | Kết quả cho thấy cấu hình RGCN cải thiện micro-F1 ở cả hai chế độ so với semantic, nhưng macro-F1 toàn quy trình giảm. Khoảng cách giữa thực thể chuẩn và toàn quy trình, cùng candidate recall 31,53% ở chế độ đầu cuối, cho thấy nhận diện đầu mút và độ bao phủ ứng viên cần được ưu tiên cải thiện. Kết quả hiện tại được đo trên một seed và nhãn tham chiếu hiện có. | comparison.csv; test_rgcn.json; run_config.json |
| Paragraph 683 | NER đạt F1 64,97%. RE toàn quy trình của PhoBERT–RGCN đạt micro-F1 19,02% và macro-F1 trên 22 nhãn 1,82%; micro-F1 tăng 1,76 điểm phần trăm so với semantic, nhưng macro-F1 giảm 0,50 điểm. Với thực thể chuẩn, RGCN đạt micro-F1 52,91% và macro-F1 17,10%. Các kết quả phản ánh một lần chạy seed 42, không đủ để khẳng định độ ổn định hoặc cải thiện trên mọi nhãn. | ner_test.json; comparison.csv |
| Paragraph 719 | Các bảng chi tiết được trích từ lần chạy phobert_rgcn_v2_run_42. TP/FP/FN là số đếm nguyên; P/R/F1 trình bày theo %. Lớp không có gold vẫn được giữ và điểm được tính theo quy ước 0. | ner_test.json; test_rgcn.json |
| Hình tại paragraph 620 | ner_training.png | figures/ner_training.png |
| Hình tại paragraph 638 | re_test_comparison.png | figures/re_test_comparison.png |
| Hình tại paragraph 643 | re_training.png | figures/re_training.png |
| Phụ lục A.2 | Liên kết GitHub, notebook, mã nguồn, kết quả và hồ sơ tái lập | git remote origin; run_config.json; selected_checkpoints.json; split_relation_support.json |

## Các phần chưa có đủ bằng chứng để điền

Các trường sau vẫn giữ màu đỏ; không suy ra số liệu kiểm tra thủ công, benchmark, ngày thu thập hoặc kết quả DOCX mới từ số tổng hợp.

- [CHEN_HINH_PHAN_BO_NHAN_VA_DO_DAI_TU_LAN_CHAY_CHINH_THUC]
- [CHEN_ANH_TO_SANG_THUC_THE_TREN_DOCX_MOI_KEM_DOAN_NGUON]
- [CHEN_DO_THI_QUAN_HE_DOCX_MOI_CO_NHAN_CANH_VA_GHI_RO_GIOI_HAN_HIEN_THI]
- [NGAY_CHOT_DU_LIEU]
- [N_NGUON]
- [N_CO_TOAN_VAN]
- [N_LOI_THU_THAP]
- [N_TAI_LIEU_HOP_LE]
- [N_VB_KIEM_TRA_NHAN]
- [THONG_TIN_KIEM_TRA_NHAN]
- [N_LOI_NER_DA_SUA]
- [N_LOI_RE_DA_SUA]
- [PHAM_VI_NHAN_TU_DONG]
- [E_TRAIN]
- [E_VAL]
- [E_TEST]
- [C_TRAIN]
- [NGAY_CHAY]
- [VI_DU_NER_BO_SOT]
- [VI_DU_NER_THUA]
- [N_NER_SAI_LOAI]
- [VI_DU_NER_SAI_LOAI]
- [N_NER_SAI_BIEN]
- [VI_DU_NER_SAI_BIEN]
- [MED_PAIRS_G]
- [MED_PAIRS_E2E]
- [N_RE_LOI_NER]
- [N_RE_LOI_UNG_VIEN]
- [N_RE_LOI_NHAN]
- [N_RE_LOI_HUONG]
- [N_RE_CAN_XAC_MINH]
- [TEN_DOCX_MOI]
- [PHAM_VI_KIEM_TRA]
- [NGUON_DOCX_MOI]
- [NGUOI_KIEM_TRA]
- [CHECKPOINT_TEST_DOCX]
- [NGUONG_TEST_DOCX]
- [SPAN_DUNG_VA_CODE]
- [DOAN_NGUON_SPAN_DUNG]
- [SPAN_SAI_VA_THAM_CHIEU]
- [NGUYEN_NHAN_CAN_KIEM_TRA]
- [BO_BA_QUAN_HE_DUNG]
- [BANG_CHUNG_QUAN_HE_DUNG]
- [BO_BA_SAI_VA_THAM_CHIEU]
- [LOI_NER_UNG_VIEN_HAY_CLASSIFIER]
- [T_DOCX]
- [DO_DAI_VAN_BAN]
- [T_ENCODER]
- [T_CACHE]
- [SO_CUA_SO_THUC_TE]
- [T_NER]
- [SO_DON_VI_TU_VUNG]
- [T_RE]
- [SO_NUT_VA_CAP]
- [T_EXPORT]
- [PHAM_VI_HIEN_THI]
- [VRAM_MAX]
- [THIET_BI_VA_CACH_DO]
- [TAI_LIEU_PHAM_VI]
- [CAU_HINH_SUY_LUAN]
- [N_SPAN_KIEM_TRA]
- [N_QUAN_HE_KIEM_TRA]
- [NGUOI_VA_NGAY_KIEM_TRA]
- [DOAN_NGUON]
- [DU_DOAN]
- [THAM_CHIEU]
- [NHAN_XET]