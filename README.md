# Fraud Detection bằng CTGAN


## 1. Tổng quan phương pháp

### Bài toán
Nhị phân: dự đoán cờ `misstate` (1 = fraud) cho mỗi firm-year từ 28 biến tài
chính raw + 14 biến ratio (theo Dechow et al. 2011). Tập gốc cực mất cân bằng
(~0.66% fraud) nên cần **xử lý imbalance**.

### Pipeline tiền xử lý
1. **Drop** hàng có >50% feature NaN.
2. **Split theo năm**: train 1991–2002 / val 2003–2005 / test 2006–2008.
3. **Impute** bằng median theo từng năm (fit trên train).
4. **Winsorize** 1%–99% (fit trên train) — chống outlier cực.
5. **Log selective**: `log1p` cho biến luôn ≥0; `signed_log` cho biến có thể
   âm (NI, EBIT, ratios) để giữ dấu.
6. **RobustScaler** — ổn định với outlier hơn StandardScaler.

### CTGAN config
- Generator/Discriminator: (128, 128), embedding 64.
- 800 epochs, batch 300, lr=1e-4, discriminator_steps=3.
- Train trên 576 mẫu fraud của tập train, sinh 1152 mẫu synth (tỉ lệ 1:2).

### Classifier
- **LR**: LogisticRegression, max_iter=2000.
- **XGB**: XGBClassifier, n_est=400, depth=6, lr=0.05, **scale_pos_weight=50**.
- 3 seed (42, 7, 123). Phase 1 chọn variant trên val → Phase 2 đánh test 1 lần.

### Đánh giá
- **Fidelity**: KS test, |corr_real − corr_synth|, PCA visual.
- **Classifier**: ROC-AUC, PR-AUC, NDCG@k, sens@k% (k ∈ {1,2,5,10}).
- **Privacy**: DCR (Distance to Closest Record) — synth ≥ holdout là an toàn.
