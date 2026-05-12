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

## 2. Dataset

### Nguồn
- **Tên**: *Financial Statement Fraud Detection Dataset* (JAR 2020), công bố bởi
  Bao, Ke, Li, Yu, Zhang (2020) trên *Journal of Accounting Research*.
- **File**: [dataset/data_FraudDetection_JAR2020.csv](dataset/data_FraudDetection_JAR2020.csv)
- **Repo gốc**: https://github.com/JarFraud/FraudDetection
- **Nhãn fraud**: dựa trên *SEC Accounting and Auditing Enforcement Releases (AAER)*
  do Center for Financial Reporting and Management (UC Berkeley) tổng hợp.

### Mô tả
| Trường | Giải thích |
|---|---|
| `fyear` | Năm tài chính (1991–2014, paper dùng 1991–2008) |
| `gvkey` | Mã định danh công ty (Compustat) |
| `p_aaer` | Số hiệu AAER (nếu firm-year bị SEC điều tra) |
| `misstate` | **Nhãn mục tiêu** — 1 nếu firm-year được công bố trong AAER (fraud) |
| 28 biến raw | Compustat fundamentals: `at`, `sale`, `ni`, `ceq`, `ppegt`, … |
| 14 biến ratio | Theo Dechow et al. (2011): `dch_wc`, `ch_rsst`, `soft_assets`, `ch_roa`, `bm`, `dpi`, `EBIT`, … |

### Đặc tính
- Tổng ~146k firm-year quan sát (Compustat US).
- **Cực mất cân bằng**: chỉ ~0.66% mẫu là fraud (`misstate=1`) → động lực
  dùng CTGAN sinh thêm minority class.
- Split theo năm (time-based, tránh leakage):
  - **Train**: 1991–2002 (576 fraud)
  - **Val**: 2003–2005
  - **Test**: 2006–2008

### Các file đã xử lý trong [data/](data/)
- `train.csv` / `val.csv` / `test.csv` — sau impute + winsorize + scale.
- `trainLog.csv` / `valLog.csv` / `testLog.csv` — bản có log-transform.
- `syntheticFraudLogB.csv` — 1152 mẫu fraud synthesized bởi CTGAN.

## 3. Tài liệu tham khảo

1. **Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J.** (2020).
   *Detecting Accounting Fraud in Publicly Traded U.S. Firms Using a Machine
   Learning Approach.* Journal of Accounting Research, 58(1), 199–235.
   https://doi.org/10.1111/1475-679X.12292
   — Nguồn dataset và benchmark RUSBoost.

2. **Dechow, P. M., Ge, W., Larson, C. R., & Sloan, R. G.** (2011).
   *Predicting Material Accounting Misstatements.* Contemporary Accounting
   Research, 28(1), 17–82.
   https://doi.org/10.1111/j.1911-3846.2010.01041.x
   — Định nghĩa 14 biến ratio dùng làm feature engineering.

3. **Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K.** (2019).
   *Modeling Tabular Data using Conditional GAN.* NeurIPS 2019.
   https://arxiv.org/abs/1907.00503
   — Bài báo gốc của CTGAN.

4. **Chen, T., & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting
   System.* KDD 2016. https://doi.org/10.1145/2939672.2939785

5. **Platt, J.** (1999). *Probabilistic Outputs for Support Vector Machines
   and Comparisons to Regularized Likelihood Methods.* — Cơ sở cho
   probability calibration (sử dụng trong đánh giá top-k sensitivity).

6. **Park, N., Mohammadi, M., Gorde, K., Jajodia, S., Park, H., & Kim, Y.**
   (2018). *Data Synthesis based on Generative Adversarial Networks.*
   VLDB 2018. https://arxiv.org/abs/1806.03384
   — Bối cảnh GAN cho dữ liệu bảng.

7. **Zhao, Z., Kunar, A., van der Scheer, H., Birke, R., & Chen, L. Y.**
   (2021). *CTAB-GAN: Effective Table Data Synthesizing.* ACML 2021.
   https://arxiv.org/abs/2102.08369
   — Mở rộng CTGAN, dùng để so sánh.

8. **SDV (Synthetic Data Vault)** — thư viện cài đặt CTGAN sử dụng trong
   notebook. https://github.com/sdv-dev/CTGAN

9. **Lundberg, S. M., & Lee, S.-I.** (2017). *A Unified Approach to
   Interpreting Model Predictions.* NeurIPS 2017.
   https://arxiv.org/abs/1705.07874
   — SHAP cho giải thích model (nếu dùng phân tích feature importance).

10. **El Emam, K., Mosquera, L., & Hoptroff, R.** (2020). *Practical
    Synthetic Data Generation.* O'Reilly Media.
    — Tham khảo về DCR và privacy metrics cho synthetic data.
