# Rà soát tài liệu và thử nghiệm khả thi: foundation model (A1) và cảnh báo conformal (A2)

*2026-10-20. Code nằm trong `pilot/`, kết quả trong `pilot/results/`. Đây là thử nghiệm khả thi (exploratory): chưa có tiền đăng ký, nên phải xác nhận lại trong luận văn.*

## 1. Rà soát tài liệu: còn khoảng trống ở đâu?

| Hướng | Đã có người làm | Còn trống |
|---|---|---|
| Foundation model chuỗi thời gian cho dịch bệnh | Benchmark zero-shot TimesFM, Moirai, Chronos, TimeMoE trên 289 chuỗi bệnh–quốc gia (BMC Public Health 2026); đánh giá trên cúm, RSV, thủy đậu, dengue (Epidemics/ScienceDirect 2026, preprint medRxiv 2025); tay chân miệng với TimesFM/Moirai (2025); chỉ trích rằng so sánh thiếu công bằng (arXiv 2606.19560) | Dữ liệu **cấp tỉnh, nhiều nước, có biến khí hậu**; so với **baseline học máy gộp mạnh**; **cross-learning** giữa các tỉnh; **ensemble** foundation model + GBM; dùng foundation model **làm đầu vào cho hệ thống cảnh báo** |
| Khoảng dự báo conformal cho dengue | ACI ở Rio de Janeiro; EnbPI trong TREA-Net (Mexico, Malaysia); hiệu chỉnh online ở Mexico (preprint 2026, độ phủ 90% chỉ còn 0,77 ở thập phân vị cao nhất) | Kiểm định đa quốc gia; **độ phủ trong các tháng dịch** vẫn chưa ai giải được |
| **Cảnh báo dịch có bảo đảm thống kê** | Ngưỡng endemic channel và "trung bình + 2 độ lệch chuẩn"; cảnh báo xác suất ở Brazil (eLife 2016: tỷ lệ báo động giả 25%); poster ngưỡng dựa trên rủi ro ở Đồng bằng sông Cửu Long (SciDataCon 2025); lý thuyết conformal risk control và online FDR cho phát hiện bất thường (NeurIPS 2021, ICLR 2024) | **Chưa tìm thấy** công trình nào đưa conformal risk control hoặc kiểm soát FDR vào cảnh báo dịch cấp tỉnh, nhiều nước. **Đây là khoảng trống chính** |

## 2. Thử nghiệm A1: Chronos-2 zero-shot

- **Mô hình:** Chronos-2 (Amazon, giấy phép Apache 2.0, trọng số tải từ `s3://autogluon/chronos-2`). Không huấn luyện, không fine-tune.
- **Thiết kế:** chạy đúng các điểm dự báo (tỉnh × tháng gốc × horizon 1–6) của benchmark v2, trên 13 bài toán thuộc 10 nước. Mô hình chỉ nhận dữ liệu đến thời điểm gốc t.

MAE tương đối so với LightGBM-AR, trung bình nhân trên 78 ô (bài toán × horizon). Số ô có ý nghĩa thống kê theo kiểm định DM, **chưa hiệu chỉnh đa kiểm định**.

| Mô hình | MAE tương đối | Tỷ lệ thắng | Bài toán tệ nhất | Tốt hơn có YN | Tệ hơn có YN |
|---|---|---|---|---|---|
| **Ens(AR, Raw-all, Chronos2-cross)** | **0,944** | **100%** | **0,976** | **63** | 0 |
| Ens(AR, Chronos2-cross) | 0,947 | 97% | 0,981 | 45 | 0 |
| Chronos2-cross (zero-shot, chia sẻ giữa các tỉnh) | 0,962 | 82% | 1,016 | 24 | 0 |
| Chronos2 + khí hậu | 0,963 | 83% | 0,996 | 21 | 0 |
| Ens(AR, Raw-all), tốt nhất ở v2 | 0,978 | 87% | 1,003 | 38 | 0 |
| Chronos2 (từng tỉnh riêng) | 0,998 | 47% | 1,073 | 12 | 3 |

Diễn giải:

1. **Cross-learning là yếu tố quyết định.** Chạy từng tỉnh riêng thì Chronos-2 chỉ ngang AR (0,998). Cho các tỉnh chia sẻ thông tin trong cùng một batch thì cải thiện còn 0,962. Điều này khớp với phát hiện "học gộp" ở v2.
2. **Đưa foundation model vào ensemble cho kết quả tốt nhất toàn nghiên cứu:** −5,6% MAE, thắng **cả 78/78 ô**, bài toán tệ nhất vẫn tốt hơn AR. Ở Việt Nam: −7,7% đến −8,3%.
3. **Biến khí hậu giúp Chronos-2 một ít** (0,998 xuống 0,963) khi chạy từng tỉnh riêng.
4. **Lưu ý về rò rỉ:** dữ liệu tiền huấn luyện của Chronos-2 có thể đã chứa một phần các chuỗi này, hoặc chuỗi tương quan với chúng, trong các năm test. Đây là rò rỉ tiềm ẩn **bắt buộc phải bàn trong luận văn**. Cách kiểm tra: so sánh kết quả trên giai đoạn sau ngày cắt dữ liệu huấn luyện của mô hình, hoặc trên dữ liệu chắc chắn chưa công khai.

## 3. Thử nghiệm A2: conformal cho khoảng dự báo và cảnh báo

Hiệu chỉnh online, chỉ dùng phần dư của các tháng đã quan sát (không nhìn trước).

### 3.1 Khoảng dự báo 90%

| Phương pháp | Độ phủ chung | **Độ phủ ở 10% tháng có số ca cao nhất** |
|---|---|---|
| Conformal gộp, áp lên dự báo LightGBM | 0,88 | 0,83 |
| **ACI**, áp lên dự báo LightGBM | **0,89** | 0,78–0,85 |
| Khoảng gốc của Chronos-2 (q05–q95) | 0,88–0,89 | **0,78–0,79** |

Độ phủ trung bình đạt gần mục tiêu, nhưng **trong các tháng dịch thì khoảng dự báo quá hẹp**. Đây là điểm yếu chung của cả học máy lẫn foundation model.

### 3.2 Cảnh báo có kiểm soát tỷ lệ báo động sai

Thử với dengue trên 9 nước, h = 1–3. Một tháng được tính là có dịch nếu số ca vượt ngưỡng trung bình + 2 độ lệch chuẩn của vùng × tháng, tính trên các năm train, và đạt ít nhất 5 ca.

| Chính sách cảnh báo | Mục tiêu FDR | FDR thực tế (gộp) | Recall |
|---|---|---|---|
| Dự báo điểm vượt ngưỡng | – | 0,24 | 0,19 |
| Conformal risk control tĩnh | 0,30 | **0,42** | 0,27 |
| + chỉ hiệu chỉnh trên 24 tháng gần nhất | 0,30 | 0,42 | 0,29 |
| + điều chỉnh mục tiêu theo phản hồi (kiểu ACI) | 0,30 | 0,42 | 0,26–0,28 |

**Kết quả âm có giá trị:** áp dụng trực tiếp conformal risk control **không giữ được bảo đảm**, vì đợt dịch không "trao đổi được" (non-exchangeable) giữa các năm. Các cách sửa đơn giản cũng không đủ. Lý do: phản hồi thưa (cảnh báo hiếm và đến muộn) và xác suất dự báo bị lệch đúng ở những tháng dịch.

## 4. Đề xuất đề tài luận văn

> **"Hệ thống cảnh báo sớm sốt xuất huyết đa quốc gia có bảo đảm thống kê, kết hợp foundation model chuỗi thời gian và học máy gộp"**

- **Đóng góp 1 (A1):** benchmark cấp tỉnh, 10 nước, cho foundation model (Chronos-2, TimesFM, Moirai) so với GBM gộp. Ensemble foundation model + GBM là mô hình nền mạnh nhất (−5,6% MAE, thắng mọi ô).
- **Đóng góp 2 (A2, phần cốt lõi về phương pháp):** chính sách cảnh báo **giữ được FDR mục tiêu khi dữ liệu không ổn định**. Các hướng cần thử:
  - (i) conformalized quantile regression trên quantile của GBM và Chronos;
  - (ii) hiệu chỉnh Mondrian theo mức rủi ro dự báo và theo nước;
  - (iii) gộp dữ liệu hiệu chỉnh giữa các nước để có nhiều phản hồi hơn;
  - (iv) quy tắc online FDR (LORD/SAFFRON, Rebjock et al. 2021);
  - (v) weighted conformal theo mức độ giống nhau giữa các năm, ví dụ theo pha ENSO.

  Tiêu chí thành công: FDR thực tế ≤ mục tiêu trên tập xác nhận, và recall cao hơn chính sách dựa trên dự báo điểm.
- **Đóng góp 3 (Hệ thống Thông tin):** prototype dashboard cảnh báo theo tỉnh, hiển thị xác suất vượt ngưỡng, khoảng dự báo và lịch sử cảnh báo đúng/sai.

## 5. Tái lập

```bash
python pilot/conformal_alerts.py          # khoảng dự báo, cảnh báo theo từng ô
python pilot/conformal_fdr.py             # conformal risk control theo FDR
python pilot/conformal_fdr_adaptive.py    # biến thể cửa sổ / thích ứng
python pilot/foundation_chronos2.py       # cần torch + chronos-forecasting, trọng số từ s3://autogluon/chronos-2
python pilot/compare_chronos.py
```

## Tài liệu

- [Benchmarking time series foundation models for multi-country infectious disease forecasting (BMC Public Health, 2026)](https://link.springer.com/article/10.1186/s12889-026-29918-1)
- [Foundation models for time series forecasting and policy evaluation in infectious disease epidemics (2026)](https://www.sciencedirect.com/science/article/pii/S1755436526000320); [preprint medRxiv](https://www.medrxiv.org/content/10.1101/2025.02.24.25322795.full.pdf)
- [Comparative study of TSFMs for HFMD forecasting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12507834/)
- [Understanding Key Features of TSFMs from Epidemic Forecasting (arXiv 2606.19560)](https://arxiv.org/html/2606.19560)
- [Probabilistic dengue forecasting and early warning across Mexico with conformal calibration (Research Square)](https://www.researchsquare.com/article/rs-10035640/v1)
- [Assessing dengue forecasting methods, Rio de Janeiro](https://tropmedhealth.biomedcentral.com/articles/10.1186/s41182-025-00723-7)
- [TREA-Net (arXiv 2607.26854)](https://arxiv.org/pdf/2607.26854)
- [Evaluating probabilistic dengue risk forecasts from a prototype early warning system for Brazil (eLife 2016)](https://elifesciences.org/articles/11285/figures)
- [Risk-based dengue thresholds, Mekong Delta (SciDataCon 2025)](https://scidatacon.org/event/9/contributions/182/contribution.pdf)
- [Online false discovery rate control for anomaly detection in time series (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/def130d0b67eb38b7a8f4e7121ed432c-Abstract.html)
- [Conformal Risk Control (ICLR 2024)](https://research.google/pubs/conformal-risk-control/)
- [Challenges in Real-Time Prediction of Infectious Disease: Dengue in Thailand (PLOS NTD)](https://journals.plos.org/plosntds/article?id=10.1371%2Fjournal.pntd.0004761)

## 6. Bổ sung (2026-10-20): cảnh báo dựa trên quantile của LightGBM và Chronos-2

`pilot/lgbm_quantiles.py`: LightGBM quantile (7 mức, pinball loss) huấn luyện trong từng fold, đặc trưng AR + lag khí hậu. `pilot/quantile_alerts.py`, `pilot/quantile_alerts_curve.py`: xác suất vượt ngưỡng được tính bằng cách nghịch đảo hàm quantile. Có ba nguồn: LightGBM, Chronos-2 (cross-learning), và quantile trung bình của hai mô hình. Dữ liệu: dengue ở 9 nước, h = 1–3.

**Khả năng phân biệt (PR-AUC, tỷ lệ tháng có dịch 8,1%):** LightGBM 0,263 · Chronos-2 0,377 · trung bình 0,321. Chronos-2 hoặc quantile trung bình tốt nhất ở cả 9/9 nước.

**Đường kiểm soát FDR** (chính sách hiệu chỉnh riêng theo từng bài toán, nguồn = quantile trung bình):

| FDR mục tiêu | FDR thực tế | Recall | FDR nước tệ nhất |
|---|---|---|---|
| 0,2 | 0,315 | 0,197 | 0,48 |
| 0,3 | 0,379 | 0,296 | 0,52 |
| 0,4 | 0,447 | 0,389 | 0,57 |
| 0,5 | 0,514 | 0,477 | 0,62 |
| 0,6 | 0,575 | 0,549 | 0,72 |

- Mục tiêu dễ (≥ 0,5) gần như được giữ; mục tiêu khắt khe vượt +0,07 đến +0,12, và bài toán tệ nhất vượt nhiều hơn.
- Gộp hiệu chỉnh giữa các nước thì quá bảo thủ: ở mục tiêu ≤ 0,4 gần như không phát cảnh báo; ở mục tiêu 0,5 thì FDR 0,27–0,32 nhưng recall chỉ 3–9%. Nguyên nhân: xác suất giữa các nước không so sánh được (Mondrian theo nước là cần thiết).
- Hiệu chỉnh lại bằng isotonic gần như không thay đổi kết quả.
- **Baseline mạnh:** cảnh báo khi trung vị của quantile trung bình vượt ngưỡng (≥ 5 ca) cho FDR 0,216, recall 0,254. Chính sách hiệu chỉnh ở mục tiêu 0,2 **chưa vượt** baseline này (FDR 0,315, recall 0,197).
- So với cách tính xác suất từ phần dư trước đó (mục 3.2): ở mục tiêu 0,3, FDR thực tế giảm từ 0,42 xuống 0,38 và recall tăng từ 0,27 lên 0,30.

**Kết luận cho luận văn:** dùng quantile (nhất là khi kết hợp Chronos-2) giúp **phân biệt** tháng dịch tốt hơn hẳn, nhưng **bảo đảm FDR ở mục tiêu khắt khe vẫn chưa đạt**. Khoảng thiếu khoảng 0,08 ở mục tiêu 0,3, và ở cấp từng nước còn lớn hơn. Thu hẹp khoảng này, đồng thời vượt được baseline trung vị, là bài toán cốt lõi của Nội dung 4.
