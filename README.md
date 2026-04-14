# lab8_TXDL
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

# ==========================================================
# CÁC HÀM CUSTOM (Dùng cho Pipeline)
# ==========================================================

def remove_outliers_func(X):
    """Xử lý Outlier bằng phương pháp Clipping (Bài 1 & 2)"""
    X_pd = pd.DataFrame(X)
    q1 = X_pd.quantile(0.25)
    q3 = X_pd.quantile(0.75)
    iqr = q3 - q1
    # Thêm axis=1 để tránh lỗi ValueError khi align Series với DataFrame
    return X_pd.clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr, axis=1).values

def extract_date_features(X):
    """Trích xuất dữ liệu từ Time Series (Bài 1)"""
    dates = pd.to_datetime(X.iloc[:, 0])
    return np.c_[dates.dt.month, dates.dt.quarter]

# ==========================================================
# BÀI 1: XÂY DỰNG PIPELINE TỔNG QUÁT
# ==========================================================
print("--- BÀI 1: XÂY DỰNG PIPELINE ---")

# Định nghĩa các nhóm cột
num_cols = ['LotArea', 'Rooms', 'NoiseFeature']
cat_cols = ['Neighborhood', 'Condition', 'HasGarage']
text_col = 'Description'
date_col = ['SaleDate']

# Pipeline thành phần
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier', FunctionTransformer(remove_outliers_func)),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

text_transformer = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=30))
])

date_transformer = Pipeline([
    ('extractor', FunctionTransformer(extract_date_features)),
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Gom tất cả vào ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols),
    ('text', text_transformer, text_col),
    ('date', date_transformer, date_col)
])

print("Pipeline preprocessing đã khởi tạo thành công.")

# ==========================================================
# BÀI 2: KIỂM THỬ (SMOKE TEST) & PHÂN PHỐI
# ==========================================================
print("\n--- BÀI 2: KIỂM THỬ & CHẤT LƯỢNG DỮ LIỆU ---")

df = pd.read_csv('ITA105_Lab_8.csv')
X_raw = df.drop(['SalePrice', 'ImagePath'], axis=1)
y = df['SalePrice']

# Chạy thử trên 10 dòng (Smoke Test)
preprocessor.fit(X_raw)
demo_transformed = preprocessor.transform(X_raw.head(10))

# Lấy tên feature sau cùng để xuất Schema
ohe_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
tfidf_names = preprocessor.named_transformers_['text'].named_steps['tfidf'].get_feature_names_out()
all_feature_names = num_cols + list(ohe_names) + list(tfidf_names) + ['Month', 'Quarter']

print(f"Shape đầu ra: {demo_transformed.shape}")
print(f"Feature names (đầu): {all_feature_names[:10]}")

# Vẽ biểu đồ so sánh phân phối trước/sau (Bài 2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['LotArea'], kde=True, ax=ax1, color='blue').set_title('LotArea Gốc')
sns.histplot(demo_transformed[:, 0], kde=True, ax=ax2, color='red').set_title('LotArea sau Pipeline (Clipped & Scaled)')
plt.show()

# ==========================================================
# BÀI 3: TÍCH HỢP MÔ HÌNH & CROSS-VALIDATION
# ==========================================================
print("\n--- BÀI 3: TÍCH HỢP MÔ HÌNH ---")

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Chạy 5-fold CV
cv_results = cross_validate(full_pipeline, X_raw, y, cv=5, 
                            scoring=['neg_root_mean_squared_error', 'r2'])

print(f"RMSE trung bình: {-cv_results['test_neg_root_mean_squared_error'].mean():.2f}")
print(f"R2 trung bình: {cv_results['test_r2'].mean():.4f}")

# Vẽ Feature Importance
full_pipeline.fit(X_raw, y)
importances = full_pipeline.named_steps['regressor'].feature_importances_
sorted_idx = np.argsort(importances)[-15:]

plt.figure(figsize=(10, 6))
plt.barh(range(15), importances[sorted_idx])
plt.yticks(range(15), [all_feature_names[i] for i in sorted_idx])
plt.title("Top 15 Feature Importances")
plt.show()

# ==========================================================
# BÀI 4: TRIỂN KHAI SẢN PHẨM (INFERENCE)
# ==========================================================
print("\n--- BÀI 4: TRIỂN KHAI DỰ BÁO ---")

# Lưu pipeline
joblib.dump(full_pipeline, 'house_model_final.pkl')

def predict_price(new_data_path):
    """Hàm dự báo cho dữ liệu mới chưa từng thấy"""
    loaded_model = joblib.load('house_model_final.pkl')
    new_df = pd.read_csv(new_data_path)
    # Xử lý nhanh các dòng bị thiếu nếu file test không sạch
    preds = loaded_model.predict(new_df)
    return preds

# Test thực tế
print(f"Dự báo cho 3 căn nhà đầu tiên: {predict_price('ITA105_Lab_8.csv')[:3]}")

# ==========================================================
# BÀI 5: TỔNG KẾT & ĐÁNH GIÁ (THEO YÊU CẦU GIẢNG VIÊN)
# ==========================================================
print("\n--- BÀI 5: BÁO CÁO TỔNG KẾT ---")
report = """
1. Pipeline tự động hóa: Giảm lỗi thủ công bằng cách đóng gói toàn bộ quy trình transform vào 1 object.
2. Tránh Data Leakage: CV trong pipeline đảm bảo Scaler chỉ 'học' từ tập Train của mỗi fold.
3. Xử lý Unseen: OneHotEncoder với handle_unknown='ignore' giúp model không bị crash khi gặp Neighborhood mới.
4. Rủi ro: Cần theo dõi Data Drift nếu phân phối diện tích hoặc giá nhà thay đổi theo thời gian thực.
"""
print(report)
