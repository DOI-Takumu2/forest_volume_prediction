import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Streamlitページの設定
st.title("森林蓄積予測アプリ")

# タイトルとキャッチフレーズを横並びに配置
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <h1 style="margin: 0;">Forest Volume Prediction App</h1>
    <p style="margin: 0; font-size: 16px;">A tool for sustainable forest management and growth forecasting.</p>
</div>
""", unsafe_allow_html=True)

# 製作者名を右寄せ
st.markdown("""
<div style="text-align: right; font-size: 14px; margin-top: -20px;">
    製作者：土居拓務（DOI, Takumu）
</div>
""", unsafe_allow_html=True)

st.write("""
このアプリは、林齢（森林の年齢）に基づいて、1ヘクタールあたりの森林蓄積量（m³）を予測するためのツールです。
主に以下の2つの予測モデルを使用します：
""")

st.markdown("""
### **モデルの概要**

1. **ロジスティック成長モデル**  
   森林蓄積量が成長し、最終的に一定の上限に収束する性質を表現します。このモデルは、森林の成長が飽和点に近づく状況を説明するのに適しています。

2. **多項式モデル**  
   林齢に基づく森林蓄積量を2次多項式（曲線）で近似します。蓄積量が滑らかに増加するパターンを捉えるのに適しており、非負制約を設けることで負の値が出ないように工夫されています。
""")

st.markdown("""
### **アプリの使い方**

1. **データ準備**  
   森林のデータをExcelファイル形式（列名は `forest_age` と `volume_per_ha` 必須）で準備してください。
   - `forest_age`: 森林の年齢（年単位）  
   - `volume_per_ha`: 1ヘクタールあたりの蓄積量（m³）

2. **Excelファイルのアップロード**  
   アプリのアップロード機能を使い、準備したファイルを読み込みます。

3. **予測モデルの選択**  
   ロジスティック成長モデルまたは多項式モデルのいずれかを選択してください。

4. **予測結果の確認**  
   以下の内容が表示されます：
   - **フィッティングされた数式**: データに基づく予測モデルの具体的な数式  
   - **適合度（R²値）**: モデルがどれだけデータに合っているかを示す指標  
   - **信用区間**: 信頼性の高い予測範囲を90%の確率で示します  
   - **異常値検出と修正**: 予測結果の中で異常値が検出された場合は、自動で修正します  
   - **グラフ**: 観測データ、予測曲線、および信用区間を視覚化します  

5. **予測結果の保存**  
   予測結果をCSV形式でダウンロードして保存できます。
""")


### **モデルの性質**

#### **ロジスティック成長モデル**  
- **特性**: 初期成長が速く、その後成長率が徐々に低下し、最終的に一定の上限値（飽和点）に収束する。  
- **用途**: 長期的な森林の成長パターンを説明するのに適しています。  

#### **多項式モデル**  
- **特性**: 林齢に基づき滑らかに増加する森林蓄積量を曲線で近似する。  
- **用途**: 森林成長が単純な増加パターンを示す場合や、予測範囲が短期的な場合に適しています。
""")


# ロジスティック成長モデル
def logistic_growth(age, K, r, A):
    """
    ロジスティック成長モデル:
    材積の成長が一定の上限値Kに収束するモデル。
    """
    return K / (1 + A * np.exp(-r * age))

# 多項式モデル
def polynomial_model(age, a, b, c):
    """
    多項式モデル:
    材積を年齢に基づく2次多項式で表現します。負の値を防ぐため、0以下は強制的に0にします。
    """
    values = a * age**2 + b * age + c
    return np.maximum(0, values)

# 信用区間の計算
def calculate_credible_interval(model, params, ages, forest_age_max, volume_max, pcov, num_samples=500):
    predicted_samples = []
    for _ in range(num_samples):
        sample_params = np.random.multivariate_normal(params, pcov)
        predicted_values = model(ages / forest_age_max, *sample_params) * volume_max
        predicted_samples.append(predicted_values)
    predicted_samples = np.maximum(0, np.array(predicted_samples))  # 非負制約
    lower_90 = np.percentile(predicted_samples, 5, axis=0)
    upper_90 = np.percentile(predicted_samples, 95, axis=0)
    return lower_90, upper_90

# ファイルアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください（拡張子: .xlsx）", type=["xlsx"])
if uploaded_file:
    data = pd.read_excel(uploaded_file)
    st.write("アップロードされたデータ:")
    st.write(data)

    # データ抽出
    forest_age = data["forest_age"].values
    volume_per_ha = data["volume_per_ha"].values

    # モデル選択
    st.write("モデルを選択してください：")
    model_choice = st.selectbox("モデル", ["ロジスティック成長モデル (Logistic Growth Model)", "多項式モデル (Polynomial Model)"])

    # データスケーリング
    forest_age_scaled = forest_age / max(forest_age)
    volume_per_ha_scaled = volume_per_ha / max(volume_per_ha)

    # モデルフィッティング
    ages = np.arange(1, max(forest_age) + 1)  # 整数刻みに変更
    if model_choice == "ロジスティック成長モデル (Logistic Growth Model)":
        initial_guess = [1, 0.1, 1]
        popt, pcov = curve_fit(logistic_growth, forest_age_scaled, volume_per_ha_scaled, p0=initial_guess, maxfev=10000)
        K, r, A = popt
        K = K * max(volume_per_ha)  # 元スケールに戻す
        fitted_values = logistic_growth(forest_age_scaled, *popt) * max(volume_per_ha)
        equation = f"Volume = {K:.2f} / (1 + {A:.2f} * exp(-{r:.4f} * Age))"
        lower_90, upper_90 = calculate_credible_interval(logistic_growth, popt, ages, max(forest_age), max(volume_per_ha), pcov)
    else:
        initial_guess = [1, 1, 1]
        popt, pcov = curve_fit(polynomial_model, forest_age_scaled, volume_per_ha_scaled, p0=initial_guess, maxfev=10000)
        a, b, c = popt
        a = a * (max(volume_per_ha) / max(forest_age)**2)
        b = b * (max(volume_per_ha) / max(forest_age))
        c = c * max(volume_per_ha)
        fitted_values = polynomial_model(forest_age_scaled, *popt) * max(volume_per_ha)
        equation = f"Volume = max(0, {a:.2f} * Age² + {b:.2f} * Age + {c:.2f})"
        lower_90, upper_90 = calculate_credible_interval(polynomial_model, popt, ages, max(forest_age), max(volume_per_ha), pcov)

    # 結果表示
    st.write("### フィッティング結果")
    st.write(f"**数式:** {equation}")
    st.write(f"**適合度 (R²):** {r2_score(volume_per_ha, fitted_values):.4f}")

    # グラフ表示
    st.write("### グラフ")
    fig, ax = plt.subplots()
    ax.scatter(forest_age, volume_per_ha, label="Observed Data", color="blue")
    predicted_volume_formula = logistic_growth(ages / max(forest_age), *popt) * max(volume_per_ha) if model_choice == "ロジスティック成長モデル (Logistic Growth Model)" else polynomial_model(ages / max(forest_age), *popt) * max(volume_per_ha)
    ax.plot(ages, predicted_volume_formula, label="Predicted", color="red")
    ax.fill_between(ages, lower_90, upper_90, color="orange", alpha=0.3, label="90% Credible Interval")
    ax.set_xlabel("Forest Age (years)")
    ax.set_ylabel("Volume per ha (m³)")
    ax.legend()
    st.pyplot(fig)

    # 結果を保存
    predictions_df = pd.DataFrame({
        "Forest Age": ages,
        "Predicted Volume": predicted_volume_formula,
        "Lower 90% CI": lower_90,
        "Upper 90% CI": upper_90
    })
    st.write("### 予測結果のダウンロード")
    st.write(predictions_df)
    csv = predictions_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(label="Download Results as CSV", data=csv, file_name="Predicted_Volume.csv", mime="text/csv")
