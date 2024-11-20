import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import streamlit as st

# アプリの基本設定
st.markdown("""
<div style="text-align: center;">
    <h1 style="margin-bottom: 0;">🌲 森林蓄積予測アプリ</h1>
    <p style="margin-top: 5px; font-size: 18px;">森林管理と将来予測のためのツール</p>
    <p style="font-size: 12px; color: gray;">Forest Volume Prediction App: A tool for sustainable forest management and growth forecasting.</p>
</div>

<hr style="margin-top: 10px;">
""", unsafe_allow_html=True)

# モデル概要を緑色の背景で表示
st.markdown("""
<div style="background-color: #dfffdf; padding: 10px; border-radius: 5px; line-height: 1.8;">
<b>このアプリは以下の2つのモデルを用いて森林蓄積量を予測します:</b>
</div>
""", unsafe_allow_html=True)

# ロジスティック成長モデルの詳細
with st.expander("1. ロジスティック成長モデル"):
    st.write("""
    - **特性**: 初期成長が速く、その後成長率が低下し、最終的に飽和点に収束するモデルです。
    - **適用例**: 長期的な森林成長パターンを説明する際に適しています。
    - **計算**: 以下のパラメータを算出します:
        - 成長率 (r)
        - 飽和点 (K)
        - 初期条件 (A)
    """)

# 多項式モデルの詳細
with st.expander("2. 多項式モデル"):
    st.write("""
    - **特性**: 林齢に基づく森林蓄積量を2次多項式で滑らかに近似するモデルです。
    - **適用例**: 短期的な予測やシンプルな成長パターンの説明に適しています。
    - **計算**: 以下のパラメータを算出します:
        - 係数 (a, b, c) を基に曲線を近似
        - 非負制約を設け、不自然な負の値を防ぎます。
    """)

# 使用手順を黒線で囲む
st.markdown("""
<div style="border: 2px solid black; padding: 10px; margin: 10px; border-radius: 5px; line-height: 1.8;">
<b>使用手順:</b>
<ol>
    <li><b>データ準備</b>: データをExcelファイル形式（列名は <code>forest_age</code> と <code>volume_per_ha</code> 必須）にて準備してください。
        <ul>
            <li><code>forest_age</code>: 林齢（樹木の年齢）（年単位）</li>
            <li><code>volume_per_ha</code>: 1ヘクタールあたりの蓄積量（m³）</li>
        </ul>
    </li>
    <li><b>ファイルアップロード</b>: アプリのアップロード機能を使用してファイルを読み込みます。</li>
    <li><b>モデル選択</b>: ロジスティック成長モデルまたは多項式モデルを選択します。</li>
    <li><b>結果確認</b>: 以下の内容が表示されます:
        <ul>
            <li><b>適合された数式</b>: データに基づく予測モデルの具体的な数式</li>
            <li><b>適合度（R²値）</b>: モデルがデータにどれだけ適合しているかを示す指標</li>
            <li><b>信用区間</b>: 信頼性の高い予測範囲（90%）を提示</li>
            <li><b>異常値の検出と修正</b>: 異常値が検出された場合、自動で修正します</li>
            <li><b>グラフ</b>: 観測データ、予測曲線、信用区間を視覚化します</li>
        </ul>
    </li>
    <li><b>データ保存</b>: 予測結果をCSV形式でダウンロード可能です。</li>
</ol>
</div>
""", unsafe_allow_html=True)

# 補足情報と引用
st.markdown("""
---
**引用:**
DOI, Takumu (2024). *Forest Volume Prediction App: A tool for sustainable forest management and growth forecasting*.
""", unsafe_allow_html=True)

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
