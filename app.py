# streamlit run app.py
"""
Web tool for RF-based APO risk prediction in pregnant women with pre-existing CKD.
Provides bilingual inputs, patient-specific SHAP explanations, and optional local LLM interpretation.
"""

import streamlit as st
import streamlit.components.v1 as components
import pickle
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os
import base64
import html
import io
import json
import hashlib
from pathlib import Path
from datetime import datetime

# ==========================================
# NumPy 2.0 Compatibility Patch
# ==========================================
# NumPy 2.0 renamed trapz to trapezoid, adding alias for backward compatibility
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# ==========================================
# Streamlit display settings
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Arial',
    'PingFang SC',
    'Hiragino Sans GB',
    'Heiti TC',
    'Songti SC',
    'SimHei',
    'Noto Sans CJK SC',
    'DejaVu Sans',
]
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 360
plt.rcParams['savefig.dpi'] = 650
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['svg.fonttype'] = 'none'


# --- 0. Helper function for displaying interactive SHAP plots ---
def st_shap(plot, height=None):
    """
    Embed SHAP JS interactive plots in Streamlit
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 150, scrolling=True)


def st_matplotlib_high_res(fig):
    """
    Render a Matplotlib figure as a high-resolution image for stable Streamlit display.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=650, bbox_inches=None, facecolor="white")
    buffer.seek(0)
    st.image(buffer, width="stretch")


# --- Smart SHAP Explainer Selection ---
def get_explainer(model, X_background):
    """
    Automatically select appropriate SHAP Explainer based on model type
    
    Args:
        model: Trained model
        X_background: Background data (for KernelExplainer)
    
    Returns:
        explainer: SHAP explainer object
        explainer_type: Explainer type (for subsequent processing)
    """
    model_type = type(model).__name__
    
    # Tree models: Use TreeExplainer
    tree_models = ['LGBMClassifier', 'XGBClassifier', 'RandomForestClassifier', 
                   'GradientBoostingClassifier', 'DecisionTreeClassifier',
                   'ExtraTreesClassifier', 'CatBoostClassifier']
    
    # Linear models: Use LinearExplainer
    linear_models = ['LogisticRegression', 'Ridge', 'Lasso', 'ElasticNet',
                     'SGDClassifier', 'LinearRegression']
    
    if model_type in tree_models:
        explainer = shap.TreeExplainer(model)
        return explainer, 'tree'
    
    elif model_type in linear_models:
        # LinearExplainer requires (model, background_data)
        explainer = shap.LinearExplainer(model, X_background)
        return explainer, 'linear'
    
    else:
        # Other models: Use generic KernelExplainer
        # Note: KernelExplainer is slower, use fewer background samples
        X_summary = shap.sample(X_background, min(100, len(X_background)))
        
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1],
                X_summary
            )
        else:
            explainer = shap.KernelExplainer(model.predict, X_summary)
        
        return explainer, 'kernel'


def compute_shap_values(explainer, explainer_type, X_data):
    """
    Calculate SHAP values and handle different explainer return formats
    
    Returns:
        shap_values: SHAP values array
        base_value: Base value
    """
    if explainer_type == 'tree':
        shap_values_all = explainer.shap_values(X_data)
        
        # Handle binary classification case
        if isinstance(shap_values_all, list):
            shap_values = shap_values_all[1]  # Take positive class
            base_value = explainer.expected_value[1]
        else:
            shap_values = shap_values_all
            base_value = explainer.expected_value
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    
    elif explainer_type == 'linear':
        shap_values_obj = explainer.shap_values(X_data)
        
        if isinstance(shap_values_obj, list):
            shap_values = shap_values_obj[1]
            base_value = explainer.expected_value[1]
        else:
            shap_values = shap_values_obj
            base_value = explainer.expected_value
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    
    else:  # kernel
        shap_values = explainer.shap_values(X_data)
        base_value = explainer.expected_value
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    
    return shap_values, base_value


# --- 1. Smart resource loading: fixed final best model ---

FINAL_BEST_MODEL = "RF"
FINAL_FEATURE_NAMES = [
    "ppHTN",
    "ppCHD",
    "t1Hb",
    "t1TSH",
    "t1TG",
    "t224hUP",
    "t2UA",
    "t2WBC",
    "t2FT4",
    "t2FPG",
    "t2MAP",
]
RISK_THRESHOLD = 0.5
LLM_PROMPT_VERSION = "ckd_apo_rf11_interpretation_v3_2026_11_15"
LLM_TEMPERATURE = 0
LLM_TOP_P = 1
LLM_MAX_TOKENS = 600
LLM_SEED = 6666
LLM_CACHE_DIR = Path(__file__).parent / "llm_interpretation_cache"

FEATURE_INPUT_CONFIG = {
    "ppHTN": {"type": "binary"},
    "ppCHD": {"type": "binary"},
    "t1Hb": {"type": "continuous", "step": 0.1, "format": "%.1f"},
    "t1TSH": {"type": "continuous", "step": 0.01, "format": "%.2f"},
    "t1TG": {"type": "continuous", "step": 0.01, "format": "%.2f"},
    "t224hUP": {"type": "continuous", "step": 0.1, "format": "%.1f"},
    "t2UA": {"type": "continuous", "step": 0.1, "format": "%.1f"},
    "t2WBC": {"type": "continuous", "step": 0.01, "format": "%.2f"},
    "t2FT4": {"type": "continuous", "step": 0.01, "format": "%.2f"},
    "t2FPG": {"type": "continuous", "step": 0.01, "format": "%.2f"},
    "t2MAP": {"type": "continuous", "step": 0.01, "format": "%.2f"},
}

TEXT = {
    "English": {
        "home_title": "LLM-Assisted Interpretation of Machine Learning Prediction for Adverse Pregnancy Outcomes in Pregnant Women with Pre-existing Chronic Kidney Disease",
        "home_p1": (
            "Pregnancy in women with pre-existing chronic kidney disease is clinically complex, "
            "and adverse pregnancy outcomes remain a major concern. Early identification "
            "of high-risk patients may support closer monitoring and individualized clinical assessment."
        ),
        "home_p2": (
            "To assist clinicians in estimating the risk of adverse pregnancy outcomes, "
            "we developed a machine-learning prediction model with optional local large language "
            "model-assisted explanation for individual risk interpretation."
        ),
        "enter": "Enter",
        "page_title": "LLM-Assisted Interpretation of Machine Learning Prediction for Adverse Pregnancy Outcomes in Pregnant Women with Pre-existing Chronic Kidney Disease",
        "single_tab": "Single Prediction",
        "batch_tab": "Batch Prediction",
        "input_note": "Hover over each variable name to view its clinical meaning. All input variables are measured before or during the second trimester.",
        "predict": "Start Prediction",
        "prediction_result": "Prediction Result",
        "risk_probability": "Risk Probability",
        "risk_level": "Risk Level",
        "low_risk": "Low Risk",
        "high_risk": "High Risk",
        "model": "Model",
        "top_features": "Global SHAP Feature Importance",
        "waterfall": "Individual Waterfall Plot",
        "llm_explanation": "LLM-Assisted Individual Interpretation",
        "llm_disabled": "Local LLM explanation is disabled.",
        "help": "Help",
        "llm_settings": "Local LLM Settings",
        "disclaimer": (
            "This tool provides model-predicted risk estimation only. It is not a diagnosis "
            "and should not replace clinical judgment or guideline-based management."
        ),
        "prepregnancy": "Pre-pregnancy Variables",
        "first_trimester": "First-trimester Variables",
        "second_trimester": "Second-trimester Variables",
        "medication": "Medication Variables",
        "general": "General Obstetric Variables",
    },
    "中文": {
        "home_title": "大语言模型辅助解释孕前伴有慢性肾脏病孕妇不良妊娠结局机器学习预测模型",
        "home_p1": (
            "孕前伴有慢性肾脏病的孕妇病情复杂，不良妊娠结局风险较高。"
            "在孕中期之前识别高风险患者，有助于开展更密切的监测和个体化临床评估。"
        ),
        "home_p2": (
            "为辅助临床医生评估不良妊娠结局风险，本研究构建了机器学习预测模型，"
            "并提供可选的本地大语言模型辅助解释模块，用于生成个体化风险解释。"
        ),
        "enter": "进入",
        "page_title": "大语言模型辅助解释孕前伴有慢性肾脏病孕妇不良妊娠结局机器学习预测模型",
        "single_tab": "单人预测",
        "batch_tab": "批量预测",
        "input_note": "将鼠标悬停在变量名称上可查看临床含义。所有输入变量均来自孕中期之前或孕中期时点。",
        "predict": "开始预测",
        "prediction_result": "预测结果",
        "risk_probability": "风险概率",
        "risk_level": "风险分层",
        "low_risk": "低风险",
        "high_risk": "高风险",
        "model": "模型",
        "top_features": "全局 SHAP 特征重要性",
        "waterfall": "个体瀑布图",
        "llm_explanation": "大语言模型辅助个体化解释",
        "llm_disabled": "本地大语言模型解释已关闭。",
        "help": "帮助",
        "llm_settings": "本地大语言模型设置",
        "disclaimer": "本工具仅提供模型预测风险估计，不等同于临床诊断，不能替代医生判断或指南推荐的临床管理。",
        "prepregnancy": "孕前变量",
        "first_trimester": "孕早期变量",
        "second_trimester": "孕中期变量",
        "medication": "用药变量",
        "general": "一般产科变量",
    },
}

FEATURE_META = {
    "ppHTN": {
        "en": "Pre-pregnancy hypertension: 0 = No, 1 = Yes.",
        "zh": "孕前高血压：0=否，1=是。",
        "english_name": "Pre-pregnancy hypertension",
        "chinese_name": "孕前高血压",
        "unit": "0 = No, 1 = Yes",
        "clinical_meaning_en": "Indicates whether hypertension was present before pregnancy and reflects baseline blood-pressure-related disease burden.",
        "clinical_meaning_zh": "表示孕前是否合并高血压，反映基础血压相关疾病负担。",
    },
    "Height": {
        "en": "Height before pregnancy or at baseline, cm.",
        "zh": "身高，单位 cm。",
        "english_name": "Height",
        "chinese_name": "身高",
        "unit": "cm",
        "clinical_meaning_en": "Baseline body size indicator used by the prediction model.",
        "clinical_meaning_zh": "反映基础体格特征，是模型中的基线体格指标。",
    },
    "ppDM": {
        "en": "Pre-pregnancy diabetes mellitus: 0 = No, 1 = Yes.",
        "zh": "孕前糖尿病：0=否，1=是。",
        "english_name": "Pre-pregnancy diabetes mellitus",
        "chinese_name": "孕前糖尿病",
        "unit": "0 = No, 1 = Yes",
        "clinical_meaning_en": "Indicates whether diabetes was present before pregnancy.",
        "clinical_meaning_zh": "表示孕前是否合并糖尿病，反映基础代谢疾病负担。",
    },
    "ppCHD": {
        "en": "Pre-pregnancy heart disease: 0 = No, 1 = Yes.",
        "zh": "孕前心脏病：0=否，1=是。",
        "english_name": "Pre-pregnancy heart disease",
        "chinese_name": "孕前心脏病",
        "unit": "0 = No, 1 = Yes",
        "clinical_meaning_en": "Indicates whether heart disease was present before pregnancy and reflects baseline cardiovascular disease burden.",
        "clinical_meaning_zh": "表示孕前是否合并心脏病，反映基础心血管疾病负担。",
    },
    "Gravidity": {
        "en": "Number of previous and current pregnancies, count.",
        "zh": "孕次，单位：次。",
        "english_name": "Gravidity",
        "chinese_name": "孕次",
        "unit": "count",
        "clinical_meaning_en": "Represents reproductive history including the current pregnancy.",
        "clinical_meaning_zh": "表示包括本次妊娠在内的妊娠次数，反映既往妊娠经历。",
    },
    "ppCsA": {
        "en": "Cyclosporine use before pregnancy: 0 = No, 1 = Yes.",
        "zh": "孕前是否使用环孢素A：0=否，1=是。",
        "english_name": "Pre-pregnancy cyclosporine A use",
        "chinese_name": "孕前是否使用环孢素A",
        "unit": "0 = No, 1 = Yes",
        "clinical_meaning_en": "Represents pre-pregnancy immunosuppressive treatment exposure and disease-treatment burden.",
        "clinical_meaning_zh": "反映孕前免疫抑制治疗暴露及疾病治疗负荷。",
    },
    "apAntiHTN": {
        "en": "Antihypertensive medication use before or during the second trimester: 0 = No, 1 = Yes.",
        "zh": "怀孕至孕中期之前/截至孕中期是否使用降压药：0=否，1=是。",
        "english_name": "Antihypertensive medication use up to the second trimester",
        "chinese_name": "孕期截至孕中期是否使用降压药",
        "unit": "0 = No, 1 = Yes",
        "clinical_meaning_en": "Represents antihypertensive treatment exposure up to the second trimester and may reflect hypertension-related treatment burden.",
        "clinical_meaning_zh": "反映截至孕中期的降压治疗暴露，可能提示高血压相关治疗负荷。",
    },
    "apCorticosteroid": {
        "en": "Corticosteroid use before or during the second trimester: 0 = No, 1 = Yes.",
        "zh": "怀孕至孕中期之前/截至孕中期是否使用糖皮质激素：0=否，1=是。",
        "english_name": "Corticosteroid use up to the second trimester",
        "chinese_name": "孕期截至孕中期是否使用糖皮质激素",
        "unit": "0 = No, 1 = Yes",
        "clinical_meaning_en": "Represents corticosteroid treatment exposure up to the second trimester and may reflect inflammatory or autoimmune disease activity.",
        "clinical_meaning_zh": "反映截至孕中期的糖皮质激素治疗暴露，可能提示炎症或自身免疫疾病相关治疗负荷。",
    },
    "apHeparin": {
        "en": "Heparin use before or during the second trimester: 0 = No, 1 = Yes.",
        "zh": "怀孕至孕中期之前/截至孕中期是否使用肝素：0=否，1=是。",
        "english_name": "Heparin use up to the second trimester",
        "chinese_name": "孕期截至孕中期是否使用肝素",
        "unit": "0 = No, 1 = Yes",
        "clinical_meaning_en": "Represents anticoagulation exposure up to the second trimester and may reflect thrombotic or pregnancy-risk management context.",
        "clinical_meaning_zh": "反映截至孕中期的抗凝治疗暴露，可能提示血栓风险或妊娠风险管理背景。",
    },
    "ppSCr": {
        "en": "Pre-pregnancy serum creatinine, umol/L.",
        "zh": "孕前血清肌酐，单位 umol/L。",
        "english_name": "Pre-pregnancy serum creatinine",
        "chinese_name": "孕前血清肌酐",
        "unit": "umol/L",
        "clinical_meaning_en": "Reflects baseline renal function before pregnancy.",
        "clinical_meaning_zh": "反映孕前基础肾功能状态。",
    },
    "t1Hb": {
        "en": "First-trimester hemoglobin, g/L.",
        "zh": "孕早期血红蛋白，单位 g/L。",
        "english_name": "First-trimester hemoglobin",
        "chinese_name": "孕早期血红蛋白",
        "unit": "g/L",
        "clinical_meaning_en": "Reflects anemia status and oxygen-carrying capacity in early pregnancy.",
        "clinical_meaning_zh": "反映孕早期贫血状态和携氧能力。",
    },
    "t1TSH": {
        "en": "First-trimester thyroid-stimulating hormone, mIU/L.",
        "zh": "孕早期促甲状腺激素，单位 mIU/L。",
        "english_name": "First-trimester thyroid-stimulating hormone",
        "chinese_name": "孕早期促甲状腺激素",
        "unit": "mIU/L",
        "clinical_meaning_en": "Reflects thyroid regulatory status during the first trimester.",
        "clinical_meaning_zh": "反映孕早期甲状腺调节状态。",
    },
    "t1TG": {
        "en": "First-trimester triglycerides, mmol/L.",
        "zh": "孕早期甘油三酯，单位 mmol/L。",
        "english_name": "First-trimester triglycerides",
        "chinese_name": "孕早期甘油三酯",
        "unit": "mmol/L",
        "clinical_meaning_en": "Reflects lipid metabolism during the first trimester.",
        "clinical_meaning_zh": "反映孕早期脂质代谢状态。",
    },
    "t1GGT": {
        "en": "First-trimester gamma-glutamyl transferase, U/L.",
        "zh": "孕早期γ-谷氨酰转肽酶，单位 U/L。",
        "english_name": "First-trimester gamma-glutamyl transferase",
        "chinese_name": "孕早期γ-谷氨酰转肽酶",
        "unit": "U/L",
        "clinical_meaning_en": "Reflects hepatobiliary enzyme status in early pregnancy.",
        "clinical_meaning_zh": "反映孕早期肝胆酶学状态。",
    },
    "t1FT4": {
        "en": "First-trimester free thyroxine, pmol/L.",
        "zh": "孕早期游离甲状腺素，单位 pmol/L。",
        "english_name": "First-trimester free thyroxine",
        "chinese_name": "孕早期游离甲状腺素",
        "unit": "pmol/L",
        "clinical_meaning_en": "Reflects thyroid hormone status in early pregnancy.",
        "clinical_meaning_zh": "反映孕早期甲状腺激素水平。",
    },
    "t1dsDNA": {
        "en": "First-trimester anti-double-stranded DNA antibody, IU/mL.",
        "zh": "孕早期抗双链DNA抗体，单位 IU/mL。",
        "english_name": "First-trimester anti-double-stranded DNA antibody",
        "chinese_name": "孕早期抗双链DNA抗体",
        "unit": "IU/mL",
        "clinical_meaning_en": "Reflects autoimmune activity context in early pregnancy.",
        "clinical_meaning_zh": "反映孕早期自身免疫活动相关背景。",
    },
    "t224hUP": {
        "en": "Second-trimester 24-hour urinary protein, mg/24h.",
        "zh": "孕中期24小时尿蛋白定量，单位 mg/24h。",
        "english_name": "Second-trimester 24-hour urinary protein",
        "chinese_name": "孕中期24小时尿蛋白定量",
        "unit": "mg/24h",
        "clinical_meaning_en": "Reflects proteinuria burden and renal glomerular barrier injury during the second trimester.",
        "clinical_meaning_zh": "反映孕中期蛋白尿负荷及肾小球滤过屏障损伤程度。",
    },
    "t2BUN": {
        "en": "Second-trimester blood urea nitrogen, mmol/L.",
        "zh": "孕中期尿素氮，单位 mmol/L。",
        "english_name": "Second-trimester blood urea nitrogen",
        "chinese_name": "孕中期尿素氮",
        "unit": "mmol/L",
        "clinical_meaning_en": "Reflects nitrogen metabolism and renal excretory burden during the second trimester.",
        "clinical_meaning_zh": "反映孕中期氮代谢及肾脏排泄负担。",
    },
    "t2UA": {
        "en": "Second-trimester uric acid, umol/L.",
        "zh": "孕中期尿酸，单位 umol/L。",
        "english_name": "Second-trimester uric acid",
        "chinese_name": "孕中期尿酸",
        "unit": "umol/L",
        "clinical_meaning_en": "Reflects purine metabolism, renal excretion, and pregnancy-related metabolic burden during the second trimester.",
        "clinical_meaning_zh": "反映孕中期嘌呤代谢、肾脏排泄及妊娠相关代谢负担。",
    },
    "t2WBC": {
        "en": "Second-trimester white blood cell count, ×10^9/L.",
        "zh": "孕中期白细胞计数，单位 ×10^9/L。",
        "english_name": "Second-trimester white blood cell count",
        "chinese_name": "孕中期白细胞计数",
        "unit": "×10^9/L",
        "clinical_meaning_en": "Reflects the circulating white blood cell level during the second trimester.",
        "clinical_meaning_zh": "反映孕中期外周血白细胞水平。",
    },
    "t2FT4": {
        "en": "Second-trimester free thyroxine, pmol/L.",
        "zh": "孕中期游离甲状腺素，单位 pmol/L。",
        "english_name": "Second-trimester free thyroxine",
        "chinese_name": "孕中期游离甲状腺素",
        "unit": "pmol/L",
        "clinical_meaning_en": "Reflects thyroid hormone status during the second trimester.",
        "clinical_meaning_zh": "反映孕中期甲状腺激素水平。",
    },
    "t2FPG": {
        "en": "Second-trimester fasting plasma glucose, mmol/L.",
        "zh": "孕中期空腹血糖，单位 mmol/L。",
        "english_name": "Second-trimester fasting plasma glucose",
        "chinese_name": "孕中期空腹血糖",
        "unit": "mmol/L",
        "clinical_meaning_en": "Reflects glucose metabolism status during the second trimester.",
        "clinical_meaning_zh": "反映孕中期糖代谢状态。",
    },
    "t2MAP": {
        "en": "Second-trimester mean arterial pressure, mmHg.",
        "zh": "孕中期平均动脉压，单位 mmHg。",
        "english_name": "Second-trimester mean arterial pressure",
        "chinese_name": "孕中期平均动脉压",
        "unit": "mmHg",
        "clinical_meaning_en": "Reflects average arterial pressure and blood-pressure burden during the second trimester.",
        "clinical_meaning_zh": "反映孕中期平均动脉压水平及血压负荷。",
    },
}

FEATURE_GROUPS = {
    "prepregnancy": ["ppHTN", "ppCHD"],
    "first_trimester": ["t1Hb", "t1TSH", "t1TG"],
    "second_trimester": ["t224hUP", "t2UA", "t2WBC", "t2FT4", "t2FPG", "t2MAP"],
}

HIGH_RISK_EXAMPLE_VALUES = {
    # Test set row 90: true APO = 1; RF predicted probability = 0.8407.
    "ppHTN": 1,
    "ppCHD": 0,
    "t1Hb": 88.0,
    "t1TSH": 0.05,
    "t1TG": 1.07,
    "t224hUP": 4719.11,
    "t2UA": 406.08,
    "t2WBC": 11.27,
    "t2FT4": 14.67,
    "t2FPG": 5.23,
    "t2MAP": 100.67,
}

GLOBAL_SHAP_IMPORTANCE = {
    "t2MAP": 0.0897097713,
    "t2UA": 0.0788977316,
    "t1TG": 0.0505515260,
    "t224hUP": 0.0496946219,
    "t2FT4": 0.0310641270,
    "t2FPG": 0.0281550954,
    "t1Hb": 0.0267959189,
    "t1TSH": 0.0198549180,
    "t2WBC": 0.0113820097,
    "ppHTN": 0.0022209060,
    "ppCHD": 0.0005418498,
}


def render_feature_input(feature_name, language="English", default_values=None):
    config = FEATURE_INPUT_CONFIG.get(
        feature_name,
        {"type": "continuous", "step": 0.1, "format": "%.4f"},
    )
    default_values = default_values or {}
    default_value = default_values.get(feature_name, 0)
    label = feature_name
    help_text = FEATURE_META.get(feature_name, {}).get("zh" if language == "中文" else "en", None)
    if help_text:
        st.markdown(
            (
                '<div class="feature-label-wrap">'
                f'<span class="feature-label">{html.escape(label)}</span>'
                f'<div class="feature-tooltip">{html.escape(help_text)}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    if config["type"] == "binary":
        default_index = 1 if float(default_value) >= 0.5 else 0
        choice = st.selectbox(
            label,
            options=["No (0)", "Yes (1)"],
            index=default_index,
            key=f"input_{feature_name}",
            label_visibility="collapsed" if help_text else "visible",
        )
        return 1 if choice == "Yes (1)" else 0

    if config["type"] == "integer":
        return st.number_input(
            label,
            min_value=0,
            value=int(round(float(default_value))),
            step=1,
            format="%d",
            key=f"input_{feature_name}",
            label_visibility="collapsed" if help_text else "visible",
        )

    return st.number_input(
        label,
        min_value=0.0,
        value=float(default_value),
        step=float(config.get("step", 0.1)),
        format=config.get("format", "%.4f"),
        key=f"input_{feature_name}",
        label_visibility="collapsed" if help_text else "visible",
    )


def render_grouped_inputs(feature_names, language, default_values=None):
    inputs = {}
    rendered = set()

    for group_key, group_features in FEATURE_GROUPS.items():
        active_features = [feat for feat in group_features if feat in feature_names]
        if not active_features:
            continue
        with st.container(border=True):
            st.markdown(f"#### {TEXT[language][group_key]}")
            n_cols = 3 if len(active_features) <= 4 else 4
            cols = st.columns(n_cols)
            for i, feat in enumerate(active_features):
                with cols[i % n_cols]:
                    inputs[feat] = render_feature_input(feat, language, default_values)
                    rendered.add(feat)

    remaining_features = [feat for feat in feature_names if feat not in rendered]
    if remaining_features:
        with st.container(border=True):
            st.markdown("#### Other Variables")
            cols = st.columns(4)
            for i, feat in enumerate(remaining_features):
                with cols[i % 4]:
                    inputs[feat] = render_feature_input(feat, language, default_values)

    return inputs


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@st.cache_data
def load_default_input_values(feature_names):
    """
    Use a representative high-risk test-set case as the initial example.
    Fall back to training-set medians/modes if a feature is not in the example.
    """
    train_path = Path(__file__).parent / "训练集.csv"
    defaults = {
        feature: HIGH_RISK_EXAMPLE_VALUES[feature]
        for feature in feature_names
        if feature in HIGH_RISK_EXAMPLE_VALUES
    }

    if not train_path.exists():
        return {feature: defaults.get(feature, 0) for feature in feature_names}

    train_df = pd.read_csv(train_path)
    for feature in feature_names:
        if feature in defaults:
            continue
        if feature not in train_df.columns:
            defaults[feature] = 0
            continue
        series = pd.to_numeric(train_df[feature], errors="coerce").dropna()
        if series.empty:
            defaults[feature] = 0
            continue
        if FEATURE_INPUT_CONFIG.get(feature, {}).get("type") == "binary":
            defaults[feature] = int(series.mode().iloc[0])
        else:
            defaults[feature] = float(series.median())
    return defaults


def build_contribution_table(shap_vector, patient_row, feature_names, top_n=None):
    """
    Build a compact table of the strongest local SHAP contributions.
    Positive values push the predicted APO risk upward; negative values push it downward.
    """
    contribution_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": [patient_row[name] for name in feature_names],
        "Contribution": np.asarray(shap_vector, dtype=float),
    })
    contribution_df["AbsContribution"] = contribution_df["Contribution"].abs()
    contribution_df = contribution_df.sort_values("AbsContribution", ascending=False)
    if top_n is None:
        return contribution_df
    return contribution_df.head(top_n)


def plot_top_contributions(contribution_df):
    plot_df = contribution_df.sort_values("AbsContribution", ascending=True).copy()
    labels = [
        f"{row.Feature} = {row.Value:g}" if isinstance(row.Value, (int, float, np.integer, np.floating))
        else f"{row.Feature} = {row.Value}"
        for row in plot_df.itertuples()
    ]
    colors = ["#d64f4f" if v > 0 else "#2d86d8" for v in plot_df["Contribution"]]

    fig_height = max(8.8, 0.42 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(8.0, fig_height), dpi=120)
    ax.barh(labels, plot_df["Contribution"], color=colors, alpha=0.9)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    max_abs = float(plot_df["Contribution"].abs().max()) if len(plot_df) else 0.0
    pad = max(max_abs * 0.035, 0.002)
    for y_pos, value in enumerate(plot_df["Contribution"]):
        if value >= 0:
            label_x = pad
            ha = "left"
        else:
            label_x = -pad
            ha = "right"
        text_color = "#ffffff" if abs(value) > max_abs * 0.10 else "#172033"
        ax.text(
            label_x,
            y_pos,
            f"{value:+.3f}",
            va="center",
            ha=ha,
            fontsize=7,
            fontweight="700",
            color=text_color,
        )
    if max_abs > 0:
        ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)
    ax.set_xlabel("SHAP value", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.44, right=0.97, top=0.965, bottom=0.12)
    return fig


def plot_waterfall_explanation(contribution_df, base_value):
    plot_df = contribution_df.sort_values("AbsContribution", ascending=False).copy()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    labels = [
        f"{row.Feature} = {row.Value:g}" if isinstance(row.Value, (int, float, np.integer, np.floating))
        else f"{row.Feature} = {row.Value}"
        for row in plot_df.itertuples()
    ]

    base = float(np.asarray(base_value).reshape(-1)[0])
    contributions = plot_df["Contribution"].astype(float).to_numpy()
    starts = base + np.cumsum(np.r_[0, contributions[:-1]])
    ends = starts + contributions
    lefts = np.minimum(starts, ends)
    widths = np.abs(contributions)
    colors = ["#d64f4f" if value > 0 else "#2d86d8" for value in contributions]

    fig_height = max(8.8, 0.42 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(8.0, fig_height), dpi=120)
    y_positions = np.arange(len(plot_df))
    ax.barh(y_positions, widths, left=lefts, color=colors, alpha=0.92, height=0.70)

    x_values = np.r_[starts, ends, base]
    x_span = float(x_values.max() - x_values.min()) if len(x_values) else 0.0
    pad = max(x_span * 0.04, 0.01)
    for y_pos, start, end, value in zip(y_positions, starts, ends, contributions):
        label_x = end + pad if value >= 0 else end - pad
        ha = "left" if value >= 0 else "right"
        ax.text(label_x, y_pos, f"{value:+.3f}", va="center", ha=ha, fontsize=7, fontweight="700")
        ax.plot([start, end], [y_pos, y_pos], color="none")

    final_value = base + contributions.sum()
    ax.text(base, len(plot_df) - 0.25, f"E[f(X)] = {base * 100:.2f}%", ha="center", va="bottom", fontsize=7, color="#666666")
    ax.text(final_value, len(plot_df) - 0.45, f"f(x) = {final_value * 100:.2f}%", ha="center", va="bottom", fontsize=7, color="#222222", fontweight="700")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Model output", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", linestyle=":", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    min_x = min(x_values.min(), final_value) - max(x_span * 0.12, 0.02)
    max_x = max(x_values.max(), final_value) + max(x_span * 0.12, 0.02)
    ax.set_xlim(min_x, max_x)
    fig.subplots_adjust(left=0.44, right=0.97, top=0.965, bottom=0.075)
    return fig


def plot_global_importance():
    """
    Display mean absolute SHAP values from the final RF model on the 2024 test cohort.
    """
    importance_values = GLOBAL_SHAP_IMPORTANCE
    plot_df = (
        pd.DataFrame({
            "Feature": list(importance_values.keys()),
            "Importance": list(importance_values.values()),
        })
        .sort_values("Importance", ascending=True)
    )

    fig_height = max(8.8, 0.42 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(8.0, fig_height), dpi=300)
    bar_colors = plt.cm.Blues(np.linspace(0.42, 0.78, len(plot_df)))
    ax.barh(plot_df["Feature"], plot_df["Importance"], color=bar_colors, alpha=0.95)
    for y_pos, value in enumerate(plot_df["Importance"]):
        ax.text(value + 0.002, y_pos, f"{value:.4f}", va="center", ha="left", fontsize=7, fontweight="700")

    ax.set_xlabel("Mean |SHAP value|", fontsize=7)
    ax.set_ylabel("Features", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(importance_values.values()) * 1.22)
    fig.subplots_adjust(left=0.34, right=0.94, top=0.965, bottom=0.12)
    return fig


def plot_combined_explanation(contribution_df, base_value):
    """
    Draw global feature importance and individual waterfall on one canvas so axes align.
    """
    importance_values = GLOBAL_SHAP_IMPORTANCE
    top_df = (
        pd.DataFrame({
            "Feature": list(importance_values.keys()),
            "Importance": list(importance_values.values()),
        })
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )
    water_df = contribution_df.sort_values("AbsContribution", ascending=False).copy()
    water_df = water_df.iloc[::-1].reset_index(drop=True)

    def make_labels(df):
        return [
            f"{row.Feature} = {row.Value:g}" if isinstance(row.Value, (int, float, np.integer, np.floating))
            else f"{row.Feature} = {row.Value}"
            for row in df.itertuples()
        ]

    fig_height = max(9.8, 0.46 * len(top_df) + 2.0)
    fig = plt.figure(figsize=(18.0, fig_height), dpi=360, facecolor="white")

    # Manual axes placement keeps both plots the same size and locks x-axis alignment.
    axis_bottom = 0.13
    axis_height = 0.80
    axis_width = 0.36
    ax_left = fig.add_axes([0.08, axis_bottom, axis_width, axis_height])
    ax_right = fig.add_axes([0.58, axis_bottom, axis_width, axis_height])

    labels_left = top_df["Feature"].tolist()
    ax_left.barh(labels_left, top_df["Importance"], color="#2d86d8", alpha=0.92, height=0.70)
    for y_pos, value in enumerate(top_df["Importance"]):
        ax_left.text(value + 0.002, y_pos, f"{value:.4f}", va="center", ha="left", fontsize=9, fontweight="700")
    ax_left.set_xlabel("Mean |SHAP value|", fontsize=10, labelpad=8)
    ax_left.set_ylabel("Features", fontsize=10, labelpad=8)
    ax_left.set_xlim(0, max(importance_values.values()) * 1.22)
    ax_left.tick_params(axis="y", labelsize=9)
    ax_left.tick_params(axis="x", labelsize=9)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.grid(axis="x", linestyle=":", alpha=0.22)

    labels_right = make_labels(water_df)
    base = float(np.asarray(base_value).reshape(-1)[0])
    contributions = water_df["Contribution"].astype(float).to_numpy()
    starts = base + np.cumsum(np.r_[0, contributions[:-1]])
    ends = starts + contributions
    lefts = np.minimum(starts, ends)
    widths = np.abs(contributions)
    colors_right = ["#d64f4f" if value > 0 else "#2d86d8" for value in contributions]
    y_positions = np.arange(len(water_df))

    ax_right.barh(y_positions, widths, left=lefts, color=colors_right, alpha=0.92, height=0.70)
    x_values = np.r_[starts, ends, base]
    x_span = float(x_values.max() - x_values.min()) if len(x_values) else 0.0
    right_pad = max(x_span * 0.05, 0.012)
    for y_pos, end, value in zip(y_positions, ends, contributions):
        label_x = end + right_pad if value >= 0 else end - right_pad
        ha = "left" if value >= 0 else "right"
        ax_right.text(label_x, y_pos, f"{value:+.3f}", va="center", ha=ha, fontsize=9, fontweight="700")

    final_value = base + contributions.sum()
    ax_right.text(base, len(water_df) - 0.25, f"E[f(X)] = {base * 100:.2f}%", ha="center", va="bottom", fontsize=9, color="#666666")
    ax_right.text(final_value, len(water_df) - 0.45, f"f(x) = {final_value * 100:.2f}%", ha="center", va="bottom", fontsize=9, color="#222222", fontweight="700")
    ax_right.set_yticks(y_positions)
    ax_right.set_yticklabels(labels_right)
    ax_right.set_xlabel("Model output", fontsize=10, labelpad=8)
    ax_right.tick_params(axis="y", labelsize=9)
    ax_right.tick_params(axis="x", labelsize=9)
    ax_right.grid(axis="x", linestyle=":", alpha=0.22)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    min_x = min(x_values.min(), final_value) - max(x_span * 0.12, 0.02)
    max_x = max(x_values.max(), final_value) + max(x_span * 0.12, 0.02)
    ax_right.set_xlim(min_x, max_x)

    return fig


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def _format_feature_value(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    return str(value)


def _feature_meta_for_prompt(feature_name):
    meta = FEATURE_META.get(feature_name, {})
    return {
        "variable": feature_name,
        "english_name": meta.get("english_name", feature_name),
        "chinese_name": meta.get("chinese_name", feature_name),
        "unit": meta.get("unit", "N/A"),
        "clinical_meaning_en": meta.get("clinical_meaning_en", meta.get("en", "")),
        "clinical_meaning_zh": meta.get("clinical_meaning_zh", meta.get("zh", "")),
    }


def _build_llm_rows(contribution_df, direction, top_n=6):
    if direction == "increase":
        selected = contribution_df[contribution_df["Contribution"] > 0].sort_values("Contribution", ascending=False)
    else:
        selected = contribution_df[contribution_df["Contribution"] < 0].sort_values("Contribution", ascending=True)

    rows = []
    for row in selected.head(top_n).itertuples():
        meta = _feature_meta_for_prompt(row.Feature)
        rows.append({
            **meta,
            "value": _format_feature_value(row.Value),
            "shap_contribution": round(float(row.Contribution), 4),
            "direction": (
                "Increases model-predicted APO probability"
                if direction == "increase"
                else "Decreases model-predicted APO probability"
            ),
        })
    return rows


def _format_llm_rows(rows, language):
    if not rows:
        return "None among the displayed SHAP contribution variables."

    lines = []
    for row in rows:
        if language == "中文":
            lines.append(
                f"- {row['variable']} ({row['chinese_name']}; {row['english_name']}; "
                f"unit: {row['unit']}): value={row['value']}, "
                f"SHAP contribution={row['shap_contribution']}; "
                f"clinical meaning: {row['clinical_meaning_zh']}"
            )
        else:
            lines.append(
                f"- {row['variable']} ({row['english_name']}; Chinese name: {row['chinese_name']}; "
                f"unit: {row['unit']}): value={row['value']}, "
                f"SHAP contribution={row['shap_contribution']}; "
                f"clinical meaning: {row['clinical_meaning_en']}"
            )
    return "\n".join(lines)


def _build_variable_dictionary(feature_names, language):
    lines = []
    for feature_name in feature_names:
        meta = _feature_meta_for_prompt(feature_name)
        if language == "中文":
            lines.append(
                f"- {feature_name}: {meta['chinese_name']} / {meta['english_name']}; "
                f"unit: {meta['unit']}; {meta['clinical_meaning_zh']}"
            )
        else:
            lines.append(
                f"- {feature_name}: {meta['english_name']} / {meta['chinese_name']}; "
                f"unit: {meta['unit']}; {meta['clinical_meaning_en']}"
            )
    return "\n".join(lines)


def _build_llm_prompt_payload(probability, contribution_df, model_name, language):
    feature_names = contribution_df["Feature"].tolist()
    risk_level = "High Risk" if probability > RISK_THRESHOLD else "Low Risk"
    if language == "中文":
        risk_level = "高风险" if probability > RISK_THRESHOLD else "低风险"

    increase_rows = _build_llm_rows(contribution_df, "increase")
    decrease_rows = _build_llm_rows(contribution_df, "decrease")

    return {
        "prompt_version": LLM_PROMPT_VERSION,
        "language": language,
        "llm_model": model_name,
        "ml_model_output": {
            "predicted_apo_probability_percent": round(float(probability) * 100, 2),
            "risk_level": risk_level,
        },
        "shap_interpretation_rule": (
            "Positive SHAP values push the model output toward a higher predicted APO probability; "
            "negative SHAP values push the model output toward a lower predicted APO probability. "
            "SHAP values are model-level contributions, not causal effects."
        ),
        "top_risk_increasing_variables": increase_rows,
        "top_risk_decreasing_variables": decrease_rows,
        "feature_order": feature_names,
        "all_feature_contributions": [
            {
                **_feature_meta_for_prompt(row.Feature),
                "value": _format_feature_value(row.Value),
                "shap_contribution": round(float(row.Contribution), 4),
            }
            for row in contribution_df.sort_values("Feature").itertuples()
        ],
    }


def _llm_cache_key(payload, system_content, prompt):
    cache_payload = {
        "payload": payload,
        "system_content": system_content,
        "prompt": prompt,
        "generation_parameters": {
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "max_tokens": LLM_MAX_TOKENS,
            "seed": LLM_SEED,
        },
    }
    canonical = json.dumps(cache_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_cached_llm_interpretation(cache_key):
    cache_path = LLM_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f).get("interpretation")
    except Exception:
        return None


def _write_cached_llm_interpretation(cache_key, interpretation, payload, system_content, prompt):
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = LLM_CACHE_DIR / f"{cache_key}.json"
    record = {
        "cache_key": cache_key,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_version": LLM_PROMPT_VERSION,
        "generation_parameters": {
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "max_tokens": LLM_MAX_TOKENS,
            "seed": LLM_SEED,
        },
        "payload": payload,
        "system_content": system_content,
        "prompt": prompt,
        "interpretation": interpretation,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=_json_default)


def generate_local_llm_explanation(probability, contribution_df, base_url, model_name, language="English"):
    """
    Generate an optional local LLM interpretation through LM Studio's OpenAI-compatible API.
    Identical structured inputs are cached so repeated runs return the same text.
    """
    payload = _build_llm_prompt_payload(probability, contribution_df, model_name, language)
    increasing_text = _format_llm_rows(payload["top_risk_increasing_variables"], language)
    decreasing_text = _format_llm_rows(payload["top_risk_decreasing_variables"], language)
    variable_dictionary = _build_variable_dictionary(payload["feature_order"], language)
    risk_probability = payload["ml_model_output"]["predicted_apo_probability_percent"]
    risk_level = payload["ml_model_output"]["risk_level"]

    if language == "中文":
        system_content = (
            "你是医学风险预测解释专家。"
            "你的任务是基于机器学习模型输出的预测概率和SHAP贡献变量，"
            "生成谨慎、清晰、可理解的个体化风险解释。"
            "只能使用用户提供的模型输出、患者变量和SHAP贡献。"
            "不要虚构患者信息、检验数值、参考文献、临床诊断、因果结论或具体治疗建议。"
            "正SHAP值表示该变量将模型预测结果推向更高的APO预测概率；"
            "负SHAP值表示该变量将模型预测结果推向更低的APO预测概率。"
            "SHAP解释只代表模型层面的预测贡献，不代表真实因果效应。"
        )
        prompt = f"""
机器学习模型预测结果：
- 预测不良妊娠结局(APO)概率：{risk_probability:.2f}%
- 风险分层：{risk_level}

SHAP解释规则：
正SHAP值表示该变量将模型预测结果推向更高的APO预测概率；负SHAP值表示该变量将模型预测结果推向更低的APO预测概率。该解释反映模型内部预测贡献，不代表因果作用。

推高模型预测概率的主要变量：
{increasing_text}

降低模型预测概率的主要变量：
{decreasing_text}

变量词典：
{variable_dictionary}

请严格按照以下4个小标题输出中文解释：
1. 预测风险
说明模型预测的APO风险，并强调这不是临床诊断。

2. 推高预测风险的主要因素
结合变量取值、单位和临床含义解释主要正SHAP变量。不要使用“导致”等因果表述。

3. 降低预测风险的主要因素
结合变量取值、单位和临床含义解释主要负SHAP变量。

4. 临床解释与注意事项
说明该结果仅用于辅助风险沟通和临床理解，不能替代医生判断，也不要给出具体治疗建议。

总长度控制在180-250个中文字。
"""
    else:
        system_content = (
            "You are an expert in medical risk prediction interpretation. "
            "Your task is to generate cautious, clear, and understandable individualized risk interpretations "
            "based on the machine-learning model's predicted probability and SHAP-derived feature contributions. "
            "Use only the provided model output, patient variables, and SHAP contributions. "
            "Do not invent patient information, laboratory values, references, clinical diagnoses, causal claims, "
            "or specific treatment recommendations. "
            "Positive SHAP values push the model output toward a higher predicted APO probability; "
            "negative SHAP values push the model output toward a lower predicted APO probability. "
            "SHAP explanations are model-level prediction contributions, not causal effects. "
        )
        prompt = f"""
Machine-learning model output:
- Predicted adverse pregnancy outcome (APO) probability: {risk_probability:.2f}%
- Risk category: {risk_level}

SHAP interpretation rule:
Positive SHAP values push the model output toward a higher predicted APO probability; negative SHAP values push the model output toward a lower predicted APO probability. These are model-level prediction contributions, not causal effects.

Main variables increasing the model-predicted probability:
{increasing_text}

Main variables decreasing the model-predicted probability:
{decreasing_text}

Variable dictionary:
{variable_dictionary}

Please generate an English interpretation using exactly the following four headings:
1. Predicted risk
State the model-predicted APO risk and emphasize that this is not a clinical diagnosis.

2. Main factors increasing predicted risk
Explain the main positive SHAP variables using their values, units, and clinical meanings. Avoid causal wording such as "caused by".

3. Main factors decreasing predicted risk
Explain the main negative SHAP variables using their values, units, and clinical meanings.

4. Clinical interpretation and caution
Explain that this result is intended only to support risk communication and clinical understanding, cannot replace clinician judgment, and should not be used as a treatment recommendation.

Keep the interpretation within 120-180 words.
"""

    cache_key = _llm_cache_key(payload, system_content, prompt)
    cached_interpretation = _read_cached_llm_interpretation(cache_key)
    if cached_interpretation is not None:
        return cached_interpretation

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed in the current Python environment.") from exc

    client = OpenAI(base_url=base_url, api_key="lm-studio")
    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False,
        "seed": LLM_SEED,
    }

    try:
        response = client.chat.completions.create(**request_kwargs)
    except TypeError:
        request_kwargs.pop("seed", None)
        response = client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        if "seed" not in str(exc).lower():
            raise
        request_kwargs.pop("seed", None)
        response = client.chat.completions.create(**request_kwargs)

    interpretation = response.choices[0].message.content
    _write_cached_llm_interpretation(cache_key, interpretation, payload, system_content, prompt)
    return interpretation

@st.cache_resource
def load_artifacts(model_name=None):
    """
    Load model, Scaler and feature names
    
    Args:
        model_name: Specified model name (without .pkl suffix), e.g., "LR"
                   If None, load the final best model selected by validation analysis
    
    Returns:
        model: Trained model
        scaler: Standardizer
        feature_names: Feature name list
        model_name: Actually used model name
    """
    model_dir = Path(__file__).parent / "saved_models"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    
    # 1. Load Scaler
    scaler_path = model_dir / "scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    # 2. Load feature names
    feature_path = model_dir / "feature_names.pkl"
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature names not found: {feature_path}")
    feature_names = joblib.load(feature_path)
    feature_names = list(feature_names)
    if feature_names != FINAL_FEATURE_NAMES:
        raise ValueError(
            "The deployed feature list does not match the final RF model. "
            f"Expected {FINAL_FEATURE_NAMES}, received {feature_names}."
        )
    
    # 3. Load model
    best_model_name = FINAL_BEST_MODEL if model_name is None else model_name
    
    # Convert model name to filename (replace spaces with underscores)
    model_filename = best_model_name.replace(' ', '_') + '.pkl'
    model_path = model_dir / model_filename
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    if type(model).__name__ != "RandomForestClassifier":
        raise TypeError(
            f"Expected RandomForestClassifier for the final model, received {type(model).__name__}."
        )
    if getattr(model, "n_features_in_", len(feature_names)) != len(feature_names):
        raise ValueError("The RF model and deployed feature list have different dimensions.")
    if getattr(scaler, "n_features_in_", len(feature_names)) != len(feature_names):
        raise ValueError("The scaler and deployed feature list have different dimensions.")
    
    return model, scaler, feature_names, best_model_name


# --- 2. Page configuration ---

st.set_page_config(
    page_title="CKD Pregnancy APO Prediction",
    page_icon="🏥",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem;
        max-width: 100%;
        padding-left: 3.2rem;
        padding-right: 3.2rem;
    }
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    .home-fullbleed {
        margin-left: -3.2rem;
        margin-right: -3.2rem;
        margin-top: 0;
    }
    .home-nav {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        min-height: 64px;
        padding: 0 46px;
        background:
            linear-gradient(120deg, rgba(57, 118, 230, 0.94) 0%, rgba(87, 92, 226, 0.92) 100%),
            radial-gradient(circle at 0% 70%, rgba(255,255,255,0.18), transparent 28%);
        color: #ffffff;
    }
    .language-menu {
        position: relative;
        display: inline-block;
        font-family: Arial, sans-serif;
        font-size: 15px;
        font-weight: 700;
        cursor: default;
        padding: 20px 16px;
        letter-spacing: 0;
    }
    .language-menu span {
        color: #ffffff;
    }
    .prediction-language-menu {
        padding: 10px 16px;
    }
    .prediction-language-menu span {
        color: #172033;
    }
    .language-menu-content {
        display: none;
        position: absolute;
        top: 100%;
        right: 0;
        min-width: 142px;
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        overflow: hidden;
        z-index: 9999;
    }
    .language-menu:hover .language-menu-content {
        display: block;
    }
    .language-menu-content a {
        display: block;
        padding: 10px 16px;
        color: #172033 !important;
        text-decoration: none !important;
        font-size: 14px;
        font-weight: 600;
    }
    .language-menu-content a:hover {
        background: #eef4ff;
        color: #2758e8 !important;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 56% 44%;
        min-height: 560px;
        overflow: hidden;
        border-radius: 0;
        box-shadow: 0 18px 44px rgba(15, 64, 128, 0.10);
    }
    .hero-left {
        color: white;
        background:
            linear-gradient(120deg, rgba(57, 118, 230, 0.94) 0%, rgba(87, 92, 226, 0.92) 100%),
            radial-gradient(circle at 0% 70%, rgba(255,255,255,0.18), transparent 28%);
        padding: 76px 78px 70px 78px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .hero-right {
        min-height: 560px;
        background-size: cover;
        background-position: 88% center;
        position: relative;
    }
    .hero-right::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, rgba(57,118,230,0.22), rgba(0,31,93,0.08));
    }
    .hero-title {
        max-width: 980px;
        font-size: 38px;
        line-height: 1.15;
        font-weight: 760;
        margin-bottom: 38px;
    }
    .hero-divider {
        width: 88%;
        height: 1px;
        background: rgba(255,255,255,0.70);
        margin-bottom: 42px;
    }
    .hero-copy {
        max-width: 900px;
        font-size: 15px;
        line-height: 1.72;
        letter-spacing: 0;
        margin-bottom: 24px;
        color: rgba(255,255,255,0.95);
    }
    .enter-wrap {
        display: flex;
        justify-content: center;
        padding-top: 34px;
        padding-bottom: 20px;
    }
    div[data-testid="stButton"] > button {
        border-radius: 999px;
        border: 3px solid #4b63f0;
        min-height: 56px;
        font-weight: 800;
        background: white;
        box-shadow: 0 12px 28px rgba(40, 74, 190, 0.13);
        font-size: 21px;
    }
    .small-muted {
        color: #5a6575;
        font-size: 14px;
    }
    .feature-label {
        display: inline-block;
        color: #303744;
        font-size: 14px;
        font-weight: 650;
        border-bottom: 1px dotted #7c8798;
        cursor: help;
    }
    .feature-label-wrap {
        position: relative;
        display: inline-block;
        margin-bottom: 4px;
        z-index: 20;
    }
    .feature-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        left: 0;
        top: calc(100% + 6px);
        width: max-content;
        max-width: 310px;
        padding: 8px 10px;
        border-radius: 6px;
        background: #172033;
        color: #ffffff;
        font-size: 12px;
        line-height: 1.45;
        font-weight: 500;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
        pointer-events: none;
        transition: opacity 0.12s ease;
        white-space: normal;
        z-index: 10000;
    }
    .feature-label-wrap:hover .feature-tooltip {
        visibility: visible;
        opacity: 1;
    }
    .risk-probability-card {
        padding: 8px 0 4px 0;
    }
    .risk-probability-label {
        font-size: 14px;
        line-height: 1.35;
        color: #303744;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .risk-probability-value {
        font-size: 32px;
        line-height: 1.1;
        font-weight: 850;
    }
    .risk-probability-value.low {
        color: #1f9d55;
    }
    .risk-probability-value.high {
        color: #d64545;
    }
    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .home-fullbleed {
            margin-left: -1rem;
            margin-right: -1rem;
        }
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .hero-right {
            min-height: 280px;
        }
        .hero-left {
            padding: 48px 34px;
        }
        .hero-title {
            font-size: 28px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "entered_prediction_tool" not in st.session_state:
    st.session_state.entered_prediction_tool = False

lang_param = st.query_params.get("lang", "en")
language = "中文" if lang_param in ["zh", "中文"] else "English"
page_param = st.query_params.get("page", "home")
if page_param == "tool":
    st.session_state.entered_prediction_tool = True

if st.session_state.entered_prediction_tool:
    topbar_left, topbar_right = st.columns([5.8, 1.2])
    with topbar_left:
        if st.button("← Home"):
            st.session_state.entered_prediction_tool = False
            st.query_params["page"] = "home"
            st.rerun()
    with topbar_right:
        st.markdown(
            """
            <div class="language-menu prediction-language-menu">
                <span>Language</span>
                <div class="language-menu-content">
                    <a href="?page=tool&lang=en" target="_self">English</a>
                    <a href="?page=tool&lang=zh" target="_self">中文</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

enable_llm = True
llm_base_url = "http://127.0.0.1:1234/v1"
llm_model_name = "qwen2.5-7b-instruct"

if not st.session_state.entered_prediction_tool:
    hero_image_path = Path(__file__).parent / "assets" / "ckd_pregnancy_hero.png"
    hero_image_base64 = image_to_base64(hero_image_path) if hero_image_path.exists() else ""
    st.markdown(
        f"""
        <div class="home-fullbleed">
            <div class="home-nav">
                <div class="language-menu">
                    <span>Language</span>
                    <div class="language-menu-content">
                        <a href="?page=home&lang=en" target="_self">English</a>
                        <a href="?page=home&lang=zh" target="_self">中文</a>
                    </div>
                </div>
            </div>
            <div class="hero-grid">
                <div class="hero-left">
                    <div class="hero-title">{TEXT[language]["home_title"]}</div>
                    <div class="hero-divider"></div>
                    <div class="hero-copy">{TEXT[language]["home_p1"]}</div>
                    <div class="hero-copy">{TEXT[language]["home_p2"]}</div>
                </div>
                <div class="hero-right" style="background-image: url('data:image/png;base64,{hero_image_base64}');"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="enter-wrap">', unsafe_allow_html=True)
    _, enter_col, _ = st.columns([1.85, 0.7, 1.85])
    with enter_col:
        if st.button(TEXT[language]["enter"], width="stretch"):
            st.session_state.entered_prediction_tool = True
            st.query_params["page"] = "tool"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

selected_model = None

# Load resources
try:
    with st.spinner("Loading model and resources..."):
        model, scaler, feature_names, model_name = load_artifacts(selected_model)
        default_input_values = load_default_input_values(feature_names)
    
except Exception as e:
    st.error(f"❌ Failed to load resources: {e}")
    st.stop()

st.title(TEXT[language]["page_title"])
with st.expander(TEXT[language]["help"], expanded=False):
    st.markdown(TEXT[language]["disclaimer"])
    st.markdown(f"**Input features ({len(feature_names)}):** {', '.join(feature_names)}")
    st.markdown(f"**Final model:** {model_name} ({type(model).__name__})")

# --- 3. Create tabs ---
tab1, tab2 = st.tabs([f"📝 {TEXT[language]['single_tab']}", f"📂 {TEXT[language]['batch_tab']}"])

# ==========================================
# Mode 1: Single Prediction (Manual Input)
# ==========================================
with tab1:
    st.info(TEXT[language]["input_note"])

    with st.form("single_predict_form"):
        inputs = render_grouped_inputs(feature_names, language, default_input_values)
        submitted = st.form_submit_button(f"🚀 {TEXT[language]['predict']}")

    if submitted:
        # Data assembly
        x_df = pd.DataFrame([inputs], columns=feature_names)

        try:
            x_scaled = scaler.transform(x_df)

            # Prediction
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(x_scaled)[0, 1]
            else:
                prob = model.predict(x_scaled)[0]

            result_box = st.container(border=True)
            with result_box:
                st.markdown(f"**{TEXT[language]['prediction_result']}**")
                result_col1, result_col2, result_col3 = st.columns([1, 1, 2])
                with result_col1:
                    risk_class = "high" if prob > RISK_THRESHOLD else "low"
                    st.markdown(
                        (
                            '<div class="risk-probability-card">'
                            f'<div class="risk-probability-label">{TEXT[language]["risk_probability"]}</div>'
                            f'<div class="risk-probability-value {risk_class}">{prob * 100:.2f}%</div>'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )
                with result_col2:
                    st.markdown(f"**{TEXT[language]['risk_level']}**")
                    if prob > RISK_THRESHOLD:
                        st.error(f"🔴 {TEXT[language]['high_risk']}")
                    else:
                        st.success(f"🟢 {TEXT[language]['low_risk']}")
                with result_col3:
                    st.markdown(f"**{TEXT[language]['model']}**")
                    st.info(f"{model_name} ({type(model).__name__})")

            contribution_df = None

            # SHAP explanation
            with st.spinner("Calculating feature contributions..."):
                # Smart Explainer selection
                explainer, explainer_type = get_explainer(model, x_scaled)
                
                # Calculate SHAP values
                shap_values, base_value = compute_shap_values(explainer, explainer_type, x_scaled)
                
                # Ensure shap_values is 1D
                if len(shap_values.shape) == 2:
                    sv = shap_values[0]
                else:
                    sv = shap_values

                contribution_df = build_contribution_table(
                    sv,
                    x_df.iloc[0],
                    feature_names,
                )

            st.divider()
            title_col1, title_col2 = st.columns(2)
            with title_col1:
                st.subheader(TEXT[language]["top_features"])
            with title_col2:
                st.subheader(TEXT[language]["waterfall"])
            fig_explanation = plot_combined_explanation(contribution_df, base_value)
            st_matplotlib_high_res(fig_explanation)
            plt.close(fig_explanation)

            if enable_llm:
                st.subheader(TEXT[language]["llm_explanation"])
                if contribution_df is None:
                    st.info("Feature contribution results are unavailable, so the LLM explanation was not generated.")
                else:
                    with st.spinner("Generating local LLM explanation..."):
                        try:
                            explanation_text = generate_local_llm_explanation(
                                prob,
                                contribution_df,
                                llm_base_url,
                                llm_model_name,
                                language,
                            )
                            st.markdown(explanation_text)
                        except Exception as llm_error:
                            st.warning(
                                "Local LLM explanation is unavailable. "
                                "Please confirm that LM Studio local server is running and the model name is correct."
                            )
                            st.caption(str(llm_error))
            else:
                st.info(TEXT[language]["llm_disabled"])

        except Exception as e:
            st.error(f"Error during execution: {e}")
            import traceback
            st.text(traceback.format_exc())

# ==========================================
# Mode 2: Batch Prediction (Upload Excel)
# ==========================================
with tab2:
    st.info("Suitable for processing multiple data entries. Please upload an Excel (.xlsx) or CSV file.")

    # 1. Download template
    with st.expander("📥 Download Data Template"):
        st.write("Please ensure your table contains the following column names:")
        st.caption("Binary variables should be coded as 0 = No and 1 = Yes.")
        st.code(str(feature_names), language="python")
        template_df = pd.DataFrame(columns=['Patient_ID'] + feature_names)
        csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Template", csv, "prediction_template.csv", "text/csv")

    # 2. File upload
    uploaded_file = st.file_uploader("Upload File", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_upload = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_upload = pd.read_csv(uploaded_file, encoding='gbk')
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write(f"Successfully read {len(df_upload)} data entries.")

            df_upload.columns = df_upload.columns.str.strip()
            missing_cols = [col for col in feature_names if col not in df_upload.columns]

            if missing_cols:
                st.error(f"The file is missing the following required feature columns:\n{missing_cols}")
            else:
                X_batch = df_upload[feature_names]
                X_batch_scaled = scaler.transform(X_batch)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_batch_scaled)[:, 1]
                else:
                    probs = model.predict(X_batch_scaled)

                df_result = df_upload.copy()
                df_result['Predicted_Probability'] = np.round(probs, 4)
                df_result['Risk_Level'] = [
                    'High Risk' if p > RISK_THRESHOLD else 'Low Risk'
                    for p in probs
                ]

                st.subheader("📊 Prediction Results Overview")
                st.dataframe(df_result.style.applymap(
                    lambda x: 'background-color: #ffcccc' if x == 'High Risk' else 'background-color: #ccffcc',
                    subset=['Risk_Level']
                ))

                csv_result = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button("💾 Download Prediction Results (.csv)", csv_result, "prediction_results.csv", "text/csv")

                # Deep dive analysis
                st.divider()
                st.subheader("🔍 In-depth Analysis: View SHAP Explanation for Specific Sample")
                selected_index = st.selectbox(
                    "Select Row Number (Index) for Analysis",
                    options=df_result.index,
                    format_func=lambda x: f"Row {x} (Probability: {df_result.loc[x, 'Predicted_Probability']:.2%})"
                )

                if st.button("Explain This Sample"):
                    x_single_df = X_batch.iloc[[selected_index]]
                    x_single_scaled = X_batch_scaled[selected_index].reshape(1, -1)

                    # Smart Explainer selection
                    explainer, explainer_type = get_explainer(model, X_batch_scaled)
                    
                    # Calculate SHAP values
                    shap_values, base_value = compute_shap_values(explainer, explainer_type, x_single_scaled)

                    # Ensure shap_values is 1D
                    if len(shap_values.shape) == 2:
                        sv = shap_values[0]
                    else:
                        sv = shap_values

                    # 1. Waterfall Plot
                    st.markdown("**1. Waterfall Plot**")
                    exp = shap.Explanation(
                        values=sv, base_values=base_value,
                        data=x_single_df.iloc[0], feature_names=feature_names
                    )
                    fig_batch = plt.figure(figsize=(6.5, 3.6), dpi=120)
                    shap.plots.waterfall(exp, max_display=8, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_batch, bbox_inches='tight', width="content")
                    plt.close(fig_batch)

                    # 2. Force Plot
                    if explainer_type in ['tree', 'kernel']:
                        with st.expander("2. Interactive Force Plot", expanded=False):
                            try:
                                force_plot_html_batch = shap.force_plot(
                                    base_value,
                                    sv,
                                    x_single_df.iloc[0],
                                    feature_names=feature_names,
                                    matplotlib=False
                                )
                                st_shap(force_plot_html_batch, height=130)
                            except:
                                st.info("Interactive force plot not available for this model type.")
                    else:
                        st.info("Force plot is only available for tree-based models.")

        except Exception as e:
            st.error(f"Error while processing file: {e}")
            import traceback
            st.text(traceback.format_exc())
