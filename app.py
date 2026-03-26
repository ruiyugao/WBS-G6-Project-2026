import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from data_processor import load_and_process_data, predict_future, scenario_analysis, calculate_impact

# =============================================================================
# 🌐 双语翻译系统 / Bilingual Translation System
# =============================================================================

translations = {
    'zh': {
        # 页面配置 / Page Config
        'page_title': '垃圾预言家 - 中国未来垃圾量预测',
        'sidebar_title': '🧭 导航面板',
        'select_page': '选择展示页面:',
        'settings': '⚙️ 参数设置',
        'target_year': '预测目标年份',
        'recycling_rate': '基准资源化利用率',
        'data_loaded': '✅ 数据加载成功！',
        'data_years': '数据年份:',
        'data_rows': '数据行数:',
        'load_error': '❌ 数据加载失败:',
        
        # 页面选项 / Page Options
        'page_home': '🏠 首页',
        'page_explore': '📊 数据探索',
        'page_lab': '🔮 预言家实验室',
        'page_challenge': '🎮 减量挑战',
        'page_report': '📈 预测报告',
        
        # 首页 / Home Page
        'main_title': '🗑️ 垃圾预言家',
        'subtitle': '基于人口数据的中国未来垃圾量预测',
        'latest_year': '📅 最新数据年份',
        'total_pop': '👥 总人口',
        'billion': '亿',
        'annual_waste': '🗑️ 年垃圾清运量',
        'per_capita': '📦 人均日垃圾量',
        'kg': 'kg',
        'project_bg': '📚 项目背景',
        'why_care': '**为什么要关注垃圾问题？**',
        'intro1': '随着中国经济的发展和城市化进程的加快，城市生活垃圾产生量持续增长。垃圾处理已经成为影响城市可持续发展的重要挑战。',
        'intro2': '本项目通过分析历史人口数据和垃圾清运数据，建立数学预测模型，帮助我们预测未来的垃圾产生量，从而更好地制定环保政策。',
        'national_plan': '**国家"十四五"规划目标：**',
        'goal1': '- 到2025年底，全国城市生活垃圾资源化利用率达到60%左右',
        'goal2': '- 到2025年底，基本实现垃圾分类全覆盖',
        'quick_trend': '📈 趋势快速预览',
        'pop_trend': '人口变化趋势',
        'waste_trend': '垃圾清运量变化趋势',
        'total_pop_100m': '总人口(亿人)',
        'total_waste_100m': '垃圾清运量(亿吨)',
        'chart_error': '图表渲染错误:',
        'data_preview': '数据预览:',
        
        # 数据探索 / Data Exploration
        'explore_title': '📊 数据探索中心',
        'tab_pop': '👥 人口数据',
        'tab_waste': '🗑️ 垃圾数据',
        'tab_corr': '🔗 相关性分析',
        'pop_analysis': '人口变化趋势分析',
        'pop_trend_title': '中国人口变化趋势',
        'pop_unit': '人口数(万人)',
        'metric': '指标',
        'urbanization_trend': '城镇化率变化趋势',
        'waste_analysis': '垃圾清运与处理分析',
        'annual_waste_title': '年生活垃圾清运量',
        'treatment_capacity': '垃圾处理能力变化',
        'treatment_unit': '处理能力(万吨/日)',
        'treatment_method': '处理方式',
        'treatment_plants': '无害化处理厂数量变化',
        'raw_data': '查看原始数据',
        'corr_analysis': '人口与垃圾的相关性分析',
        'corr_title': '总人口与垃圾清运量的关系',
        'correlation': '📊 总人口与垃圾清运量的相关系数:',
        'corr_tip': '💡 相关系数接近1表示两者高度正相关，说明人口增长确实是影响垃圾量的重要因素！',
        
        # 预言家实验室 / Prediction Lab
        'lab_title': '🔮 预言家实验室',
        'lab_subtitle': '在这里建立你的预测模型，成为垃圾预言家！',
        'params_adjust': '🎛️ 模型参数调节',
        'pop_adjust': '人口增长调整系数',
        'pop_adjust_help': '正值表示人口增长更快，负值表示人口减少更快',
        'waste_adjust': '人均垃圾量增长调整系数',
        'waste_adjust_help': '正值表示人均垃圾产生量增加，负值表示减少',
        'prediction_result': '📉 预测结果',
        'historical_data': '历史数据',
        'predicted_data': '预测数据',
        'prediction_title': '中国垃圾总量预测（至{}年）',
        'prediction_start': '预测起点',
        'pred_total_pop': '📅 {}年预测总人口',
        'pred_per_capita': '📦 {}年预测人均日垃圾量',
        'pred_total_waste': '🗑️ {}年预测垃圾总量',
        'pred_increase': '📈 较{}年增长',
        'view_full_data': '📋 查看完整预测数据表',
        'year': '年份',
        'adj_total_pop': '调整后总人口(万人)',
        'adj_per_capita': '调整后人均垃圾量(公斤/日)',
        'adj_total_waste': '调整后垃圾总量(亿吨)',
        'pred_error': '预测错误:',
        'waste_total_100m': '垃圾总量(亿吨)',
        'data_type': '类型',
        
        # 减量挑战 / Reduction Challenge
        'challenge_title': '🎮 垃圾减量大挑战',
        'challenge_subtitle': '看看我们能为地球做些什么！',
        'national_scenario': '🌍 国家级情景模拟',
        'reduction_rate': '垃圾减量率',
        'reduction_help': '通过减少浪费、源头减量等方式减少的垃圾比例',
        'recycling_help': '通过回收、堆肥等方式资源化利用的垃圾比例',
        'target_year_text': '🎯 目标年份: {}年',
        'baseline_pred': '基准预测垃圾总量:',
        'after_reduction': '减量后垃圾总量:',
        'recycled_amount': '可回收利用量:',
        'to_fillburn': '需要填埋/焚烧量:',
        'total_saved': '🌱 总共减少填埋/焚烧:',
        'reduction_effect': '垃圾减量效果对比',
        'calc_error': '计算错误:',
        'school_challenge': '🏫 我的校园挑战',
        'school_challenge_tip': '计算一下你们学校的垃圾减量潜力！',
        'student_count': '学校学生人数',
        'waste_per_student': '每人每日垃圾量(kg)',
        'school_goal': '校园垃圾减量率目标',
        'school_result_title': '🎉 你们学校每年可以:',
        'reduce_waste': '减少垃圾:',
        'ton': '吨',
        'reduce_co2': '减少CO₂排放:',
        'trees_planted': '相当于种树:',
        'leaderboard': '🏆 减量英雄榜',
        'check_actions': '勾选你能做到的减量化行动：',
        'challenge_score': '🌟 你的减量化贡献总分',
        'challenge_encourage': '继续努力，你是地球的守护者！',
        
        # 预测报告 / Prediction Report
        'report_title': '📈 我的预测报告',
        'report_summary': '📋 报告摘要',
        'item': '项目',
        'value': '数值',
        'pred_year': '预测年份',
        'base_year': '基准年份',
        'pred_total_pop_report': '预测总人口',
        'pred_per_capita_report': '预测人均日垃圾量',
        'pred_total_waste_report': '预测年垃圾总量',
        'compare_base': '相较于{}年增长',
        'key_findings': '🔍 主要发现',
        'finding1': '1. 人口增长与垃圾产生量呈显著正相关关系',
        'finding2': '2. 如果保持当前趋势，到{}年中国垃圾年产量将达到{}亿吨',
        'finding3': '3. 通过实施有效的垃圾减量政策，可以显著降低未来的垃圾处理压力',
        'finding4': '4. 提高资源化利用率是减少填埋和焚烧的关键',
        'policy_suggestions': '💡 政策建议',
        'suggestion1': '1. 加强宣传教育，提高全民环保意识',
        'suggestion2': '2. 完善垃圾分类体系，提高资源化利用率',
        'suggestion3': '3. 推广清洁生产，从源头减少垃圾产生',
        'suggestion4': '4. 加大环保投入，提升垃圾处理能力',
        'suggestion5': '5. 鼓励绿色生活方式，倡导低碳消费',
        'report_generator': '📄 报告生成器',
        'student_name': '学生姓名',
        'school_name': '学校名称',
        'class_name': '班级',
        'generate_report': '生成我的预测报告',
        'report_header': '# 中国未来垃圾量预测报告',
        'report_by': '报告作者:',
        'report_school': '学校:',
        'report_class': '班级:',
        'report_date': '日期:',
    },
    'en': {
        # Page Config
        'page_title': 'Garbage Prophet - China Waste Prediction',
        'sidebar_title': '🧭 Navigation',
        'select_page': 'Select Page:',
        'settings': '⚙️ Settings',
        'target_year': 'Target Prediction Year',
        'recycling_rate': 'Base Recycling Rate',
        'data_loaded': '✅ Data Loaded Successfully!',
        'data_years': 'Data Years:',
        'data_rows': 'Data Rows:',
        'load_error': '❌ Data Loading Error:',
        
        # Page Options
        'page_home': '🏠 Home',
        'page_explore': '📊 Explore Data',
        'page_lab': '🔮 Prediction Lab',
        'page_challenge': '🎮 Challenge',
        'page_report': '📈 Report',
        
        # Home Page
        'main_title': '🗑️ Garbage Prophet',
        'subtitle': 'Predicting China\'s Future Waste Volume Based on Population Data',
        'latest_year': '📅 Latest Data Year',
        'total_pop': '👥 Total Population',
        'billion': 'B',
        'annual_waste': '🗑️ Annual Waste',
        'per_capita': '📦 Per Capita Daily',
        'kg': 'kg',
        'project_bg': '📚 Project Background',
        'why_care': '**Why Care About Waste Issues?**',
        'intro1': 'With China\'s economic development and urbanization, municipal solid waste generation continues to grow. Waste management has become a critical challenge for urban sustainability.',
        'intro2': 'This project analyzes historical population and waste data to build a mathematical prediction model, helping us forecast future waste generation and develop better environmental policies.',
        'national_plan': '**National 14th Five-Year Plan Goals:**',
        'goal1': '- By 2025, achieve 60% recycling rate for municipal solid waste',
        'goal2': '- By 2025, achieve full coverage of waste sorting',
        'quick_trend': '📈 Trend Preview',
        'pop_trend': 'Population Trend',
        'waste_trend': 'Waste Trend',
        'total_pop_100m': 'Total Population (100M)',
        'total_waste_100m': 'Total Waste (100M Tons)',
        'chart_error': 'Chart Rendering Error:',
        'data_preview': 'Data Preview:',
        
        # Data Exploration
        'explore_title': '📊 Data Exploration Center',
        'tab_pop': '👥 Population Data',
        'tab_waste': '🗑️ Waste Data',
        'tab_corr': '🔗 Correlation Analysis',
        'pop_analysis': 'Population Trend Analysis',
        'pop_trend_title': 'China Population Trends',
        'pop_unit': 'Population (10,000)',
        'metric': 'Metric',
        'urbanization_trend': 'Urbanization Rate Trend',
        'waste_analysis': 'Waste Collection & Treatment Analysis',
        'annual_waste_title': 'Annual Municipal Waste Collection',
        'treatment_capacity': 'Waste Treatment Capacity',
        'treatment_unit': 'Capacity (10,000 Tons/Day)',
        'treatment_method': 'Treatment Method',
        'treatment_plants': 'Number of Harmless Treatment Plants',
        'raw_data': 'View Raw Data',
        'corr_analysis': 'Population-Waste Correlation Analysis',
        'corr_title': 'Population vs Waste Correlation',
        'correlation': '📊 Correlation Coefficient:',
        'corr_tip': '💡 A coefficient close to 1 indicates strong positive correlation - population growth significantly affects waste volume!',
        
        # Prediction Lab
        'lab_title': '🔮 Prediction Lab',
        'lab_subtitle': 'Build your prediction model and become a Garbage Prophet!',
        'params_adjust': '🎛️ Parameter Adjustment',
        'pop_adjust': 'Population Growth Adjustment',
        'pop_adjust_help': 'Positive = faster growth, Negative = slower growth/decline',
        'waste_adjust': 'Waste Per Capita Adjustment',
        'waste_adjust_help': 'Positive = more waste per person, Negative = less waste',
        'prediction_result': '📉 Prediction Results',
        'historical_data': 'Historical Data',
        'predicted_data': 'Predicted Data',
        'prediction_title': 'China Waste Prediction (to {})',
        'prediction_start': 'Prediction Start',
        'pred_total_pop': '📅 {} Predicted Population',
        'pred_per_capita': '📦 {} Predicted Waste Per Capita',
        'pred_total_waste': '🗑️ {} Predicted Total Waste',
        'pred_increase': '📈 Increase from {}',
        'view_full_data': '📋 View Full Prediction Data',
        'year': 'Year',
        'adj_total_pop': 'Adjusted Population (10,000)',
        'adj_per_capita': 'Adjusted Waste Per Capita (kg/day)',
        'adj_total_waste': 'Adjusted Total Waste (100M Tons)',
        'pred_error': 'Prediction Error:',
        'waste_total_100m': 'Total Waste (100M Tons)',
        'data_type': 'Type',
        
        # Reduction Challenge
        'challenge_title': '🎮 Waste Reduction Challenge',
        'challenge_subtitle': 'See what we can do for our planet!',
        'national_scenario': '🌍 National Scenario Simulation',
        'reduction_rate': 'Waste Reduction Rate',
        'reduction_help': 'Waste reduction through source reduction and waste minimization',
        'recycling_help': 'Waste recycled through recycling and composting',
        'target_year_text': '🎯 Target Year: {}',
        'baseline_pred': 'Baseline Prediction:',
        'after_reduction': 'After Reduction:',
        'recycled_amount': 'Recyclable Amount:',
        'to_fillburn': 'To Landfill/Incineration:',
        'total_saved': '🌱 Total Reduction:',
        'reduction_effect': 'Waste Reduction Effect Comparison',
        'calc_error': 'Calculation Error:',
        'school_challenge': '🏫 My School Challenge',
        'school_challenge_tip': 'Calculate your school\'s waste reduction potential!',
        'student_count': 'Number of Students',
        'waste_per_student': 'Waste Per Student (kg/day)',
        'school_goal': 'School Reduction Goal',
        'school_result_title': '🎉 Your school can achieve annually:',
        'reduce_waste': 'Waste Reduced:',
        'ton': 'tons',
        'reduce_co2': 'CO₂ Reduced:',
        'trees_planted': 'Equivalent to planting:',
        'leaderboard': '🏆 Reduction Leaderboard',
        'check_actions': 'Check the actions you can commit to:',
        'challenge_score': '🌟 Your Reduction Score',
        'challenge_encourage': 'Keep up the great work - you\'re a planet protector!',
        
        # Prediction Report
        'report_title': '📈 My Prediction Report',
        'report_summary': '📋 Report Summary',
        'item': 'Item',
        'value': 'Value',
        'pred_year': 'Prediction Year',
        'base_year': 'Base Year',
        'pred_total_pop_report': 'Predicted Population',
        'pred_per_capita_report': 'Predicted Waste Per Capita',
        'pred_total_waste_report': 'Predicted Annual Waste',
        'compare_base': 'Increase from {}',
        'key_findings': '🔍 Key Findings',
        'finding1': '1. Population growth and waste generation show significant positive correlation',
        'finding2': '2. At current trends, China\'s annual waste will reach {} 100M tons by {}',
        'finding3': '3. Effective waste reduction policies can significantly reduce future waste management pressure',
        'finding4': '4. Improving recycling rate is key to reducing landfill and incineration',
        'policy_suggestions': '💡 Policy Recommendations',
        'suggestion1': '1. Strengthen environmental education and awareness',
        'suggestion2': '2. Improve waste sorting systems and recycling rates',
        'suggestion3': '3. Promote cleaner production to reduce waste at source',
        'suggestion4': '4. Increase environmental investment and waste treatment capacity',
        'suggestion5': '5. Encourage green lifestyles and low-carbon consumption',
        'report_generator': '📄 Report Generator',
        'student_name': 'Student Name',
        'school_name': 'School Name',
        'class_name': 'Class',
        'generate_report': 'Generate My Report',
        'report_header': '# China Future Waste Prediction Report',
        'report_by': 'Report By:',
        'report_school': 'School:',
        'report_class': 'Class:',
        'report_date': 'Date:',
    }
}

# 挑战项翻译 / Challenge Items Translation
challenge_items = {
    'zh': [
        ("自带水杯", "减少一次性塑料瓶", 0.02),
        ("光盘行动", "减少厨余垃圾", 0.05),
        ("双面打印", "减少纸张浪费", 0.01),
        ("旧物回收", "减少固体废弃物", 0.03),
        ("布袋购物", "减少塑料袋使用", 0.01),
    ],
    'en': [
        ("Bring Your Own Bottle", "Reduce single-use plastic bottles", 0.02),
        ("Clean Plate Campaign", "Reduce food waste", 0.05),
        ("Double-sided Printing", "Reduce paper waste", 0.01),
        ("Recycle Used Items", "Reduce solid waste", 0.03),
        ("Cloth Shopping Bags", "Reduce plastic bags", 0.01),
    ]
}

# =============================================================================
# 🎨 页面配置与样式 / Page Configuration & Styling
# =============================================================================

st.set_page_config(
    page_title="Garbage Prophet | 垃圾预言家",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 / Custom CSS Styles
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem !important;
        background: linear-gradient(45deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .metric-card h3 {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
        font-weight: 500;
    }
    .metric-card p {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    .section-header {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .lang-selector {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🌍 语言选择与数据加载 / Language Selection & Data Loading
# =============================================================================

# 语言选择器 - 放在侧边栏最顶部
with st.sidebar:
    st.markdown("### 🌐 Language | 语言")
    lang = st.selectbox(
        "Select Language | 选择语言",
        options=["中文", "English"],
        index=0,
        label_visibility="collapsed"
    )
    lang_code = 'zh' if lang == "中文" else 'en'
    t = translations[lang_code]
    challenges = challenge_items[lang_code]
    
    st.markdown("---")

# 加载数据
try:
    merged_data, pop_data, waste_data = load_and_process_data()
    data_loaded = True
    with st.sidebar:
        st.success(t['data_loaded'])
        st.write(f"{t['data_years']} {merged_data['年份'].min()}-{merged_data['年份'].max()}")
        st.write(f"{t['data_rows']} {len(merged_data)}")
except Exception as e:
    data_loaded = False
    st.error(f"{t['load_error']} {e}")
    st.stop()

# =============================================================================
# 🧭 侧边栏导航 / Sidebar Navigation
# =============================================================================

with st.sidebar:
    st.title(t['sidebar_title'])
    page = st.radio(
        t['select_page'],
        [t['page_home'], t['page_explore'], t['page_lab'], t['page_challenge'], t['page_report']]
    )
    
    st.markdown("---")
    st.header(t['settings'])
    target_year = st.slider(t['target_year'], 2026, 2040, 2035)
    base_recycling_rate = st.slider(t['recycling_rate'], 0.3, 0.8, 0.6, 
                                    help="国家十四五规划目标: 60%" if lang_code == 'zh' else "National Goal: 60%")

# =============================================================================
# 📄 页面渲染 / Page Rendering
# =============================================================================

# 🏠 首页 / Home Page
if page == t['page_home']:
    st.markdown(f"<h1 class='main-title'>{t['main_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='sub-title'>{t['subtitle']}</p>", unsafe_allow_html=True)
    
    # 最新数据概览卡片 / Key Metrics Cards
    latest = merged_data.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{t['latest_year']}</h3>
            <p>{int(latest['年份'])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
            <h3>{t['total_pop']}</h3>
            <p>{latest['总人口(万人)']/10000:.2f}{t['billion']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);'>
            <h3>{t['annual_waste']}</h3>
            <p>{latest['生活垃圾清运量(万吨)']/10000:.2f}{t['billion']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #45b7d1 0%, #96c93d 100%);'>
            <h3>{t['per_capita']}</h3>
            <p>{latest['人均垃圾产生量(公斤/日)']:.2f}{t['kg']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 项目介绍 / Project Introduction
    st.header(t['project_bg'])
    st.markdown(f"""
    <div class='info-box'>
        <strong>{t['why_care']}</strong><br><br>
        {t['intro1']}<br><br>
        {t['intro2']}<br><br>
        <strong>{t['national_plan']}</strong><br>
        {t['goal1']}<br>
        {t['goal2']}
    </div>
    """, unsafe_allow_html=True)
    
    # 快速预览图表 / Quick Trend Charts
    st.header(t['quick_trend'])
    try:
        fig = make_subplots(rows=1, cols=2, subplot_titles=(t['pop_trend'], t['waste_trend']))
        
        fig.add_trace(
            go.Scatter(x=merged_data['年份'], y=merged_data['总人口(万人)']/10000, 
                       name=t['total_pop_100m'], line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=merged_data['年份'], y=merged_data['生活垃圾清运量(万吨)']/10000, 
                       name=t['total_waste_100m'], line=dict(color='#ff7f0e', width=3)),
            row=1, col=2
        )
        
        fig.update_layout(height=450, showlegend=True, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"{t['chart_error']} {e}")
        st.write(t['data_preview'])
        st.dataframe(merged_data[['年份', '总人口(万人)', '生活垃圾清运量(万吨)']].head())

# 📊 数据探索页面 / Data Exploration Page
elif page == t['page_explore']:
    st.markdown(f"<h1 class='main-title'>{t['explore_title']}</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([t['tab_pop'], t['tab_waste'], t['tab_corr']])
    
    with tab1:
        st.subheader(t['pop_analysis'])
        
        try:
            fig = px.line(pop_data, x='年份', y=['总人口(万人)', '城镇人口(万人)', '乡村人口(万人)'],
                          title=t['pop_trend_title'],
                          labels={'value': t['pop_unit'], 'variable': t['metric']},
                          template='plotly_white')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # 城镇化率 / Urbanization Rate
            fig2 = px.line(pop_data, x='年份', y='城镇化率(%)',
                           title=t['urbanization_trend'],
                           color_discrete_sequence=['#2ca02c'],
                           template='plotly_white')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"{t['chart_error']} {e}")
            st.dataframe(pop_data)
    
    with tab2:
        st.subheader(t['waste_analysis'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                fig = px.bar(waste_data, x='年份', y='生活垃圾清运量(万吨)',
                             title=t['annual_waste_title'],
                             color='生活垃圾清运量(万吨)',
                             color_continuous_scale='Reds',
                             template='plotly_white')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"{t['chart_error']} {e}")
        
        with col2:
            try:
                fig2 = px.line(waste_data, x='年份', y=['卫生填埋处理能力(万吨/日)', '焚烧处理能力(万吨/日)'],
                               title=t['treatment_capacity'],
                               labels={'value': t['treatment_unit'], 'variable': t['treatment_method']},
                               template='plotly_white')
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"{t['chart_error']} {e}")
        
        # 处理厂数量 / Treatment Plants
        try:
            fig3 = px.bar(waste_data, x='年份', y='无害化处理厂数(座)',
                          title=t['treatment_plants'],
                          color_discrete_sequence=['#9467bd'],
                          template='plotly_white')
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"{t['chart_error']} {e}")
            
        # 显示原始数据 / Raw Data
        with st.expander(t['raw_data']):
            st.dataframe(waste_data, use_container_width=True)
    
    with tab3:
        st.subheader(t['corr_analysis'])
        
        try:
            # 散点图 / Scatter Plot
            fig = px.scatter(merged_data, x='总人口(万人)', y='生活垃圾清运量(万吨)',
                             title=t['corr_title'],
                             trendline='ols',
                             color='年份',
                             size='人均垃圾产生量(公斤/日)',
                             hover_data=['年份', '城镇化率(%)'],
                             template='plotly_white')
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
            
            # 计算相关系数 / Correlation Coefficient
            corr = merged_data['总人口(万人)'].corr(merged_data['生活垃圾清运量(万吨)'])
            st.markdown(f"""
            <div class='success-box'>
                <strong>{t['correlation']} {corr:.4f}</strong><br><br>
                {t['corr_tip']}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"{t['chart_error']} {e}")

# 🔮 预言家实验室 / Prediction Lab
elif page == t['page_lab']:
    st.markdown(f"<h1 class='main-title'>{t['lab_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='sub-title'>{t['lab_subtitle']}</p>", unsafe_allow_html=True)
    
    # 模型参数调节 / Parameter Adjustment
    st.header(t['params_adjust'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        pop_growth_adjust = st.slider(
            t['pop_adjust'],
            -2.0, 2.0, 0.0, 0.1,
            help=t['pop_adjust_help']
        )
    
    with col2:
        waste_growth_adjust = st.slider(
            t['waste_adjust'],
            -0.1, 0.1, 0.0, 0.01,
            help=t['waste_adjust_help']
        )
    
    # 获取预测 / Get Predictions
    try:
        predictions, model_pop, model_waste = predict_future(merged_data, target_year)
        
        # 应用用户调整 / Apply User Adjustments
        predictions['调整后总人口(万人)'] = predictions['预测总人口(万人)'] * (1 + pop_growth_adjust / 100)
        predictions['调整后人均垃圾量(公斤/日)'] = predictions['预测人均垃圾量(公斤/日)'] * (1 + waste_growth_adjust)
        predictions['调整后垃圾总量(亿吨)'] = (predictions['调整后总人口(万人)'] * 10000) * (predictions['调整后人均垃圾量(公斤/日)'] / 1000) * 365 / 1e8
        
        # 展示预测结果 / Show Results
        st.header(t['prediction_result'])
        
        # 合并历史数据和预测数据 / Combine Historical and Predicted Data
        historical = merged_data[['年份', '生活垃圾清运量(万吨)']].copy()
        historical[t['data_type']] = t['historical_data']
        historical[t['waste_total_100m']] = historical['生活垃圾清运量(万吨)'] / 10000
        
        pred_display = predictions[['年份', '调整后垃圾总量(亿吨)']].copy()
        pred_display[t['data_type']] = t['predicted_data']
        pred_display[t['waste_total_100m']] = pred_display['调整后垃圾总量(亿吨)']
        
        combined = pd.concat([
            historical[['年份', t['waste_total_100m'], t['data_type']]],
            pred_display[['年份', t['waste_total_100m'], t['data_type']]]
        ])
        
        # 绘制预测图 / Prediction Chart
        fig = px.line(combined, x='年份', y=t['waste_total_100m'], color=t['data_type'],
                      title=t['prediction_title'].format(target_year),
                      color_discrete_map={t['historical_data']: '#1f77b4', t['predicted_data']: '#ff7f0e'},
                      line_dash=t['data_type'],
                      line_dash_map={t['historical_data']: 'solid', t['predicted_data']: 'dash'},
                      template='plotly_white')
        
        fig.update_traces(line=dict(width=4))
        fig.add_vline(x=merged_data['年份'].max(), line_dash="dash", line_color="gray", 
                      annotation_text=t['prediction_start'], annotation_position="top right")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        
        # 关键预测数字展示 / Key Metrics Display
        final_pred = predictions.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label=t['pred_total_pop'].format(target_year),
                      value=f"{final_pred['调整后总人口(万人)']/10000:.2f}{t['billion']}",
                      delta=f"{pop_growth_adjust:+.1f}%")
        
        with col2:
            st.metric(label=t['pred_per_capita'].format(target_year),
                      value=f"{final_pred['调整后人均垃圾量(公斤/日)']:.2f}{t['kg']}",
                      delta=f"{waste_growth_adjust*100:+.1f}%")
        
        with col3:
            st.metric(label=t['pred_total_waste'].format(target_year),
                      value=f"{final_pred['调整后垃圾总量(亿吨)']:.2f}{t['billion']}{t['ton']}")
        
        with col4:
            increase = final_pred['调整后垃圾总量(亿吨)'] - merged_data.iloc[-1]['生活垃圾清运量(万吨)']/10000
            st.metric(label=t['pred_increase'].format(merged_data['年份'].max()),
                      value=f"{increase:.2f}{t['billion']}{t['ton']}",
                      delta=f"{increase/(merged_data.iloc[-1]['生活垃圾清运量(万吨)']/10000)*100:+.1f}%")
        
        # 预测数据表格 / Prediction Data Table
        with st.expander(t['view_full_data']):
            display_df = predictions.rename(columns={
                '年份': t['year'],
                '调整后总人口(万人)': t['adj_total_pop'],
                '调整后人均垃圾量(公斤/日)': t['adj_per_capita'],
                '调整后垃圾总量(亿吨)': t['adj_total_waste']
            })
            st.dataframe(display_df[[t['year'], t['adj_total_pop'], t['adj_per_capita'], t['adj_total_waste']]]
                         .style.format({
                             t['adj_total_pop']: '{:.0f}',
                             t['adj_per_capita']: '{:.2f}',
                             t['adj_total_waste']: '{:.2f}'
                         }), use_container_width=True)
    except Exception as e:
        st.error(f"{t['pred_error']} {e}")

# 🎮 减量挑战 / Reduction Challenge
elif page == t['page_challenge']:
    st.markdown(f"<h1 class='main-title'>{t['challenge_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='sub-title'>{t['challenge_subtitle']}</p>", unsafe_allow_html=True)
    
    # 情景分析 / Scenario Analysis
    st.header(t['national_scenario'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        reduction_rate = st.slider(
            t['reduction_rate'],
            0.0, 0.5, 0.1, 0.05,
            help=t['reduction_help']
        )
        recycling_rate = st.slider(
            t['recycling_rate'],
            0.3, 0.9, base_recycling_rate, 0.05,
            help=t['recycling_help']
        )
    
    # 获取基准预测 / Get Baseline Predictions
    try:
        predictions, _, _ = predict_future(merged_data, target_year)
        scenario_results = scenario_analysis(predictions, reduction_rate, recycling_rate)
        
        final = scenario_results.iloc[-1]
        baseline = predictions.iloc[-1]['预测垃圾总量(亿吨)']
        
        with col2:
            st.markdown(f"""
            <div style='background: #e8f5e8; padding: 20px; border-radius: 15px;'>
                <h3>{t['target_year_text'].format(target_year)}</h3>
                <p>{t['baseline_pred']} <b>{baseline:.2f}{t['billion']}{t['ton']}</b></p>
                <p>{t['after_reduction']} <b>{final['减量后垃圾总量(亿吨)']:.2f}{t['billion']}{t['ton']}</b></p>
                <p>{t['recycled_amount']} <b>{final['可回收利用量(亿吨)']:.2f}{t['billion']}{t['ton']}</b></p>
                <p>{t['to_fillburn']} <b>{final['需要填埋/焚烧量(亿吨)']:.2f}{t['billion']}{t['ton']}</b></p>
                <hr>
                <p style='color: green; font-weight: bold;'>
                    {t['total_saved']} {(baseline - final['需要填埋/焚烧量(亿吨)']):.2f}{t['billion']}{t['ton']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # 绘制情景比较图 / Scenario Comparison Chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Baseline' if lang_code == 'en' else '基准预测', 'After Reduction' if lang_code == 'en' else '减量后'],
            y=[baseline, final['减量后垃圾总量(亿吨)']],
            name='Total Waste' if lang_code == 'en' else '垃圾总量',
            marker_color=['#ff7f0e', '#2ca02c']
        ))
        
        fig.add_trace(go.Bar(
            x=['', ''],
            y=[0, final['可回收利用量(亿吨)']],
            name='Recyclable' if lang_code == 'en' else '可回收利用',
            marker_color=['#1f77b4', '#1f77b4']
        ))
        
        fig.update_layout(
            title=t['reduction_effect'],
            barmode='stack',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"{t['calc_error']} {e}")
    
    # 校园挑战 / School Challenge
    st.header(t['school_challenge'])
    st.markdown(f"""
    <div class='info-box'>
        {t['school_challenge_tip']}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        students = st.number_input(t['student_count'], 100, 10000, 1000)
    with col2:
        waste_per = st.number_input(t['waste_per_student'], 0.1, 2.0, 0.5)
    with col3:
        school_reduction = st.slider(t['school_goal'], 0.0, 0.8, 0.3)
    
    impact = calculate_impact(students, waste_per, school_reduction)
    
    st.markdown(f"""
    <div class='success-box'>
        <strong>{t['school_result_title']}</strong><br><br>
        • {t['reduce_waste']} {impact['年减量(吨)']} {t['ton']}<br>
        • {t['reduce_co2']} {impact['CO2减排(吨)']} {t['ton']}<br>
        • {t['trees_planted']} {int(impact['相当于种树(棵)'])} 🌳
    </div>
    """, unsafe_allow_html=True)
    
    # 挑战排行榜 / Challenge Leaderboard
    st.header(t['leaderboard'])
    
    total = 0
    st.write(t['check_actions'])
    for name, desc, saving in challenges:
        if st.checkbox(f"✅ {name} - {desc}", key=name):
            total += saving
            st.success(f"+ {saving*100:.0f}% {'contribution' if lang_code == 'en' else '减量化贡献'}")
    
    st.markdown(f"""
    <div style='background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); 
                padding: 25px; border-radius: 15px; color: white; text-align: center; margin-top: 20px;'>
        <h3>{t['challenge_score']}</h3>
        <p style='font-size: 3.5rem; font-weight: bold; margin: 10px 0;'>{total*100:.0f}%</p>
        <p style='font-size: 1.1rem; opacity: 0.95;'>{t['challenge_encourage']}</p>
    </div>
    """, unsafe_allow_html=True)

# 📈 预测报告 / Prediction Report
elif page == t['page_report']:
    st.markdown(f"<h1 class='main-title'>{t['report_title']}</h1>", unsafe_allow_html=True)
    
    # 获取预测 / Get Predictions
    try:
        predictions, _, _ = predict_future(merged_data, target_year)
        final_pred = predictions.iloc[-1]
        base_year = merged_data['年份'].max()
        base_waste = merged_data.iloc[-1]['生活垃圾清运量(万吨)'] / 10000
        
        # 报告摘要 / Report Summary
        st.header(t['report_summary'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            | {t['item']} | {t['value']} |
            |------------|---------|
            | {t['pred_year']} | {target_year} |
            | {t['base_year']} | {base_year} |
            | {t['pred_total_pop_report']} | {final_pred['预测总人口(万人)']/10000:.2f} {t['billion']} |
            | {t['pred_per_capita_report']} | {final_pred['预测人均垃圾量(公斤/日)']:.2f} {t['kg']} |
            """)
        
        with col2:
            increase = (final_pred['预测垃圾总量(亿吨)'] - base_waste) / base_waste * 100
            st.markdown(f"""
            | {t['item']} | {t['value']} |
            |------------|---------|
            | {t['pred_total_waste_report']} | {final_pred['预测垃圾总量(亿吨)']:.2f} {t['billion']}{t['ton']} |
            | {t['compare_base'].format(base_year)} | {increase:.1f}% |
            """)
        
        # 主要发现 / Key Findings
        st.header(t['key_findings'])
        waste_amount = f"{final_pred['预测垃圾总量(亿吨)']:.2f}"
        st.markdown(f"""
        <div class='info-box'>
        1. {t['finding1']}<br>
        2. {t['finding2'].format(waste_amount, target_year)}<br>
        3. {t['finding3']}<br>
        4. {t['finding4']}
        </div>
        """, unsafe_allow_html=True)
        
        # 政策建议 / Policy Recommendations
        st.header(t['policy_suggestions'])
        st.markdown(f"""
        <div class='success-box'>
        {t['suggestion1']}<br>
        {t['suggestion2']}<br>
        {t['suggestion3']}<br>
        {t['suggestion4']}<br>
        {t['suggestion5']}
        </div>
        """, unsafe_allow_html=True)
        
        # 报告生成器 / Report Generator
        st.header(t['report_generator'])
        col1, col2, col3 = st.columns(3)
        with col1:
            student_name = st.text_input(t['student_name'])
        with col2:
            school_name = st.text_input(t['school_name'])
        with col3:
            class_name = st.text_input(t['class_name'])
        
        if st.button(t['generate_report'], type='primary'):
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            report_content = f"""
{t['report_header']}

{'=' * 50}

**{t['report_by']}** {student_name if student_name else '_______________'}

**{t['report_school']}** {school_name if school_name else '_______________'}

**{t['report_class']}** {class_name if class_name else '_______________'}

**{t['report_date']}** {current_date}

{'=' * 50}

## {t['report_summary']}

| {t['item']} | {t['value']} |
|------------|---------|
| {t['pred_year']} | {target_year} |
| {t['base_year']} | {base_year} |
| {t['pred_total_pop_report']} | {final_pred['预测总人口(万人)']/10000:.2f} {t['billion']} |
| {t['pred_per_capita_report']} | {final_pred['预测人均垃圾量(公斤/日)']:.2f} {t['kg']} |
| {t['pred_total_waste_report']} | {final_pred['预测垃圾总量(亿吨)']:.2f} {t['billion']}{t['ton']} |

## {t['key_findings']}

{t['finding1']}

{t['finding2'].format(waste_amount, target_year)}

{t['finding3']}

{t['finding4']}

## {t['policy_suggestions']}

{t['suggestion1']}

{t['suggestion2']}

{t['suggestion3']}

{t['suggestion4']}

{t['suggestion5']}

---
*{'报告生成自：垃圾预言家预测系统' if lang_code == 'zh' else 'Generated by: Garbage Prophet Prediction System'}*
            """
            
            st.download_button(
                label="📥 Download Report | 下载报告",
                data=report_content,
                file_name=f"waste_prediction_report_{current_date}.md",
                mime="text/markdown"
            )
            
            st.success("✅ Report generated successfully! | 报告生成成功！")
            
    except Exception as e:
        st.error(f"Error: {e}")

# 底部信息 / Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 20px;'>"
    "🌍 Garbage Prophet | 垃圾预言家 - PBL Project Learning Tool"
    "</div>",
    unsafe_allow_html=True
)
