import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from data_processor import load_and_process_data, predict_future, scenario_analysis, calculate_impact

# 设置页面配置
st.set_page_config(
    page_title="垃圾预言家 - 中国未来垃圾量预测",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-title {
        font-size: 3rem !important;
        background: linear-gradient(45deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        font-size: 1rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    .metric-card p {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .section-header {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 加载数据
try:
    merged_data, pop_data, waste_data = load_and_process_data()
    data_loaded = True
    st.sidebar.success(f"✅ 数据加载成功！")
    st.sidebar.write(f"数据年份: {merged_data['年份'].min()}-{merged_data['年份'].max()}")
    st.sidebar.write(f"数据行数: {len(merged_data)}")
except Exception as e:
    data_loaded = False
    st.error(f"❌ 数据加载失败: {e}")
    st.stop()

# 侧边栏导航
st.sidebar.title("🧭 导航面板")
page = st.sidebar.radio(
    "选择展示页面:",
    ["🏠 首页", "📊 数据探索", "🔮 预言家实验室", "🎮 减量挑战", "📈 我的预测报告"]
)

# 侧边栏设置
st.sidebar.header("⚙️ 参数设置")
target_year = st.sidebar.slider("预测目标年份", 2026, 2040, 2035)
base_recycling_rate = st.sidebar.slider("基准资源化利用率", 0.3, 0.8, 0.6, help="国家十四五规划目标: 60%")

# 首页
if page == "🏠 首页":
    st.markdown("<h1 class='main-title'>🗑️ 垃圾预言家</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>基于人口数据的中国未来垃圾量预测</p>", unsafe_allow_html=True)
    
    # 最新数据概览卡片
    latest = merged_data.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>📅 最新数据年份</h3>
            <p>{int(latest['年份'])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
            <h3>👥 总人口</h3>
            <p>{latest['总人口(万人)']/10000:.2f}亿</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);'>
            <h3>🗑️ 年垃圾清运量</h3>
            <p>{latest['生活垃圾清运量(万吨)']/10000:.2f}亿吨</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #45b7d1 0%, #96c93d 100%);'>
            <h3>📦 人均日垃圾量</h3>
            <p>{latest['人均垃圾产生量(公斤/日)']:.2f}kg</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 项目介绍
    st.header("📚 项目背景")
    st.info("""
    **为什么要关注垃圾问题？**
    
    随着中国经济的发展和城市化进程的加快，城市生活垃圾产生量持续增长。
    垃圾处理已经成为影响城市可持续发展的重要挑战。
    
    本项目通过分析历史人口数据和垃圾清运数据，建立数学预测模型，
    帮助我们预测未来的垃圾产生量，从而更好地制定环保政策。
    
    **国家"十四五"规划目标：**
    - 到2025年底，全国城市生活垃圾资源化利用率达到60%左右
    - 到2025年底，基本实现垃圾分类全覆盖
    """)
    
    # 快速预览图表
    st.header("📈 趋势快速预览")
    try:
        fig = make_subplots(rows=1, cols=2, subplot_titles=('人口变化趋势', '垃圾清运量变化趋势'))
        
        fig.add_trace(
            go.Scatter(x=merged_data['年份'], y=merged_data['总人口(万人)']/10000, 
                       name='总人口(亿人)', line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=merged_data['年份'], y=merged_data['生活垃圾清运量(万吨)']/10000, 
                       name='垃圾清运量(亿吨)', line=dict(color='#ff7f0e', width=3)),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"图表渲染错误: {e}")
        # 降级显示数据表格
        st.write("数据预览:")
        st.dataframe(merged_data[['年份', '总人口(万人)', '生活垃圾清运量(万吨)']].head())

# 数据探索页面
elif page == "📊 数据探索":
    st.markdown("<h1 class='main-title'>📊 数据探索中心</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["👥 人口数据", "🗑️ 垃圾数据", "🔗 相关性分析"])
    
    with tab1:
        st.subheader("人口变化趋势分析")
        
        try:
            fig = px.line(pop_data, x='年份', y=['总人口(万人)', '城镇人口(万人)', '乡村人口(万人)'],
                          title='中国人口变化趋势',
                          labels={'value': '人口数(万人)', 'variable': '指标'})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 城镇化率
            fig2 = px.line(pop_data, x='年份', y='城镇化率(%)',
                           title='城镇化率变化趋势',
                           color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"图表渲染错误: {e}")
            st.dataframe(pop_data)
    
    with tab2:
        st.subheader("垃圾清运与处理分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                fig = px.bar(waste_data, x='年份', y='生活垃圾清运量(万吨)',
                             title='年生活垃圾清运量',
                             color='生活垃圾清运量(万吨)',
                             color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"图表渲染错误: {e}")
        
        with col2:
            try:
                fig2 = px.line(waste_data, x='年份', y=['卫生填埋处理能力(万吨/日)', '焚烧处理能力(万吨/日)'],
                               title='垃圾处理能力变化',
                               labels={'value': '处理能力(万吨/日)', 'variable': '处理方式'})
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"图表渲染错误: {e}")
        
        # 处理厂数量
        try:
            fig3 = px.bar(waste_data, x='年份', y='无害化处理厂数(座)',
                          title='无害化处理厂数量变化',
                          color_discrete_sequence=['#9467bd'])
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"图表渲染错误: {e}")
            
        # 显示原始数据
        with st.expander("查看原始数据"):
            st.dataframe(waste_data)
    
    with tab3:
        st.subheader("人口与垃圾的相关性分析")
        
        try:
            # 散点图
            fig = px.scatter(merged_data, x='总人口(万人)', y='生活垃圾清运量(万吨)',
                             title='总人口与垃圾清运量的关系',
                             trendline='ols',
                             color='年份',
                             size='人均垃圾产生量(公斤/日)',
                             hover_data=['年份', '城镇化率(%)'])
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # 计算相关系数
            corr = merged_data['总人口(万人)'].corr(merged_data['生活垃圾清运量(万吨)'])
            st.success(f"📊 总人口与垃圾清运量的相关系数: **{corr:.4f}**")
            st.info("💡 相关系数接近1表示两者高度正相关，说明人口增长确实是影响垃圾量的重要因素！")
        except Exception as e:
            st.error(f"图表渲染错误: {e}")

# 预言家实验室
elif page == "🔮 预言家实验室":
    st.markdown("<h1 class='main-title'>🔮 预言家实验室</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>在这里建立你的预测模型，成为垃圾预言家！</p>", unsafe_allow_html=True)
    
    # 模型参数调节
    st.header("🎛️ 模型参数调节")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pop_growth_adjust = st.slider(
            "人口增长调整系数",
            -2.0, 2.0, 0.0, 0.1,
            help="正值表示人口增长更快，负值表示人口减少更快"
        )
    
    with col2:
        waste_growth_adjust = st.slider(
            "人均垃圾量增长调整系数",
            -0.1, 0.1, 0.0, 0.01,
            help="正值表示人均垃圾产生量增加，负值表示减少"
        )
    
    # 获取预测
    try:
        predictions, model_pop, model_waste = predict_future(merged_data, target_year)
        
        # 应用用户调整
        predictions['调整后总人口(万人)'] = predictions['预测总人口(万人)'] * (1 + pop_growth_adjust / 100)
        predictions['调整后人均垃圾量(公斤/日)'] = predictions['预测人均垃圾量(公斤/日)'] * (1 + waste_growth_adjust)
        predictions['调整后垃圾总量(亿吨)'] = (predictions['调整后总人口(万人)'] * 10000) * (predictions['调整后人均垃圾量(公斤/日)'] / 1000) * 365 / 1e8
        
        # 展示预测结果
        st.header("📉 预测结果")
        
        # 合并历史数据和预测数据
        historical = merged_data[['年份', '生活垃圾清运量(万吨)']].copy()
        historical['类型'] = '历史数据'
        historical['垃圾总量(亿吨)'] = historical['生活垃圾清运量(万吨)'] / 10000
        
        pred_display = predictions[['年份', '调整后垃圾总量(亿吨)']].copy()
        pred_display['类型'] = '预测数据'
        pred_display['垃圾总量(亿吨)'] = pred_display['调整后垃圾总量(亿吨)']
        
        combined = pd.concat([
            historical[['年份', '垃圾总量(亿吨)', '类型']],
            pred_display[['年份', '垃圾总量(亿吨)', '类型']]
        ])
        
        # 绘制预测图
        fig = px.line(combined, x='年份', y='垃圾总量(亿吨)', color='类型',
                      title=f'中国垃圾总量预测（至{target_year}年）',
                      color_discrete_map={'历史数据': '#1f77b4', '预测数据': '#ff7f0e'},
                      line_dash='类型',
                      line_dash_map={'历史数据': 'solid', '预测数据': 'dash'})
        
        fig.update_traces(line=dict(width=4))
        fig.add_vline(x=merged_data['年份'].max(), line_dash="dash", line_color="gray", 
                      annotation_text="预测起点", annotation_position="top right")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # 关键预测数字展示
        final_pred = predictions.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label=f"📅 {target_year}年预测总人口",
                      value=f"{final_pred['调整后总人口(万人)']/10000:.2f}亿",
                      delta=f"{pop_growth_adjust:+.1f}%")
        
        with col2:
            st.metric(label=f"📦 {target_year}年预测人均日垃圾量",
                      value=f"{final_pred['调整后人均垃圾量(公斤/日)']:.2f}kg",
                      delta=f"{waste_growth_adjust*100:+.1f}%")
        
        with col3:
            st.metric(label=f"🗑️ {target_year}年预测垃圾总量",
                      value=f"{final_pred['调整后垃圾总量(亿吨)']:.2f}亿吨")
        
        with col4:
            increase = final_pred['调整后垃圾总量(亿吨)'] - merged_data.iloc[-1]['生活垃圾清运量(万吨)']/10000
            st.metric(label=f"📈 较{merged_data['年份'].max()}年增长",
                      value=f"{increase:.2f}亿吨",
                      delta=f"{increase/(merged_data.iloc[-1]['生活垃圾清运量(万吨)']/10000)*100:+.1f}%")
        
        # 预测数据表格
        with st.expander("📋 查看完整预测数据表"):
            display_cols = ['年份', '调整后总人口(万人)', '调整后人均垃圾量(公斤/日)', '调整后垃圾总量(亿吨)']
            st.dataframe(predictions[display_cols].style.format({
                '调整后总人口(万人)': '{:.0f}',
                '调整后人均垃圾量(公斤/日)': '{:.2f}',
                '调整后垃圾总量(亿吨)': '{:.2f}'
            }), use_container_width=True)
    except Exception as e:
        st.error(f"预测错误: {e}")

# 减量挑战
elif page == "🎮 减量挑战":
    st.markdown("<h1 class='main-title'>🎮 垃圾减量大挑战</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>看看我们能为地球做些什么！</p>", unsafe_allow_html=True)
    
    # 情景分析
    st.header("🌍 国家级情景模拟")
    
    col1, col2 = st.columns(2)
    
    with col1:
        reduction_rate = st.slider(
            "垃圾减量率",
            0.0, 0.5, 0.1, 0.05,
            help="通过减少浪费、源头减量等方式减少的垃圾比例"
        )
        recycling_rate = st.slider(
            "资源化利用率",
            0.3, 0.9, base_recycling_rate, 0.05,
            help="通过回收、堆肥等方式资源化利用的垃圾比例"
        )
    
    # 获取基准预测
    try:
        predictions, _, _ = predict_future(merged_data, target_year)
        scenario_results = scenario_analysis(predictions, reduction_rate, recycling_rate)
        
        final = scenario_results.iloc[-1]
        baseline = predictions.iloc[-1]['预测垃圾总量(亿吨)']
        
        with col2:
            st.markdown(f"""
            <div style='background: #e8f5e8; padding: 20px; border-radius: 15px;'>
                <h3>🎯 目标年份: {target_year}年</h3>
                <p>基准预测垃圾总量: <b>{baseline:.2f}亿吨</b></p>
                <p>减量后垃圾总量: <b>{final['减量后垃圾总量(亿吨)']:.2f}亿吨</b></p>
                <p>可回收利用量: <b>{final['可回收利用量(亿吨)']:.2f}亿吨</b></p>
                <p>需要填埋/焚烧量: <b>{final['需要填埋/焚烧量(亿吨)']:.2f}亿吨</b></p>
                <hr>
                <p style='color: green; font-weight: bold;'>
                    🌱 总共减少填埋/焚烧: {(baseline - final['需要填埋/焚烧量(亿吨)']):.2f}亿吨
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # 绘制情景比较图
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['基准预测', '减量后'],
            y=[baseline, final['减量后垃圾总量(亿吨)']],
            name='垃圾总量',
            marker_color=['#ff7f0e', '#2ca02c']
        ))
        
        fig.add_trace(go.Bar(
            x=['', ''],
            y=[0, final['可回收利用量(亿吨)']],
            name='可回收利用',
            marker_color=['#1f77b4', '#1f77b4']
        ))
        
        fig.update_layout(
            title='垃圾减量效果对比',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"计算错误: {e}")
    
    # 校园挑战
    st.header("🏫 我的校园挑战")
    st.info("计算一下你们学校的垃圾减量潜力！")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        students = st.number_input("学校学生人数", 100, 10000, 1000)
    with col2:
        waste_per = st.number_input("每人每日垃圾量(kg)", 0.1, 2.0, 0.5)
    with col3:
        school_reduction = st.slider("校园垃圾减量率目标", 0.0, 0.8, 0.3)
    
    impact = calculate_impact(students, waste_per, school_reduction)
    
    st.success(f"""
    🎉 你们学校每年可以:
    - 减少垃圾: {impact['年减量(吨)']} 吨
    - 减少CO₂排放: {impact['CO2减排(吨)']} 吨
    - 相当于种树: {int(impact['相当于种树(棵)'])} 棵！
    """)
    
    # 挑战排行榜
    st.header("🏆 减量英雄榜")
    
    challenges = [
        ("自带水杯", "减少一次性塑料瓶", 0.02),
        ("光盘行动", "减少厨余垃圾", 0.05),
        ("双面打印", "减少纸张浪费", 0.01),
        ("旧物回收", "减少固体废弃物", 0.03),
        ("布袋购物", "减少塑料袋使用", 0.01),
    ]
    
    total = 0
    st.write("勾选你能做到的减量化行动：")
    for name, desc, saving in challenges:
        if st.checkbox(f"✅ {name} - {desc}", key=name):
            total += saving
            st.success(f"+ {saving*100:.0f}% 减量化贡献")
    
    st.markdown(f"""
    <div style='background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); 
                padding: 20px; border-radius: 15px; color: white; text-align: center;'>
        <h3>🌟 你的减量化贡献总分</h3>
        <p style='font-size: 3rem; font-weight: bold;'>{total*100:.0f}%</p>
        <p>继续努力，你是地球的守护者！</p>
    </div>
    """, unsafe_allow_html=True)

# 我的预测报告
elif page == "📈 我的预测报告":
    st.markdown("<h1 class='main-title'>📈 我的预测报告</h1>", unsafe_allow_html=True)
    
    # 获取预测
    try:
        predictions, _, _ = predict_future(merged_data, target_year)
        final_pred = predictions.iloc[-1]
        base_year = merged_data['年份'].max()
        base_waste = merged_data.iloc[-1]['生活垃圾清运量(万吨)'] / 10000
        
        # 报告内容
        st.header("📋 报告摘要")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            | 项目 | 数值 |
            |------|------|
            | 预测年份 | {target_year}年 |
            | 基准年份 | {base_year}年 |
            | 预测总人口 | {final_pred['预测总人口(万人)']/10000:.2f}亿人 |
            | 预测人均日垃圾量 | {final_pred['预测人均垃圾量(公斤/日)']:.2f}公斤 |
            """)
        
        with col2:
            st.markdown(f"""
            | 项目 | 数值 |
            |------|------|
            | 预测年垃圾总量 | {final_pred['预测垃圾总量(亿吨)']:.2f}亿吨 |
            | 较基准年增长 | {final_pred['预测垃圾总量(亿吨)'] - base_waste:.2f}亿吨 |
            | 增长率 | {(final_pred['预测垃圾总量(亿吨)'] - base_waste)/base_waste*100:.1f}% |
            """)
        
        # 关键发现
        st.header("🔍 关键发现")
        
        findings = [
            ("人口趋势", f"人口总量在{base_year}年达到峰值后开始缓慢下降"),
            ("垃圾趋势", "尽管人口下降，但由于生活水平提高，人均垃圾产生量仍在增加"),
            ("综合影响", f"预计到{target_year}年，全国垃圾总量将达到{final_pred['预测垃圾总量(亿吨)']:.2f}亿吨"),
            ("政策建议", f"需要实现至少{(final_pred['预测垃圾总量(亿吨)'] - 3)/final_pred['预测垃圾总量(亿吨)']*100:.0f}%的减量化才能将垃圾总量控制在3亿吨以内")
        ]
        
        for title, content in findings:
            st.markdown(f"""
            <div style='background: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <h4>📌 {title}</h4>
                <p>{content}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 可视化报告
        st.header("📊 可视化展示")
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            '总人口预测', '人均垃圾量预测', '垃圾总量预测', '增长趋势对比'
        ))
        
        # 总人口
        fig.add_trace(
            go.Scatter(x=predictions['年份'], y=predictions['预测总人口(万人)']/10000,
                       name='总人口(亿)', line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        # 人均垃圾量
        fig.add_trace(
            go.Scatter(x=predictions['年份'], y=predictions['预测人均垃圾量(公斤/日)'],
                       name='人均日垃圾量(kg)', line=dict(color='#ff7f0e')),
            row=1, col=2
        )
        
        # 垃圾总量
        fig.add_trace(
            go.Scatter(x=predictions['年份'], y=predictions['预测垃圾总量(亿吨)'],
                       name='垃圾总量(亿吨)', line=dict(color='#2ca02c')),
            row=2, col=1
        )
        
        # 对比增长
        base_pop = merged_data.iloc[-1]['总人口(万人)']/10000
        base_waste_per = merged_data.iloc[-1]['人均垃圾产生量(公斤/日)']
        base_total = base_waste
        
        pop_growth = (predictions['预测总人口(万人)']/10000 - base_pop) / base_pop * 100
        waste_per_growth = (predictions['预测人均垃圾量(公斤/日)'] - base_waste_per) / base_waste_per * 100
        total_growth = (predictions['预测垃圾总量(亿吨)'] - base_total) / base_total * 100
        
        fig.add_trace(
            go.Scatter(x=predictions['年份'], y=pop_growth, name='人口增长(%)'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=predictions['年份'], y=waste_per_growth, name='人均垃圾增长(%)'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=predictions['年份'], y=total_growth, name='垃圾总量增长(%)'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 行动建议
        st.header("💡 行动建议")
        
        suggestions = [
            {"级别": "🌟🌟🌟", "建议": "学校全面推行垃圾分类，设置可回收、厨余、其他三类垃圾桶", "影响": "高"},
            {"级别": "🌟🌟", "建议": '开展"光盘行动"，减少厨余垃圾产生', "影响": "中"},
            {"级别": "🌟🌟", "建议": "设置旧物交换角，促进物品循环利用", "影响": "中"},
            {"级别": "🌟", "建议": "使用环保购物袋，减少一次性塑料制品", "影响": "低"},
        ]
        
        for s in suggestions:
            bg_color = '#dc3545' if s['影响'] == '高' else '#ffc107' if s['影响'] == '中' else '#28a745'
            st.markdown(f"""
            <div style='background: #fff3cd; padding: 10px; border-radius: 8px; margin: 5px 0;'>
                <span style='font-weight: bold;'>{s['级别']}</span>
                <span style='margin: 0 10px;'>{s['建议']}</span>
                <span style='background: {bg_color}; 
                             color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem;'>{s['影响']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # 导出报告按钮
        if st.button("📄 生成完整报告（打印版）"):
            st.balloons()
            st.success("报告已生成！可以通过浏览器打印功能保存为PDF。")
    except Exception as e:
        st.error(f"生成报告错误: {e}")

# 页脚
st.sidebar.markdown("---")
st.sidebar.info("""
**🧮 PBL数学项目：垃圾预言家**

这是一个基于真实数据的数学建模项目，学生可以：
- 学习数据分析技能
- 建立预测模型
- 培养环保意识

*适合初中7-9年级学生*
""")
