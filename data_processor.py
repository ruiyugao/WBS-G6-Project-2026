import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_population_data():
    """加载并处理人口数据"""
    df = pd.read_excel('人口年度数据.xls', header=None, engine='xlrd')
    
    # 提取年份
    years_row = df.iloc[2, 1:].values  # 第3行（索引2），从第2列开始是年份
    years = []
    for y in years_row:
        if isinstance(y, str) and '年' in y:
            years.append(int(y.replace('年', '')))
        else:
            years.append(None)
    
    # 提取指标数据
    indicators = df.iloc[3:8, 0].values  # 第4行到第8行是指标名
    
    # 创建数据字典
    data_dict = {'年份': years}
    for i, indicator in enumerate(indicators):
        if pd.notna(indicator):
            values = df.iloc[3 + i, 1:].values
            data_dict[indicator] = values
    
    # 转换为DataFrame
    pop_df = pd.DataFrame(data_dict)
    pop_df = pop_df.dropna(subset=['年份'])
    pop_df['年份'] = pop_df['年份'].astype(int)
    
    # 转换数值列为数值类型
    for col in pop_df.columns:
        if col != '年份':
            pop_df[col] = pd.to_numeric(pop_df[col], errors='coerce')
    
    # 选择需要的列
    result_df = pd.DataFrame({
        '年份': pop_df['年份'],
        '总人口(万人)': pop_df.get('年末总人口(万人)', pd.NA),
        '城镇人口(万人)': pop_df.get('城镇人口(万人)', pd.NA),
        '乡村人口(万人)': pop_df.get('乡村人口(万人)', pd.NA)
    })
    
    # 计算城镇化率
    result_df['城镇化率(%)'] = (result_df['城镇人口(万人)'] / result_df['总人口(万人)'] * 100).round(2)
    result_df = result_df.dropna().sort_values('年份').reset_index(drop=True)
    
    return result_df

def load_waste_data():
    """加载并处理垃圾数据"""
    df = pd.read_excel('城市生活垃圾清运和处理情况年度数据.xls', header=None, engine='xlrd')
    
    # 提取年份
    years_row = df.iloc[2, 1:].values  # 第3行（索引2），从第2列开始是年份
    years = []
    for y in years_row:
        if isinstance(y, str) and '年' in y:
            years.append(int(y.replace('年', '')))
        elif isinstance(y, (int, float)):
            years.append(int(y))
        else:
            years.append(None)
    
    # 创建数据字典
    data_dict = {'年份': years}
    
    # 查找指标行（前10行）
    for i in range(min(20, len(df))):
        indicator_name = df.iloc[i, 0]
        if pd.notna(indicator_name) and isinstance(indicator_name, str):
            values = df.iloc[i, 1:].values
            data_dict[indicator_name] = values
    
    # 转换为DataFrame
    waste_df = pd.DataFrame(data_dict)
    waste_df = waste_df.dropna(subset=['年份'])
    waste_df['年份'] = waste_df['年份'].astype(int)
    
    # 转换数值列为数值类型
    for col in waste_df.columns:
        if col != '年份':
            waste_df[col] = pd.to_numeric(waste_df[col], errors='coerce')
    
    # 选择需要的列
    result_df = pd.DataFrame({
        '年份': waste_df['年份'],
        '生活垃圾清运量(万吨)': waste_df.get('生活垃圾清运量(万吨)', pd.NA),
        '无害化处理厂数(座)': waste_df.get('无害化处理厂数(座)', pd.NA),
        '卫生填埋处理能力(吨/日)': waste_df.get('生活垃圾卫生填埋无害化处理能力(吨/日)', pd.NA),
        '焚烧处理能力(吨/日)': waste_df.get('生活垃圾焚烧无害化处理能力(吨/日)', pd.NA)
    })
    
    # 转换为万吨/日便于阅读
    result_df['卫生填埋处理能力(万吨/日)'] = result_df['卫生填埋处理能力(吨/日)'] / 10000
    result_df['焚烧处理能力(万吨/日)'] = result_df['焚烧处理能力(吨/日)'] / 10000
    
    result_df = result_df.dropna().sort_values('年份').reset_index(drop=True)
    
    return result_df

def load_and_process_data():
    """加载并合并所有数据"""
    pop_data = load_population_data()
    waste_data = load_waste_data()
    
    # 合并数据
    merged_data = pd.merge(pop_data, waste_data, on='年份', how='inner')
    
    # 计算衍生指标：垃圾清运量(万吨) / 总人口(万人) = 吨/万人/年 = 公斤/人/年
    # 1万吨垃圾 / 1万人 = 1吨/人 = 1000公斤/人
    merged_data['人均垃圾产生量(公斤/年)'] = (merged_data['生活垃圾清运量(万吨)'] / merged_data['总人口(万人)']) * 1000
    merged_data['人均垃圾产生量(公斤/日)'] = merged_data['人均垃圾产生量(公斤/年)'] / 365
    
    return merged_data, pop_data, waste_data

def predict_future(data, target_year=2035):
    """基于线性回归预测未来数据"""
    
    # 预测人口
    X = data['年份'].values.reshape(-1, 1)
    y_pop = data['总人口(万人)'].values
    
    model_pop = LinearRegression()
    model_pop.fit(X, y_pop)
    
    # 预测人均垃圾产生量
    y_waste_percap = data['人均垃圾产生量(公斤/日)'].values
    model_waste = LinearRegression()
    model_waste.fit(X, y_waste_percap)
    
    # 生成预测年份
    last_year = data['年份'].max()
    future_years = np.arange(last_year + 1, target_year + 1).reshape(-1, 1)
    
    # 进行预测
    pred_pop = model_pop.predict(future_years)
    pred_waste_percap = model_waste.predict(future_years)
    
    # 计算垃圾总量（亿吨）
    pred_total_waste = (pred_pop * 10000) * (pred_waste_percap / 1000) * 365 / 1e8
    
    # 准备预测结果
    predictions = pd.DataFrame({
        '年份': future_years.flatten(),
        '预测总人口(万人)': pred_pop.round(0),
        '预测人均垃圾量(公斤/日)': pred_waste_percap.round(2),
        '预测垃圾总量(亿吨)': pred_total_waste.round(2)
    })
    
    return predictions, model_pop, model_waste

def scenario_analysis(base_predictions, reduction_rate=0, recycling_rate=0.6):
    """情景分析：不同减量政策下的垃圾量"""
    
    results = base_predictions.copy()
    results['减量后垃圾总量(亿吨)'] = results['预测垃圾总量(亿吨)'] * (1 - reduction_rate)
    results['可回收利用量(亿吨)'] = results['减量后垃圾总量(亿吨)'] * recycling_rate
    results['需要填埋/焚烧量(亿吨)'] = results['减量后垃圾总量(亿吨)'] * (1 - recycling_rate)
    
    return results

def calculate_impact(students_per_school=1000, waste_per_student=0.5, reduction_rate=0.3):
    """计算一个学校的垃圾减量影响"""
    daily_waste = students_per_school * waste_per_student / 1000  # 吨
    yearly_waste = daily_waste * 365  # 吨/年
    
    reduced = yearly_waste * reduction_rate
    co2_saved = reduced * 0.5  # 假设每吨垃圾减少0.5吨CO2排放
    
    return {
        '年垃圾总量(吨)': round(yearly_waste, 1),
        '年减量(吨)': round(reduced, 1),
        'CO2减排(吨)': round(co2_saved, 1),
        '相当于种树(棵)': round(co2_saved * 0.5, 0)  # 每棵树每年吸收约0.5吨CO2
    }

if __name__ == '__main__':
    merged_data, pop_data, waste_data = load_and_process_data()
    print("=== 人口数据 ===")
    print(pop_data.head())
    print("\n=== 垃圾数据 ===")
    print(waste_data.head())
    print("\n=== 合并数据 ===")
    print(f"数据年份范围: {merged_data['年份'].min()} - {merged_data['年份'].max()}")
    print("\n最新年份数据:")
    print(merged_data.tail(1))
