#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银行客户流失预测 - 完整解决方案

这是一个端到端的机器学习解决方案，用于预测银行客户流失。
包含数据探索、预处理、特征工程、建模和评估的完整流程。

作者: AI专家
日期: 2024
目标: 达到AUC > 0.9的预测性能
"""

# ==================== 1. 环境设置和导入库 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import pickle
from collections import Counter
import os

# 机器学习相关库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
import lightgbm as lgb

# 贝叶斯优化
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# 设置
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

print("=" * 60)
print("银行客户流失预测 - 完整解决方案")
print("=" * 60)
print("环境设置完成！\n")

# ==================== 2. 数据加载函数 ====================
def create_mock_data(n_samples, include_label=True):
    """
    创建模拟数据用于演示
    实际使用时请删除此函数，直接加载真实数据
    """
    np.random.seed(RANDOM_STATE)
    
    data = {
        'id': [f'customer_{i}' for i in range(n_samples)],
        'f1': np.random.choice([1, 2], n_samples),  # 性别
        'f2': np.random.choice(['60', 'Y3', 'Z1', '5', '50'], n_samples),  # 职业
        'f3': np.random.randint(18, 80, n_samples),  # 年龄
        'f4': np.random.randint(0, 9, n_samples),  # 最高学历
        'f5': np.random.lognormal(8, 1, n_samples),  # 月收入
        'f6': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], n_samples),  # 行业
        'f7': np.random.choice(['1010', '1020', '1030'], n_samples),  # 建立渠道
        'f8': np.random.choice([101, 102, 201, 202], n_samples),  # 客户类型
        'f11': pd.to_datetime(np.random.choice(pd.date_range('2000-01-01', '2023-12-31'), n_samples)),  # 开户日期
        'f12': np.random.poisson(2, n_samples),  # 存款产品数量
        'f13': np.random.poisson(1, n_samples),  # 投资产品数量
        'f14': np.random.poisson(5, n_samples),  # 近1月登录次数
        'f15': np.random.poisson(15, n_samples),  # 近3月登录次数
        'f16': np.random.poisson(30, n_samples),  # 近6月登录次数
        'f17': np.random.lognormal(10, 2, n_samples),  # 存款余额
        'f18': np.random.lognormal(10, 2, n_samples),  # 存款月均余额
        'f19': np.random.lognormal(8, 3, n_samples),  # 投资余额
        'f20': np.random.lognormal(8, 3, n_samples),  # 投资月均余额
        
        # 交易特征（来自Hive SQL处理）
        'total_transactions': np.random.poisson(50, n_samples),
        'active_transaction_days': np.random.poisson(20, n_samples),
        'inflow_count': np.random.poisson(25, n_samples),
        'outflow_count': np.random.poisson(25, n_samples),
        'total_inflow': np.random.lognormal(9, 2, n_samples),
        'total_outflow': np.random.lognormal(9, 2, n_samples),
        'avg_transaction_amount': np.random.lognormal(6, 1, n_samples),
        'std_transaction_amount': np.random.lognormal(6, 1, n_samples),
        'max_transaction_amount': np.random.lognormal(8, 2, n_samples),
        'outflow_inflow_ratio': np.random.uniform(0, 2, n_samples),
        'net_flow': np.random.normal(0, 10000, n_samples),
        'quick_payment_count': np.random.poisson(10, n_samples),
        'interbank_transfer_count': np.random.poisson(5, n_samples),
        'mobile_banking_count': np.random.poisson(15, n_samples),
        'online_banking_count': np.random.poisson(8, n_samples),
        'channel_diversity': np.random.randint(1, 6, n_samples),
        'transactions_last_7days': np.random.poisson(3, n_samples),
        'transactions_last_30days': np.random.poisson(12, n_samples),
        'amount_last_7days': np.random.lognormal(7, 1, n_samples),
        'amount_last_30days': np.random.lognormal(8, 1, n_samples),
        'large_transaction_count': np.random.poisson(2, n_samples),
        'small_transaction_count': np.random.poisson(8, n_samples),
    }
    
    if include_label:
        # 创建目标变量，流失率约15%
        data['label'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    return pd.DataFrame(data)

def load_data(train_path=None, test_path=None):
    """
    加载训练和测试数据
    假设数据已经通过Hive SQL进行了特征工程处理
    """
    if train_path and os.path.exists(train_path):
        train_data = pd.read_csv(train_path)
    else:
        print("正在创建模拟数据用于演示...")
        train_data = create_mock_data(10000, include_label=True)
    
    if test_path and os.path.exists(test_path):
        test_data = pd.read_csv(test_path)
    else:
        test_data = create_mock_data(5000, include_label=False)
    
    return train_data, test_data

# ==================== 3. 数据探索性分析(EDA) ====================
def perform_eda(data):
    """执行数据探索性分析"""
    print("=" * 40)
    print("数据探索性分析(EDA)")
    print("=" * 40)
    
    print(f"数据集形状: {data.shape}")
    print(f"数据类型分布: {data.dtypes.value_counts().to_dict()}")
    
    # 缺失值统计
    missing_info = data.isnull().sum()
    missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
    if len(missing_info) > 0:
        print(f"\n缺失值统计:")
        for col, count in missing_info.items():
            print(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
    else:
        print("\n✓ 没有缺失值")
    
    # 目标变量分布
    if 'label' in data.columns:
        label_counts = data['label'].value_counts()
        print(f"\n目标变量分布:")
        print(f"  非流失客户 (0): {label_counts[0]} ({label_counts[0]/len(data)*100:.1f}%)")
        print(f"  流失客户 (1): {label_counts[1]} ({label_counts[1]/len(data)*100:.1f}%)")
        
        # 可视化
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        data['label'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
        plt.title('目标变量分布')
        plt.xticks([0, 1], ['非流失', '流失'], rotation=0)
        plt.ylabel('客户数量')
        
        plt.subplot(1, 3, 2)
        data['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'orange'])
        plt.title('目标变量比例')
        plt.ylabel('')
        
        # 数值特征分布示例
        plt.subplot(1, 3, 3)
        if 'f3' in data.columns:
            data['f3'].hist(bins=30, alpha=0.7, edgecolor='black')
            plt.title('年龄分布')
            plt.xlabel('年龄')
            plt.ylabel('频数')
        
        plt.tight_layout()
        plt.show()
    
    return missing_info

def analyze_features(data):
    """分析特征类型和分布"""
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numerical_cols:
        numerical_cols.remove('label')
    
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    
    print(f"\n数值型特征数量: {len(numerical_cols)}")
    print(f"分类型特征数量: {len(categorical_cols)}")
    
    # 异常值检测
    print(f"\n异常值检测结果 (前10个数值特征):")
    for col in numerical_cols[:10]:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_percent = len(outliers) / len(data) * 100
        print(f"  {col}: {len(outliers)} 个异常值 ({outlier_percent:.1f}%)")
    
    return numerical_cols, categorical_cols

# ==================== 4. 数据清洗与预处理 ====================
def clean_and_preprocess_data(train_data, test_data):
    """数据清洗与预处理"""
    print("\n" + "=" * 40)
    print("数据清洗与预处理")
    print("=" * 40)
    
    train_processed = train_data.copy()
    test_processed = test_data.copy()
    
    # 1. 处理ID列
    train_ids = train_processed['id']
    test_ids = test_processed['id']
    train_processed = train_processed.drop('id', axis=1)
    test_processed = test_processed.drop('id', axis=1)
    
    # 2. 处理日期特征
    date_columns = ['f11']
    for col in date_columns:
        if col in train_processed.columns:
            train_processed[col] = pd.to_datetime(train_processed[col], errors='coerce')
            test_processed[col] = pd.to_datetime(test_processed[col], errors='coerce')
            
            reference_date = pd.to_datetime('2025-04-30')
            
            train_processed[f'{col}_year'] = train_processed[col].dt.year
            train_processed[f'{col}_month'] = train_processed[col].dt.month
            train_processed[f'{col}_days_since'] = (reference_date - train_processed[col]).dt.days
            
            test_processed[f'{col}_year'] = test_processed[col].dt.year
            test_processed[f'{col}_month'] = test_processed[col].dt.month
            test_processed[f'{col}_days_since'] = (reference_date - test_processed[col]).dt.days
            
            train_processed = train_processed.drop(col, axis=1)
            test_processed = test_processed.drop(col, axis=1)
    
    # 3. 处理分类变量
    categorical_columns = ['f2', 'f6', 'f7', 'f8', 'f9', 'f10']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in train_processed.columns:
            le = LabelEncoder()
            combined_values = pd.concat([
                train_processed[col].astype(str), 
                test_processed[col].astype(str)
            ]).fillna('missing')
            
            le.fit(combined_values)
            
            train_processed[col] = le.transform(train_processed[col].astype(str).fillna('missing'))
            test_processed[col] = le.transform(test_processed[col].astype(str).fillna('missing'))
            
            label_encoders[col] = le
    
    # 4. 处理缺失值
    numerical_columns = train_processed.select_dtypes(include=[np.number]).columns
    if 'label' in numerical_columns:
        numerical_columns = numerical_columns.drop('label')
    
    for col in numerical_columns:
        median_value = train_processed[col].median()
        train_processed[col] = train_processed[col].fillna(median_value)
        if col in test_processed.columns:
            test_processed[col] = test_processed[col].fillna(median_value)
    
    # 5. 处理异常值
    outlier_bounds = {}
    for col in numerical_columns:
        Q1 = train_processed[col].quantile(0.01)
        Q3 = train_processed[col].quantile(0.99)
        
        train_processed[col] = train_processed[col].clip(lower=Q1, upper=Q3)
        if col in test_processed.columns:
            test_processed[col] = test_processed[col].clip(lower=Q1, upper=Q3)
        
        outlier_bounds[col] = (Q1, Q3)
    
    print(f"训练集处理后形状: {train_processed.shape}")
    print(f"测试集处理后形状: {test_processed.shape}")
    
    return train_processed, test_processed, train_ids, test_ids, label_encoders, outlier_bounds

def balance_dataset(X, y, method='oversample'):
    """处理数据不平衡问题 - 手动重采样"""
    print(f"\n原始数据分布: {Counter(y)}")
    
    if method == 'undersample':
        minority_count = sum(y == 1)
        majority_indices = np.where(y == 0)[0]
        minority_indices = np.where(y == 1)[0]
        
        np.random.seed(RANDOM_STATE)
        selected_majority_indices = np.random.choice(majority_indices, minority_count, replace=False)
        balanced_indices = np.concatenate([selected_majority_indices, minority_indices])
        
        X_balanced = X.iloc[balanced_indices]
        y_balanced = y[balanced_indices]
        
    elif method == 'oversample':
        majority_count = sum(y == 0)
        minority_indices = np.where(y == 1)[0]
        majority_indices = np.where(y == 0)[0]
        
        np.random.seed(RANDOM_STATE)
        oversampled_minority_indices = np.random.choice(minority_indices, majority_count, replace=True)
        balanced_indices = np.concatenate([majority_indices, oversampled_minority_indices])
        
        X_balanced = X.iloc[balanced_indices]
        y_balanced = y[balanced_indices]
    
    else:
        X_balanced, y_balanced = X, y
    
    print(f"平衡后数据分布: {Counter(y_balanced)}")
    return X_balanced, y_balanced

# ==================== 5. 特征工程 ====================
def create_additional_features(data):
    """创建额外的特征"""
    print("\n" + "=" * 40)
    print("特征工程")
    print("=" * 40)
    
    feature_data = data.copy()
    
    # 1. 金融行为特征
    if 'f17' in feature_data.columns and 'f18' in feature_data.columns:
        feature_data['deposit_change_ratio'] = (
            feature_data['f17'] - feature_data['f18']
        ) / (feature_data['f18'] + 1)
    
    if 'f19' in feature_data.columns and 'f20' in feature_data.columns:
        feature_data['investment_change_ratio'] = (
            feature_data['f19'] - feature_data['f20']
        ) / (feature_data['f20'] + 1)
    
    # 2. 数字化使用程度
    if all(col in feature_data.columns for col in ['f14', 'f15', 'f16']):
        feature_data['mobile_usage_trend'] = (
            feature_data['f14'] * 6 + feature_data['f15'] * 2 + feature_data['f16']
        ) / 9
        
        feature_data['mobile_usage_stability'] = (
            feature_data['f16'] / 6 - feature_data['f15'] / 3
        ).abs()
    
    # 3. 交易行为特征
    if 'total_transactions' in feature_data.columns and 'active_transaction_days' in feature_data.columns:
        feature_data['transaction_intensity'] = (
            feature_data['total_transactions'] / (feature_data['active_transaction_days'] + 1)
        )
    
    if 'total_inflow' in feature_data.columns and 'total_outflow' in feature_data.columns:
        feature_data['fund_efficiency'] = (
            (feature_data['total_inflow'] + feature_data['total_outflow']) / 
            (feature_data['total_transactions'] + 1)
        )
    
    # 4. 风险特征
    if 'large_transaction_count' in feature_data.columns and 'total_transactions' in feature_data.columns:
        feature_data['large_transaction_ratio'] = (
            feature_data['large_transaction_count'] / (feature_data['total_transactions'] + 1)
        )
    
    # 5. 分组特征
    if 'f3' in feature_data.columns:
        feature_data['age_group'] = pd.cut(
            feature_data['f3'], 
            bins=[0, 30, 45, 60, 100], 
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(int)
    
    if 'f5' in feature_data.columns:
        feature_data['income_group'] = pd.qcut(
            feature_data['f5'], 
            q=5, 
            labels=[0, 1, 2, 3, 4],
            duplicates='drop'
        ).astype(int)
    
    print(f"特征工程后特征数量: {feature_data.shape[1]}")
    return feature_data

def select_features(X_train, y_train, X_test, method='correlation', top_k=50):
    """特征选择"""
    print(f"\n特征选择 (方法: {method}, Top {top_k})")
    
    if method == 'correlation':
        correlations = pd.DataFrame(X_train).corrwith(pd.Series(y_train)).abs()
        selected_features = correlations.nlargest(top_k).index.tolist()
    elif method == 'variance':
        variances = pd.DataFrame(X_train).var()
        selected_features = variances.nlargest(top_k).index.tolist()
    else:
        selected_features = X_train.columns.tolist()
    
    print(f"选择的特征数量: {len(selected_features)}")
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_test_selected, selected_features

# ==================== 6. 机器学习建模 ====================
def train_and_evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """训练和评估模型"""
    print(f"\n--- 训练 {model_name} ---")
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    print(f"训练集 AUC: {train_auc:.4f}")
    print(f"验证集 AUC: {val_auc:.4f}")
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model, val_auc, cv_scores.mean()

def train_all_models(X_train, X_val, y_train, y_val):
    """训练所有模型"""
    print("\n" + "=" * 40)
    print("机器学习建模")
    print("=" * 40)
    
    model_results = {}
    
    # 1. Random Forest with RandomizedSearchCV
    print("\n🌲 训练随机森林模型")
    rf_param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_random_search = RandomizedSearchCV(
        rf_base, rf_param_dist, n_iter=15, cv=3, 
        scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1
    )
    
    rf_random_search.fit(X_train, y_train)
    rf_best = rf_random_search.best_estimator_
    
    rf_model, rf_val_auc, rf_cv_auc = train_and_evaluate_model(
        rf_best, X_train, X_val, y_train, y_val, "Random Forest"
    )
    
    model_results['RandomForest'] = {
        'model': rf_model,
        'val_auc': rf_val_auc,
        'cv_auc': rf_cv_auc
    }
    
    # 2. XGBoost with BayesSearchCV
    print("\n🚀 训练XGBoost模型")
    xgb_param_space = {
        'n_estimators': Integer(100, 300),
        'max_depth': Integer(3, 8),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'reg_alpha': Real(0, 1),
        'reg_lambda': Real(0, 1)
    }
    
    xgb_base = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
    xgb_bayes_search = BayesSearchCV(
        xgb_base, xgb_param_space, n_iter=20, cv=3,
        scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1
    )
    
    xgb_bayes_search.fit(X_train, y_train)
    xgb_best = xgb_bayes_search.best_estimator_
    
    xgb_model, xgb_val_auc, xgb_cv_auc = train_and_evaluate_model(
        xgb_best, X_train, X_val, y_train, y_val, "XGBoost"
    )
    
    model_results['XGBoost'] = {
        'model': xgb_model,
        'val_auc': xgb_val_auc,
        'cv_auc': xgb_cv_auc
    }
    
    # 3. LightGBM with BayesSearchCV
    print("\n💡 训练LightGBM模型")
    lgb_param_space = {
        'n_estimators': Integer(100, 300),
        'max_depth': Integer(3, 8),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'num_leaves': Integer(10, 100),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'reg_alpha': Real(0, 1),
        'reg_lambda': Real(0, 1)
    }
    
    lgb_base = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_bayes_search = BayesSearchCV(
        lgb_base, lgb_param_space, n_iter=20, cv=3,
        scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1
    )
    
    lgb_bayes_search.fit(X_train, y_train)
    lgb_best = lgb_bayes_search.best_estimator_
    
    lgb_model, lgb_val_auc, lgb_cv_auc = train_and_evaluate_model(
        lgb_best, X_train, X_val, y_train, y_val, "LightGBM"
    )
    
    model_results['LightGBM'] = {
        'model': lgb_model,
        'val_auc': lgb_val_auc,
        'cv_auc': lgb_cv_auc
    }
    
    return model_results

def create_stacking_ensemble(models, X_train, X_val, y_train, y_val):
    """创建Stacking集成模型"""
    print("\n🎯 创建Stacking集成模型")
    
    # 生成基础模型的预测
    train_meta_features = np.zeros((X_train.shape[0], len(models)))
    val_meta_features = np.zeros((X_val.shape[0], len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"生成 {name} 的元特征...")
        
        cv_predictions = cross_val_predict(
            model['model'], X_train, y_train, cv=5, method='predict_proba', n_jobs=-1
        )
        train_meta_features[:, i] = cv_predictions[:, 1]
        
        val_predictions = model['model'].predict_proba(X_val)[:, 1]
        val_meta_features[:, i] = val_predictions
    
    # 训练元学习器
    meta_learner = LogisticRegression(random_state=RANDOM_STATE)
    meta_learner.fit(train_meta_features, y_train)
    
    train_stacking_pred = meta_learner.predict_proba(train_meta_features)[:, 1]
    val_stacking_pred = meta_learner.predict_proba(val_meta_features)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_stacking_pred)
    val_auc = roc_auc_score(y_val, val_stacking_pred)
    
    print(f"Stacking训练集 AUC: {train_auc:.4f}")
    print(f"Stacking验证集 AUC: {val_auc:.4f}")
    
    return meta_learner, val_auc, train_meta_features, val_meta_features

# ==================== 7. 模型评估和特征重要性 ====================
def evaluate_and_compare_models(model_results):
    """模型性能比较"""
    print("\n" + "=" * 40)
    print("模型性能对比")
    print("=" * 40)
    
    performance_df = pd.DataFrame([
        {
            '模型': name,
            '验证集AUC': results['val_auc'],
            '交叉验证AUC': results['cv_auc']
        }
        for name, results in model_results.items()
    ])
    
    performance_df = performance_df.sort_values('验证集AUC', ascending=False)
    print(performance_df.round(4))
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    colors = ['lightblue', 'lightgreen', 'orange', 'pink'][:len(performance_df)]
    bars = plt.bar(performance_df['模型'], performance_df['验证集AUC'], color=colors)
    plt.title('模型验证集AUC对比')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    for bar, v in zip(bars, performance_df['验证集AUC']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{v:.3f}', ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(performance_df['模型'], performance_df['交叉验证AUC'], color=colors)
    plt.title('模型交叉验证AUC对比')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    for bar, v in zip(bars, performance_df['交叉验证AUC']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    best_model_name = performance_df.iloc[0]['模型']
    best_auc = performance_df.iloc[0]['验证集AUC']
    
    print(f"\n🏆 最佳模型: {best_model_name} (验证集AUC: {best_auc:.4f})")
    
    if best_auc >= 0.9:
        print("✅ 达到AUC > 0.9的目标要求！")
    else:
        print(f"⚠️  当前AUC {best_auc:.4f}，距离0.9目标还需提升")
    
    return best_model_name, performance_df

def analyze_feature_importance(model, feature_names, model_name, top_k=20):
    """分析特征重要性"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"模型 {model_name} 不支持特征重要性分析")
        return None
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n{model_name} Top {top_k} 重要特征:")
    top_features = feature_importance_df.head(top_k)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='lightblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性')
    plt.title(f'{model_name} - Top {top_k} 重要特征')
    plt.gca().invert_yaxis()
    
    for i, v in enumerate(top_features['importance']):
        plt.text(v + max(top_features['importance']) * 0.01, i, f'{v:.3f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:30s} {row['importance']:.4f}")
    
    return feature_importance_df

# ==================== 8. 预测和结果输出 ====================
def generate_predictions(models, X_test, test_ids, best_model_name):
    """生成最终预测结果"""
    print("\n" + "=" * 40)
    print("生成预测结果")
    print("=" * 40)
    
    best_model = models[best_model_name]['model']
    print(f"使用最佳模型进行预测: {best_model_name}")
    
    if best_model_name == 'Stacking':
        # Stacking模型需要特殊处理
        test_meta_features = np.zeros((X_test.shape[0], len(models) - 1))
        i = 0
        for name, model_info in models.items():
            if name != 'Stacking':
                test_predictions = model_info['model'].predict_proba(X_test)[:, 1]
                test_meta_features[:, i] = test_predictions
                i += 1
        final_predictions = best_model.predict_proba(test_meta_features)[:, 1]
    else:
        final_predictions = best_model.predict_proba(X_test)[:, 1]
    
    results_df = pd.DataFrame({
        'id': test_ids,
        'prediction_proba': final_predictions,
        'prediction_label': (final_predictions > 0.5).astype(int)
    })
    
    output_file = '/workspace/bank_churn_predictions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")
    
    print(f"\n预测统计:")
    print(f"预测流失客户数量: {results_df['prediction_label'].sum()}")
    print(f"预测流失率: {results_df['prediction_label'].mean():.3f}")
    print(f"预测概率分布:")
    print(f"  最小值: {final_predictions.min():.3f}")
    print(f"  最大值: {final_predictions.max():.3f}")
    print(f"  平均值: {final_predictions.mean():.3f}")
    print(f"  中位数: {np.median(final_predictions):.3f}")
    
    return results_df

def save_model_artifacts(models, best_model_name, scaler, label_encoders, selected_features, train_data, test_data):
    """保存模型组件"""
    print("\n" + "=" * 40)
    print("保存模型组件")
    print("=" * 40)
    
    best_model = models[best_model_name]['model']
    
    # 保存最佳模型
    with open('/workspace/best_churn_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("✓ 最佳模型已保存")
    
    # 保存预处理组件
    preprocessing_components = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'selected_features': selected_features,
        'best_model_name': best_model_name
    }
    
    with open('/workspace/preprocessing_components.pkl', 'wb') as f:
        pickle.dump(preprocessing_components, f)
    print("✓ 预处理组件已保存")
    
    # 保存所有模型结果
    with open('/workspace/all_model_results.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("✓ 所有模型结果已保存")
    
    # 生成模型报告
    best_auc = models[best_model_name]['val_auc']
    report = f"""
银行客户流失预测模型报告
==============================

模型开发时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
最佳模型: {best_model_name}
验证集AUC: {best_auc:.4f}
交叉验证AUC: {models[best_model_name]['cv_auc']:.4f}

数据集信息:
- 训练样本数量: {len(train_data)}
- 测试样本数量: {len(test_data)}
- 特征数量: {len(selected_features)}
- 流失率: {train_data['label'].mean():.3f}

模型性能对比:
"""
    
    for name, results in models.items():
        report += f"- {name}: {results['val_auc']:.4f}\n"
    
    with open('/workspace/model_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ 模型报告已保存")

def generate_business_insights(best_model, selected_features, best_model_name):
    """生成业务洞察"""
    print("\n" + "=" * 60)
    print("业务洞察与模型解释")
    print("=" * 60)
    
    # 分析特征重要性
    feature_importance_df = analyze_feature_importance(best_model, selected_features, best_model_name, top_k=20)
    
    if feature_importance_df is not None:
        print(f"\n💡 关键业务洞察:")
        print("-" * 40)
        
        # 特征名称映射
        feature_mapping = {
            'f14': '近1月移动银行登录次数',
            'f15': '近3月移动银行登录次数', 
            'f16': '近6月移动银行登录次数',
            'f17': '存款时点余额',
            'f18': '存款月均余额',
            'f19': '投资时点余额',
            'f20': '投资月均余额',
            'total_transactions': '总交易次数',
            'mobile_banking_count': '手机银行使用次数',
            'transactions_last_30days': '近30天交易次数',
            'outflow_inflow_ratio': '资金流出流入比',
            'avg_transaction_amount': '平均交易金额',
            'mobile_usage_trend': '手机银行使用趋势',
            'transaction_intensity': '交易活跃度'
        }
        
        top_5_features = feature_importance_df.head(5)
        for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
            feature_name = feature_mapping.get(row['feature'], row['feature'])
            print(f"{i}. {feature_name} - 客户流失的强预测因子")
        
        print(f"\n📋 银行客户流失预防建议:")
        print("""
1. 🏦 数字化服务优化
   - 重点关注移动银行使用频率下降的客户
   - 提升手机银行APP用户体验和功能
   - 推出个性化的数字化金融服务

2. 💰 资产管理服务
   - 建立客户资产变化监控系统
   - 为资产流失客户提供专属理财顾问
   - 优化投资产品组合和收益率

3. 🔄 交易行为分析
   - 建立交易频率和金额预警机制
   - 对交易活跃度下降的客户主动服务
   - 分析客户资金流向，提前识别流失风险

4. 🎯 精准营销策略
   - 基于流失概率进行客户精准分层
   - 为高风险客户制定个性化挽留方案
   - 优化客户服务流程和产品推荐算法
        """)

# ==================== 9. 主函数 ====================
def main():
    """主函数 - 执行完整的机器学习流程"""
    try:
        # 1. 数据加载
        print("📊 步骤1: 数据加载")
        train_data, test_data = load_data()
        print(f"训练集形状: {train_data.shape}")
        print(f"测试集形状: {test_data.shape}")
        print(f"流失率: {train_data['label'].mean():.3f}")
        
        # 2. 数据探索性分析
        print("\n📈 步骤2: 数据探索性分析")
        missing_info = perform_eda(train_data)
        numerical_cols, categorical_cols = analyze_features(train_data)
        
        # 3. 数据清洗与预处理
        print("\n🧹 步骤3: 数据清洗与预处理")
        train_processed, test_processed, train_ids, test_ids, label_encoders, outlier_bounds = clean_and_preprocess_data(train_data, test_data)
        
        # 4. 准备特征和目标变量
        X = train_processed.drop('label', axis=1)
        y = train_processed['label'].values
        
        # 5. 数据平衡处理
        print("\n⚖️ 步骤4: 数据平衡处理")
        X_balanced, y_balanced = balance_dataset(X, y, method='oversample')
        
        # 6. 特征工程
        print("\n🔧 步骤5: 特征工程")
        X_balanced_enhanced = create_additional_features(X_balanced)
        test_processed_enhanced = create_additional_features(test_processed)
        
        # 7. 特征选择
        print("\n🎯 步骤6: 特征选择")
        X_train_selected, X_test_selected, selected_features = select_features(
            X_balanced_enhanced, y_balanced, test_processed_enhanced, 
            method='correlation', top_k=50
        )
        
        # 8. 数据标准化
        print("\n📏 步骤7: 数据标准化")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)
        
        # 9. 分割验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_balanced, test_size=0.2, random_state=RANDOM_STATE, stratify=y_balanced
        )
        
        print(f"最终训练集形状: {X_train_final.shape}")
        print(f"验证集形状: {X_val.shape}")
        
        # 10. 模型训练
        print("\n🤖 步骤8: 机器学习建模")
        model_results = train_all_models(X_train_final, X_val, y_train_final, y_val)
        
        # 11. Stacking集成
        print("\n🎯 步骤9: Stacking集成模型")
        stacking_model, stacking_val_auc, _, _ = create_stacking_ensemble(
            model_results, X_train_final, X_val, y_train_final, y_val
        )
        
        model_results['Stacking'] = {
            'model': stacking_model,
            'val_auc': stacking_val_auc,
            'cv_auc': stacking_val_auc
        }
        
        # 12. 模型评估和比较
        print("\n📊 步骤10: 模型评估与比较")
        best_model_name, performance_df = evaluate_and_compare_models(model_results)
        
        # 13. 特征重要性分析
        if best_model_name != 'Stacking':
            print(f"\n🔍 步骤11: {best_model_name} 特征重要性分析")
            analyze_feature_importance(
                model_results[best_model_name]['model'], 
                selected_features, 
                best_model_name, 
                top_k=20
            )
        
        # 14. 生成预测结果
        print("\n🎯 步骤12: 生成预测结果")
        prediction_results = generate_predictions(
            model_results, X_test_scaled, test_ids, best_model_name
        )
        
        # 15. 保存模型和组件
        print("\n💾 步骤13: 保存模型组件")
        save_model_artifacts(
            model_results, best_model_name, scaler, label_encoders, 
            selected_features, train_data, test_data
        )
        
        # 16. 生成业务洞察
        print("\n💡 步骤14: 业务洞察分析")
        generate_business_insights(
            model_results[best_model_name]['model'], 
            selected_features, 
            best_model_name
        )
        
        # 17. 最终总结
        best_auc = model_results[best_model_name]['val_auc']
        print("\n" + "=" * 60)
        print("🎉 银行客户流失预测项目完成！")
        print("=" * 60)
        print(f"🏆 最佳模型: {best_model_name}")
        print(f"📈 验证集AUC: {best_auc:.4f}")
        
        if best_auc >= 0.9:
            print("✅ 成功达到AUC > 0.9的目标要求！")
        else:
            print(f"⚠️  当前AUC {best_auc:.4f}，可进一步优化特征工程和模型参数")
        
        print(f"\n📁 输出文件:")
        print(f"• bank_churn_predictions.csv - 预测结果")
        print(f"• best_churn_model.pkl - 最佳模型")
        print(f"• preprocessing_components.pkl - 预处理组件")
        print(f"• all_model_results.pkl - 所有模型结果")
        print(f"• model_report.txt - 详细模型报告")
        
        return model_results, best_model_name, prediction_results
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ==================== 10. 执行主程序 ====================
if __name__ == "__main__":
    print("开始执行银行客户流失预测完整解决方案...")
    model_results, best_model_name, prediction_results = main()
    
    if model_results is not None:
        print("\n🎊 项目执行成功完成！")
        print("📋 请查看生成的预测结果和模型报告。")
    else:
        print("\n💥 项目执行失败，请检查错误信息并重试。")