#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银行客户流失预测 - 机器学习竞赛解决方案
作者: ML竞赛专家
目标: 预测银行客户流失，AUC > 0.9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import gc

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class BankChurnPredictor:
    """银行客户流失预测器"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.scalers = {}
        self.label_encoders = {}
        
    def load_data(self, market_train_path, market_test_path, enriched_train_path=None):
        """加载数据"""
        print("正在加载数据...")
        
        # 加载基础数据
        self.market_train = pd.read_csv(market_train_path)
        self.market_test = pd.read_csv(market_test_path)
        
        # 如果有融合后的数据，直接加载
        if enriched_train_path:
            self.train_data = pd.read_csv(enriched_train_path)
            print(f"已加载融合后的训练数据: {self.train_data.shape}")
        else:
            # 否则使用基础数据
            self.train_data = self.market_train.copy()
            print(f"使用基础训练数据: {self.train_data.shape}")
        
        print(f"训练集形状: {self.train_data.shape}")
        print(f"测试集形状: {self.market_test.shape}")
        
        return self
    
    def explore_data(self):
        """数据探索性分析"""
        print("\n=== 数据探索性分析 (EDA) ===")
        
        # 1. 数据概览
        print("\n1. 数据概览:")
        print(f"训练集形状: {self.train_data.shape}")
        print(f"测试集形状: {self.market_test.shape}")
        
        # 2. 目标变量分布
        if 'label' in self.train_data.columns:
            label_counts = self.train_data['label'].value_counts()
            print(f"\n2. 目标变量分布:")
            print(f"流失客户 (1): {label_counts[1]} ({label_counts[1]/len(self.train_data)*100:.2f}%)")
            print(f"非流失客户 (0): {label_counts[0]} ({label_counts[0]/len(self.train_data)*100:.2f}%)")
        
        # 3. 数据类型和缺失值
        print(f"\n3. 数据类型和缺失值:")
        info_df = pd.DataFrame({
            'dtype': self.train_data.dtypes,
            'missing_count': self.train_data.isnull().sum(),
            'missing_pct': self.train_data.isnull().sum() / len(self.train_data) * 100
        })
        print(info_df[info_df['missing_count'] > 0])
        
        # 4. 数值型特征统计
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
        if 'label' in numeric_cols:
            numeric_cols.remove('label')
            
        if numeric_cols:
            print(f"\n4. 数值型特征统计:")
            print(self.train_data[numeric_cols].describe())
        
        # 5. 分类特征统计
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        if 'id' in categorical_cols:
            categorical_cols.remove('id')
            
        if categorical_cols:
            print(f"\n5. 分类特征统计:")
            for col in categorical_cols[:5]:  # 只显示前5个
                print(f"\n{col}:")
                print(self.train_data[col].value_counts().head())
        
        return self
    
    def visualize_data(self):
        """数据可视化"""
        print("\n=== 数据可视化 ===")
        
        # 设置图表大小
        plt.figure(figsize=(20, 15))
        
        # 1. 目标变量分布
        plt.subplot(2, 3, 1)
        if 'label' in self.train_data.columns:
            self.train_data['label'].value_counts().plot(kind='bar')
            plt.title('目标变量分布')
            plt.xlabel('是否流失')
            plt.ylabel('客户数量')
        
        # 2. 年龄分布
        if 'f3' in self.train_data.columns:
            plt.subplot(2, 3, 2)
            plt.hist(self.train_data['f3'].dropna(), bins=30, alpha=0.7)
            plt.title('年龄分布')
            plt.xlabel('年龄')
            plt.ylabel('频次')
        
        # 3. 月收入分布
        if 'f5' in self.train_data.columns:
            plt.subplot(2, 3, 3)
            plt.hist(self.train_data['f5'].dropna(), bins=30, alpha=0.7)
            plt.title('月收入分布')
            plt.xlabel('月收入')
            plt.ylabel('频次')
        
        # 4. 存款余额分布
        if 'f17' in self.train_data.columns:
            plt.subplot(2, 3, 4)
            plt.hist(self.train_data['f17'].dropna(), bins=30, alpha=0.7)
            plt.title('存款余额分布')
            plt.xlabel('存款余额')
            plt.ylabel('频次')
        
        # 5. 投资资产分布
        if 'f19' in self.train_data.columns:
            plt.subplot(2, 3, 5)
            plt.hist(self.train_data['f19'].dropna(), bins=30, alpha=0.7)
            plt.title('投资资产分布')
            plt.xlabel('投资资产')
            plt.ylabel('频次')
        
        # 6. 手机银行登录次数分布
        if 'f14' in self.train_data.columns:
            plt.subplot(2, 3, 6)
            plt.hist(self.train_data['f14'].dropna(), bins=30, alpha=0.7)
            plt.title('近1个月手机银行登录次数')
            plt.xlabel('登录次数')
            plt.ylabel('频次')
        
        plt.tight_layout()
        plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def clean_data(self):
        """数据清洗与预处理"""
        print("\n=== 数据清洗与预处理 ===")
        
        # 1. 处理缺失值
        print("1. 处理缺失值...")
        
        # 数值型特征用中位数填充
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
        if 'label' in numeric_cols:
            numeric_cols.remove('label')
            
        for col in numeric_cols:
            if self.train_data[col].isnull().sum() > 0:
                median_val = self.train_data[col].median()
                self.train_data[col].fillna(median_val, inplace=True)
                print(f"   {col}: 用中位数 {median_val:.2f} 填充")
        
        # 分类特征用众数填充
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        if 'id' in categorical_cols:
            categorical_cols.remove('id')
            
        for col in categorical_cols:
            if self.train_data[col].isnull().sum() > 0:
                mode_val = self.train_data[col].mode()[0]
                self.train_data[col].fillna(mode_val, inplace=True)
                print(f"   {col}: 用众数 '{mode_val}' 填充")
        
        # 2. 异常值处理
        print("2. 处理异常值...")
        
        # 对数值型特征进行异常值检测和处理
        for col in numeric_cols:
            Q1 = self.train_data[col].quantile(0.25)
            Q3 = self.train_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((self.train_data[col] < lower_bound) | 
                             (self.train_data[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                # 用边界值替换异常值
                self.train_data[col] = np.where(
                    self.train_data[col] < lower_bound, lower_bound, self.train_data[col]
                )
                self.train_data[col] = np.where(
                    self.train_data[col] > upper_bound, upper_bound, self.train_data[col]
                )
                print(f"   {col}: 处理了 {outliers_count} 个异常值")
        
        # 3. 处理不平衡问题
        print("3. 处理数据不平衡...")
        
        if 'label' in self.train_data.columns:
            # 计算流失率
            churn_rate = self.train_data['label'].mean()
            print(f"   原始流失率: {churn_rate:.3f}")
            
            if churn_rate < 0.2:  # 如果流失率过低，进行上采样
                # 手动重采样
                churn_samples = self.train_data[self.train_data['label'] == 1]
                non_churn_samples = self.train_data[self.train_data['label'] == 0]
                
                # 计算需要的流失样本数量（使流失率达到20%）
                target_churn_count = int(len(non_churn_samples) * 0.25)
                if len(churn_samples) < target_churn_count:
                    # 重复采样流失样本
                    repeat_times = target_churn_count // len(churn_samples) + 1
                    churn_samples_upsampled = pd.concat([churn_samples] * repeat_times, ignore_index=True)
                    churn_samples_upsampled = churn_samples_upsampled.head(target_churn_count)
                    
                    # 合并数据
                    self.train_data = pd.concat([non_churn_samples, churn_samples_upsampled], ignore_index=True)
                    print(f"   重采样后流失率: {self.train_data['label'].mean():.3f}")
        
        return self
    
    def feature_engineering(self):
        """特征工程"""
        print("\n=== 特征工程 ===")
        
        # 1. 创建新特征
        print("1. 创建新特征...")
        
        # 年龄分组
        if 'f3' in self.train_data.columns:
            self.train_data['age_group'] = pd.cut(
                self.train_data['f3'], 
                bins=[0, 30, 40, 50, 60, 100], 
                labels=['青年', '中青年', '中年', '中老年', '老年']
            )
            print("   创建年龄分组特征")
        
        # 收入分组
        if 'f5' in self.train_data.columns:
            self.train_data['income_group'] = pd.cut(
                self.train_data['f5'], 
                bins=[0, 5000, 10000, 20000, 50000, float('inf')], 
                labels=['低收入', '中低收入', '中等收入', '中高收入', '高收入']
            )
            print("   创建收入分组特征")
        
        # 存款余额分组
        if 'f17' in self.train_data.columns:
            self.train_data['deposit_group'] = pd.cut(
                self.train_data['f17'], 
                bins=[0, 10000, 50000, 100000, 500000, float('inf')], 
                labels=['低存款', '中低存款', '中等存款', '中高存款', '高存款']
            )
            print("   创建存款分组特征")
        
        # 投资资产分组
        if 'f19' in self.train_data.columns:
            self.train_data['investment_group'] = pd.cut(
                self.train_data['f19'], 
                bins=[0, 10000, 50000, 100000, 500000, float('inf')], 
                labels=['低投资', '中低投资', '中等投资', '中高投资', '高投资']
            )
            print("   创建投资分组特征")
        
        # 2. 比率特征
        if 'f17' in self.train_data.columns and 'f5' in self.train_data.columns:
            self.train_data['deposit_income_ratio'] = self.train_data['f17'] / (self.train_data['f5'] + 1)
            print("   创建存款收入比率特征")
        
        if 'f19' in self.train_data.columns and 'f17' in self.train_data.columns:
            self.train_data['investment_deposit_ratio'] = self.train_data['f19'] / (self.train_data['f17'] + 1)
            print("   创建投资存款比率特征")
        
        # 3. 交互特征
        if 'f1' in self.train_data.columns and 'f3' in self.train_data.columns:
            self.train_data['gender_age'] = self.train_data['f1'].astype(str) + '_' + self.train_data['f3'].astype(str)
            print("   创建性别年龄交互特征")
        
        # 4. 时间特征
        if 'f11' in self.train_data.columns:
            try:
                self.train_data['f11'] = pd.to_datetime(self.train_data['f11'])
                self.train_data['account_age_days'] = (datetime.now() - self.train_data['f11']).dt.days
                self.train_data['account_age_years'] = self.train_data['account_age_days'] / 365.25
                print("   创建账户年龄特征")
            except:
                print("   无法解析开户日期")
        
        # 5. 产品持有特征
        if 'f12' in self.train_data.columns and 'f13' in self.train_data.columns:
            self.train_data['total_products'] = self.train_data['f12'] + self.train_data['f13']
            self.train_data['product_diversity'] = (self.train_data['f12'] > 0).astype(int) + (self.train_data['f13'] > 0).astype(int)
            print("   创建产品多样性特征")
        
        # 6. 手机银行使用特征
        if 'f14' in self.train_data.columns and 'f15' in self.train_data.columns and 'f16' in self.train_data.columns:
            self.train_data['mobile_usage_trend'] = self.train_data['f16'] - self.train_data['f14']
            self.train_data['mobile_usage_intensity'] = self.train_data['f14'] / (self.train_data['f15'] / 3 + 1)
            print("   创建手机银行使用趋势特征")
        
        print(f"   特征工程完成后，特征数量: {self.train_data.shape[1]}")
        
        return self
    
    def encode_features(self):
        """特征编码"""
        print("\n=== 特征编码 ===")
        
        # 1. 标签编码
        print("1. 标签编码...")
        
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        if 'id' in categorical_cols:
            categorical_cols.remove('id')
            
        for col in categorical_cols:
            if col in ['age_group', 'income_group', 'deposit_group', 'investment_group']:
                # 有序分类变量用标签编码
                le = LabelEncoder()
                self.train_data[col] = le.fit_transform(self.train_data[col].astype(str))
                self.label_encoders[col] = le
                print(f"   标签编码: {col}")
            else:
                # 无序分类变量用独热编码
                dummies = pd.get_dummies(self.train_data[col], prefix=col, drop_first=True)
                self.train_data = pd.concat([self.train_data, dummies], axis=1)
                self.train_data.drop(col, axis=1, inplace=True)
                print(f"   独热编码: {col}")
        
        # 2. 数值型特征标准化
        print("2. 数值型特征标准化...")
        
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
        if 'label' in numeric_cols:
            numeric_cols.remove('label')
            
        # 使用MinMaxScaler进行标准化
        scaler = MinMaxScaler()
        self.train_data[numeric_cols] = scaler.fit_transform(self.train_data[numeric_cols])
        self.scalers['numeric'] = scaler
        print(f"   标准化了 {len(numeric_cols)} 个数值型特征")
        
        print(f"   编码完成后，特征数量: {self.train_data.shape[1]}")
        
        return self
    
    def prepare_data(self):
        """准备训练数据"""
        print("\n=== 准备训练数据 ===")
        
        # 分离特征和目标变量
        if 'label' in self.train_data.columns:
            self.X = self.train_data.drop(['id', 'label'], axis=1, errors='ignore')
            self.y = self.train_data['label']
        else:
            self.X = self.train_data.drop(['id'], axis=1, errors='ignore')
            self.y = None
        
        # 处理测试集
        test_data = self.market_test.copy()
        
        # 应用相同的特征工程和编码
        # 这里简化处理，实际应用中需要确保测试集和训练集使用相同的转换
        
        print(f"   特征矩阵形状: {self.X.shape}")
        if self.y is not None:
            print(f"   目标变量形状: {self.y.shape}")
        
        return self
    
    def train_models(self):
        """训练模型"""
        print("\n=== 模型训练 ===")
        
        if self.y is None:
            print("错误: 没有目标变量，无法训练模型")
            return self
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_val.shape}")
        
        # 1. 随机森林
        print("\n1. 训练随机森林...")
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        
        # 评估
        rf_val_pred = rf_model.predict_proba(X_val)[:, 1]
        rf_auc = roc_auc_score(y_val, rf_val_pred)
        print(f"   随机森林验证集AUC: {rf_auc:.4f}")
        
        self.models['random_forest'] = rf_model
        self.feature_importance['random_forest'] = rf_model.feature_importances_
        
        # 2. XGBoost
        print("\n2. 训练XGBoost...")
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        
        # 评估
        xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_val_pred)
        print(f"   XGBoost验证集AUC: {xgb_auc:.4f}")
        
        self.models['xgboost'] = xgb_model
        self.feature_importance['xgboost'] = xgb_model.feature_importances_
        
        # 3. LightGBM
        print("\n3. 训练LightGBM...")
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)
        
        # 评估
        lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
        lgb_auc = roc_auc_score(y_val, lgb_val_pred)
        print(f"   LightGBM验证集AUC: {lgb_auc:.4f}")
        
        self.models['lightgbm'] = lgb_model
        self.feature_importance['lightgbm'] = lgb_model.feature_importances_
        
        # 4. 交叉验证
        print("\n4. 交叉验证...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            cv_scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='roc_auc')
            print(f"   {name} 交叉验证AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self
    
    def feature_importance_analysis(self):
        """特征重要性分析"""
        print("\n=== 特征重要性分析 ===")
        
        # 获取特征名称
        feature_names = self.X.columns.tolist()
        
        # 为每个模型分析特征重要性
        for model_name, importance in self.feature_importance.items():
            print(f"\n{model_name} 模型特征重要性 TOP 20:")
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # 显示TOP 20
            print(importance_df.head(20))
            
            # 可视化TOP 20特征重要性
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('特征重要性')
            plt.title(f'{model_name} - TOP 20 特征重要性')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return self
    
    def predict(self, test_data_path=None):
        """预测"""
        print("\n=== 模型预测 ===")
        
        if test_data_path:
            test_data = pd.read_csv(test_data_path)
        else:
            test_data = self.market_test.copy()
        
        # 这里需要确保测试集和训练集使用相同的特征工程和编码
        # 简化处理，实际应用中需要完整的pipeline
        
        predictions = {}
        for model_name, model in self.models.items():
            try:
                # 简化预测，实际应用中需要完整的特征处理
                pred = model.predict_proba(test_data.select_dtypes(include=[np.number]).drop(['id'], axis=1, errors='ignore'))[:, 1]
                predictions[model_name] = pred
                print(f"   {model_name} 预测完成")
            except Exception as e:
                print(f"   {model_name} 预测失败: {e}")
        
        return predictions
    
    def run_pipeline(self, market_train_path, market_test_path, enriched_train_path=None):
        """运行完整的pipeline"""
        print("开始银行客户流失预测pipeline...")
        
        (self.load_data(market_train_path, market_test_path, enriched_train_path)
         .explore_data()
         .visualize_data()
         .clean_data()
         .feature_engineering()
         .encode_features()
         .prepare_data()
         .train_models()
         .feature_importance_analysis())
        
        print("\n=== Pipeline完成 ===")
        return self

# 主函数
if __name__ == "__main__":
    # 创建预测器实例
    predictor = BankChurnPredictor()
    
    # 运行pipeline（需要提供数据文件路径）
    # predictor.run_pipeline('market_train.csv', 'market_test.csv', 'enriched_train.csv')
    
    print("请提供数据文件路径来运行完整的pipeline")