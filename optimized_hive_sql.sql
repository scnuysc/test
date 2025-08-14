-- 优化的Hive SQL查询 - 银行客户流失预测特征工程
-- 增加了更多业务相关的特征，包括时间序列特征、行为模式特征等

CREATE TABLE aplthive_ohzdl_0mgk1_d.enriched_train_optimized AS
SELECT 
    t.id, t.label, t.f1, t.f2, t.f3, t.f4, t.f5, t.f6, t.f7, t.f8, t.f9, t.f10, t.f11, 
    t.f12, t.f13, t.f14, t.f15, t.f16, t.f17, t.f18, t.f19, t.f20,
    
    -- 基础交易统计特征
    COUNT(r.id) AS total_transactions,
    COUNT(DISTINCT r.f13) AS active_transaction_days,
    SUM(CASE WHEN r.f4 = 'C' THEN 1 ELSE 0 END) AS inflow_count,
    SUM(CASE WHEN r.f4 = 'D' THEN 1 ELSE 0 END) AS outflow_count,
    SUM(CASE WHEN r.f4 = 'C' THEN r.f11 ELSE 0 END) AS total_inflow,
    SUM(CASE WHEN r.f4 = 'D' THEN r.f11 ELSE 0 END) AS total_outflow,
    
    -- 交易金额统计特征
    AVG(r.f11) AS avg_transaction_amount,
    STDDEV(r.f11) AS std_transaction_amount,
    MAX(r.f11) AS max_transaction_amount,
    MIN(r.f11) AS min_transaction_amount,
    PERCENTILE_APPROX(r.f11, 0.5) AS median_transaction_amount,
    PERCENTILE_APPROX(r.f11, 0.75) AS q75_transaction_amount,
    PERCENTILE_APPROX(r.f11, 0.25) AS q25_transaction_amount,
    
    -- 资金流动特征
    CASE WHEN SUM(CASE WHEN r.f4 = 'C' THEN r.f11 ELSE 0 END) > 0 
         THEN SUM(CASE WHEN r.f4 = 'D' THEN r.f11 ELSE 0 END) / SUM(CASE WHEN r.f4 = 'C' THEN r.f11 ELSE 0 END)
         ELSE 0 END AS outflow_inflow_ratio,
    
    SUM(CASE WHEN r.f4 = 'C' THEN r.f11 ELSE 0 END) - SUM(CASE WHEN r.f4 = 'D' THEN r.f11 ELSE 0 END) AS net_flow,
    
    -- 主要交易类型特征 (基于原SQL中的类型)
    SUM(CASE WHEN r.f1 = '84' THEN 1 ELSE 0 END) AS quick_payment_count,
    SUM(CASE WHEN r.f1 = '8101' THEN 1 ELSE 0 END) AS interbank_transfer_count,
    SUM(CASE WHEN r.f1 = '31' THEN 1 ELSE 0 END) AS type_31_count,
    SUM(CASE WHEN r.f1 = '85' THEN 1 ELSE 0 END) AS type_85_count,
    SUM(CASE WHEN r.f1 = '16' THEN 1 ELSE 0 END) AS type_16_count,
    SUM(CASE WHEN r.f1 = '6' THEN 1 ELSE 0 END) AS transfer_income_count,
    SUM(CASE WHEN r.f1 = '5383' THEN 1 ELSE 0 END) AS type_5383_count,
    
    -- 主要交易类型金额特征
    SUM(CASE WHEN r.f1 = '84' THEN r.f11 ELSE 0 END) AS quick_payment_amount,
    SUM(CASE WHEN r.f1 = '8101' THEN r.f11 ELSE 0 END) AS interbank_transfer_amount,
    SUM(CASE WHEN r.f1 = '6' THEN r.f11 ELSE 0 END) AS transfer_income_amount,
    
    -- 渠道使用特征
    SUM(CASE WHEN r.f16 = '02' THEN 1 ELSE 0 END) AS online_banking_count,
    SUM(CASE WHEN r.f16 = '10' THEN 1 ELSE 0 END) AS mobile_banking_count,
    SUM(CASE WHEN r.f16 = '01' THEN 1 ELSE 0 END) AS web_banking_count,
    SUM(CASE WHEN r.f16 = '00' THEN 1 ELSE 0 END) AS counter_count,
    SUM(CASE WHEN r.f16 = '99' THEN 1 ELSE 0 END) AS other_channel_count,
    
    -- 渠道多样性
    COUNT(DISTINCT r.f16) AS channel_diversity,
    
    -- 时间特征
    MAX(r.f13) AS last_transaction_date,
    MIN(r.f13) AS first_transaction_date,
    DATEDIFF(MAX(r.f13), MIN(r.f13)) AS transaction_span_days,
    
    -- 最近活跃度特征 (假设观测点是2025-04-30)
    SUM(CASE WHEN DATEDIFF('2025-04-30', r.f13) <= 7 THEN 1 ELSE 0 END) AS transactions_last_7days,
    SUM(CASE WHEN DATEDIFF('2025-04-30', r.f13) <= 30 THEN 1 ELSE 0 END) AS transactions_last_30days,
    SUM(CASE WHEN DATEDIFF('2025-04-30', r.f13) <= 7 THEN r.f11 ELSE 0 END) AS amount_last_7days,
    SUM(CASE WHEN DATEDIFF('2025-04-30', r.f13) <= 30 THEN r.f11 ELSE 0 END) AS amount_last_30days,
    
    -- 交易频率特征
    CASE WHEN COUNT(DISTINCT r.f13) > 0 
         THEN COUNT(r.id) / COUNT(DISTINCT r.f13) 
         ELSE 0 END AS avg_transactions_per_day,
    
    -- 大额交易特征 (定义大额为金额>10000)
    SUM(CASE WHEN r.f11 > 10000 THEN 1 ELSE 0 END) AS large_transaction_count,
    SUM(CASE WHEN r.f11 > 10000 THEN r.f11 ELSE 0 END) AS large_transaction_amount,
    
    -- 小额交易特征 (定义小额为金额<100)
    SUM(CASE WHEN r.f11 < 100 THEN 1 ELSE 0 END) AS small_transaction_count,
    
    -- 交易模式特征
    SUM(CASE WHEN r.f16 = '10' AND r.f4 = 'D' THEN 1 ELSE 0 END) AS mobile_outflow_count,
    SUM(CASE WHEN r.f16 = '10' AND r.f4 = 'C' THEN 1 ELSE 0 END) AS mobile_inflow_count,
    
    -- 支付类交易特征 (基于交易描述包含"支付")
    SUM(CASE WHEN r.f2 LIKE '%支付%' THEN 1 ELSE 0 END) AS payment_transaction_count,
    SUM(CASE WHEN r.f2 LIKE '%支付%' THEN r.f11 ELSE 0 END) AS payment_transaction_amount,
    
    -- 转账类交易特征
    SUM(CASE WHEN r.f2 LIKE '%转账%' THEN 1 ELSE 0 END) AS transfer_transaction_count,
    SUM(CASE WHEN r.f2 LIKE '%转账%' THEN r.f11 ELSE 0 END) AS transfer_transaction_amount,
    
    -- 月度交易趋势特征
    SUM(CASE WHEN MONTH(r.f13) = 2 THEN 1 ELSE 0 END) AS feb_transaction_count,
    SUM(CASE WHEN MONTH(r.f13) = 3 THEN 1 ELSE 0 END) AS mar_transaction_count,
    SUM(CASE WHEN MONTH(r.f13) = 4 THEN 1 ELSE 0 END) AS apr_transaction_count,
    SUM(CASE WHEN MONTH(r.f13) = 2 THEN r.f11 ELSE 0 END) AS feb_transaction_amount,
    SUM(CASE WHEN MONTH(r.f13) = 3 THEN r.f11 ELSE 0 END) AS mar_transaction_amount,
    SUM(CASE WHEN MONTH(r.f13) = 4 THEN r.f11 ELSE 0 END) AS apr_transaction_amount

FROM aplthive_ohzdl_0mgk1_d.market_train t
LEFT JOIN aplthive_ohzdl_0mgk1_d.risk_train r ON t.id = r.id
GROUP BY t.id, t.label, t.f1, t.f2, t.f3, t.f4, t.f5, t.f6, t.f7, t.f8, t.f9, t.f10, 
         t.f11, t.f12, t.f13, t.f14, t.f15, t.f16, t.f17, t.f18, t.f19, t.f20;

-- 对测试集执行同样的操作
CREATE TABLE aplthive_ohzdl_0mgk1_d.enriched_test_optimized AS
SELECT 
    t.id, t.f1, t.f2, t.f3, t.f4, t.f5, t.f6, t.f7, t.f8, t.f9, t.f10, t.f11, 
    t.f12, t.f13, t.f14, t.f15, t.f16, t.f17, t.f18, t.f19, t.f20,
    
    -- 基础交易统计特征
    COUNT(r.id) AS total_transactions,
    COUNT(DISTINCT r.f13) AS active_transaction_days,
    SUM(CASE WHEN r.f4 = 'C' THEN 1 ELSE 0 END) AS inflow_count,
    SUM(CASE WHEN r.f4 = 'D' THEN 1 ELSE 0 END) AS outflow_count,
    SUM(CASE WHEN r.f4 = 'C' THEN r.f11 ELSE 0 END) AS total_inflow,
    SUM(CASE WHEN r.f4 = 'D' THEN r.f11 ELSE 0 END) AS total_outflow,
    
    -- 交易金额统计特征
    AVG(r.f11) AS avg_transaction_amount,
    STDDEV(r.f11) AS std_transaction_amount,
    MAX(r.f11) AS max_transaction_amount,
    MIN(r.f11) AS min_transaction_amount,
    PERCENTILE_APPROX(r.f11, 0.5) AS median_transaction_amount,
    PERCENTILE_APPROX(r.f11, 0.75) AS q75_transaction_amount,
    PERCENTILE_APPROX(r.f11, 0.25) AS q25_transaction_amount,
    
    -- 资金流动特征
    CASE WHEN SUM(CASE WHEN r.f4 = 'C' THEN r.f11 ELSE 0 END) > 0 
         THEN SUM(CASE WHEN r.f4 = 'D' THEN r.f11 ELSE 0 END) / SUM(CASE WHEN r.f4 = 'C' THEN r.f11 ELSE 0 END)
         ELSE 0 END AS outflow_inflow_ratio,
    
    SUM(CASE WHEN r.f4 = 'C' THEN r.f11 ELSE 0 END) - SUM(CASE WHEN r.f4 = 'D' THEN r.f11 ELSE 0 END) AS net_flow,
    
    -- 主要交易类型特征
    SUM(CASE WHEN r.f1 = '84' THEN 1 ELSE 0 END) AS quick_payment_count,
    SUM(CASE WHEN r.f1 = '8101' THEN 1 ELSE 0 END) AS interbank_transfer_count,
    SUM(CASE WHEN r.f1 = '31' THEN 1 ELSE 0 END) AS type_31_count,
    SUM(CASE WHEN r.f1 = '85' THEN 1 ELSE 0 END) AS type_85_count,
    SUM(CASE WHEN r.f1 = '16' THEN 1 ELSE 0 END) AS type_16_count,
    SUM(CASE WHEN r.f1 = '6' THEN 1 ELSE 0 END) AS transfer_income_count,
    SUM(CASE WHEN r.f1 = '5383' THEN 1 ELSE 0 END) AS type_5383_count,
    
    -- 主要交易类型金额特征
    SUM(CASE WHEN r.f1 = '84' THEN r.f11 ELSE 0 END) AS quick_payment_amount,
    SUM(CASE WHEN r.f1 = '8101' THEN r.f11 ELSE 0 END) AS interbank_transfer_amount,
    SUM(CASE WHEN r.f1 = '6' THEN r.f11 ELSE 0 END) AS transfer_income_amount,
    
    -- 渠道使用特征
    SUM(CASE WHEN r.f16 = '02' THEN 1 ELSE 0 END) AS online_banking_count,
    SUM(CASE WHEN r.f16 = '10' THEN 1 ELSE 0 END) AS mobile_banking_count,
    SUM(CASE WHEN r.f16 = '01' THEN 1 ELSE 0 END) AS web_banking_count,
    SUM(CASE WHEN r.f16 = '00' THEN 1 ELSE 0 END) AS counter_count,
    SUM(CASE WHEN r.f16 = '99' THEN 1 ELSE 0 END) AS other_channel_count,
    
    -- 渠道多样性
    COUNT(DISTINCT r.f16) AS channel_diversity,
    
    -- 时间特征
    MAX(r.f13) AS last_transaction_date,
    MIN(r.f13) AS first_transaction_date,
    DATEDIFF(MAX(r.f13), MIN(r.f13)) AS transaction_span_days,
    
    -- 最近活跃度特征
    SUM(CASE WHEN DATEDIFF('2025-04-30', r.f13) <= 7 THEN 1 ELSE 0 END) AS transactions_last_7days,
    SUM(CASE WHEN DATEDIFF('2025-04-30', r.f13) <= 30 THEN 1 ELSE 0 END) AS transactions_last_30days,
    SUM(CASE WHEN DATEDIFF('2025-04-30', r.f13) <= 7 THEN r.f11 ELSE 0 END) AS amount_last_7days,
    SUM(CASE WHEN DATEDIFF('2025-04-30', r.f13) <= 30 THEN r.f11 ELSE 0 END) AS amount_last_30days,
    
    -- 交易频率特征
    CASE WHEN COUNT(DISTINCT r.f13) > 0 
         THEN COUNT(r.id) / COUNT(DISTINCT r.f13) 
         ELSE 0 END AS avg_transactions_per_day,
    
    -- 大额交易特征
    SUM(CASE WHEN r.f11 > 10000 THEN 1 ELSE 0 END) AS large_transaction_count,
    SUM(CASE WHEN r.f11 > 10000 THEN r.f11 ELSE 0 END) AS large_transaction_amount,
    
    -- 小额交易特征
    SUM(CASE WHEN r.f11 < 100 THEN 1 ELSE 0 END) AS small_transaction_count,
    
    -- 交易模式特征
    SUM(CASE WHEN r.f16 = '10' AND r.f4 = 'D' THEN 1 ELSE 0 END) AS mobile_outflow_count,
    SUM(CASE WHEN r.f16 = '10' AND r.f4 = 'C' THEN 1 ELSE 0 END) AS mobile_inflow_count,
    
    -- 支付类交易特征
    SUM(CASE WHEN r.f2 LIKE '%支付%' THEN 1 ELSE 0 END) AS payment_transaction_count,
    SUM(CASE WHEN r.f2 LIKE '%支付%' THEN r.f11 ELSE 0 END) AS payment_transaction_amount,
    
    -- 转账类交易特征
    SUM(CASE WHEN r.f2 LIKE '%转账%' THEN 1 ELSE 0 END) AS transfer_transaction_count,
    SUM(CASE WHEN r.f2 LIKE '%转账%' THEN r.f11 ELSE 0 END) AS transfer_transaction_amount,
    
    -- 月度交易趋势特征
    SUM(CASE WHEN MONTH(r.f13) = 2 THEN 1 ELSE 0 END) AS feb_transaction_count,
    SUM(CASE WHEN MONTH(r.f13) = 3 THEN 1 ELSE 0 END) AS mar_transaction_count,
    SUM(CASE WHEN MONTH(r.f13) = 4 THEN 1 ELSE 0 END) AS apr_transaction_count,
    SUM(CASE WHEN MONTH(r.f13) = 2 THEN r.f11 ELSE 0 END) AS feb_transaction_amount,
    SUM(CASE WHEN MONTH(r.f13) = 3 THEN r.f11 ELSE 0 END) AS mar_transaction_amount,
    SUM(CASE WHEN MONTH(r.f13) = 4 THEN r.f11 ELSE 0 END) AS apr_transaction_amount

FROM aplthive_ohzdl_0mgk1_d.market_test t
LEFT JOIN aplthive_ohzdl_0mgk1_d.risk_test r ON t.id = r.id
GROUP BY t.id, t.f1, t.f2, t.f3, t.f4, t.f5, t.f6, t.f7, t.f8, t.f9, t.f10, 
         t.f11, t.f12, t.f13, t.f14, t.f15, t.f16, t.f17, t.f18, t.f19, t.f20;