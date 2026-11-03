-- Отток по способу оплаты.
-- Дополнительно смотрим долю клиентов на помесячном контракте в каждой группе:
-- это помогает понять, связан ли высокий отток с самим способом оплаты
-- или с тем, что в группе больше «необязательных» помесячных клиентов.
SELECT
    PaymentMethod                                   AS payment_method,
    COUNT(*)                                        AS customers,
    SUM(Churn = 'Yes')                              AS churned,
    ROUND(100.0 * AVG(Churn = 'Yes'), 1)            AS churn_rate_pct,
    ROUND(100.0 * AVG(Contract = 'Month-to-month'), 1) AS month_to_month_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;
