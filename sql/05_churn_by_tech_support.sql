-- Отток в зависимости от подключённой техподдержки (TechSupport).
-- 'No internet service' — клиенты без интернета, у них этой услуги быть не может.
SELECT
    TechSupport                                     AS tech_support,
    COUNT(*)                                        AS customers,
    SUM(Churn = 'Yes')                              AS churned,
    ROUND(100.0 * AVG(Churn = 'Yes'), 1)            AS churn_rate_pct
FROM customers
GROUP BY TechSupport
ORDER BY churn_rate_pct DESC;
