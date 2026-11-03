-- Отток по типу интернета.
-- Средний чек показываем рядом, потому что оптоволокно (Fiber optic) — самый дорогой тариф,
-- и важно не перепутать «уходят из-за технологии» и «уходят из-за цены».
SELECT
    InternetService                                 AS internet_service,
    COUNT(*)                                        AS customers,
    SUM(Churn = 'Yes')                              AS churned,
    ROUND(100.0 * AVG(Churn = 'Yes'), 1)            AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                   AS avg_monthly_charges
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;
