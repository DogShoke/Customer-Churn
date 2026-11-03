-- Отток по сроку обслуживания (tenure, в месяцах).
-- Группируем клиентов в «корзины» по годам, чтобы увидеть, как отток меняется
-- с возрастом клиента. tenure = 0 — клиенты, которые только что подключились.
SELECT
    CASE
        WHEN tenure <= 12 THEN '1) 0-12 мес'
        WHEN tenure <= 24 THEN '2) 13-24 мес'
        WHEN tenure <= 48 THEN '3) 25-48 мес'
        ELSE                   '4) 49-72 мес'
    END                                             AS tenure_group,
    COUNT(*)                                        AS customers,
    SUM(Churn = 'Yes')                              AS churned,
    ROUND(100.0 * AVG(Churn = 'Yes'), 1)            AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                   AS avg_monthly_charges
FROM customers
GROUP BY tenure_group
ORDER BY tenure_group;
