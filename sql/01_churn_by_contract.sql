-- Отток по типу контракта.
-- Churn хранится как 'Yes'/'No', поэтому долю ушедших считаем как AVG от условия:
-- (Churn = 'Yes') даёт 1 или 0, среднее от этих единиц и нулей = доля ушедших.
SELECT
    Contract                                        AS contract,
    COUNT(*)                                        AS customers,
    SUM(Churn = 'Yes')                              AS churned,
    ROUND(100.0 * AVG(Churn = 'Yes'), 1)            AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;
