-- Views built for the Looker Studio dashboard to sit directly on top of,
-- rather than making every chart re-derive the same aggregations from raw
-- scored_transactions. Run once against the deployed dataset:
--   bq query --use_legacy_sql=false < infra/dashboard_views.sql
-- (with :PROJECT replaced by the real project id; the bq CLI does not
-- expand @-style query parameters inside DDL, so substitute it first,
-- e.g. `sed "s/:PROJECT/$(gcloud config get-value project)/g"`).

CREATE OR REPLACE VIEW `:PROJECT.realtime_scoring.hourly_summary` AS
SELECT
  TIMESTAMP_TRUNC(scored_at, HOUR) AS hour,
  COUNT(*) AS transactions_scored,
  SUM(is_alert) AS alerts_raised,
  SAFE_DIVIDE(SUM(is_alert), COUNT(*)) AS alert_rate,
  SUM(IF(severity = 'High', 1, 0)) AS high_severity,
  SUM(IF(severity = 'Medium', 1, 0)) AS medium_severity,
  SUM(IF(severity = 'Low', 1, 0)) AS low_severity,
  SUM(amount) AS total_amount_scored,
  SUM(IF(is_alert = 1, amount, 0)) AS total_amount_alerted
FROM `:PROJECT.realtime_scoring.scored_transactions`
GROUP BY hour;

CREATE OR REPLACE VIEW `:PROJECT.realtime_scoring.top_alerts` AS
SELECT
  transaction_id,
  scored_at,
  type,
  amount,
  nameOrig,
  nameDest,
  orig_emptied,
  zscore_score,
  iqr_score,
  iforest_score,
  n_methods,
  severity,
  severity_score
FROM `:PROJECT.realtime_scoring.scored_transactions`
WHERE is_alert = 1
ORDER BY severity_score DESC, scored_at DESC;

CREATE OR REPLACE VIEW `:PROJECT.realtime_scoring.detector_agreement` AS
SELECT
  CONCAT(
    IF(zscore_flag, 'zscore ', ''),
    IF(iqr_flag, 'iqr ', ''),
    IF(iforest_flag, 'iforest ', '')
  ) AS detectors_fired,
  n_methods,
  COUNT(*) AS transactions
FROM `:PROJECT.realtime_scoring.scored_transactions`
GROUP BY detectors_fired, n_methods
ORDER BY transactions DESC;
