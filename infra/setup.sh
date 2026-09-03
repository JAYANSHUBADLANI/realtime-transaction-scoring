#!/usr/bin/env bash
# Deploys the streaming scorer for real: Artifact Registry, Cloud Run
# (private, Pub/Sub-only), a dedicated push-auth service account,
# BigQuery, a Pub/Sub topic, and a push subscription wired to the
# deployed service. Idempotent: every gcloud command below is safe to
# re-run.
#
# Prerequisites: `python main.py` has produced artifacts/ (the frozen
# reference this image bakes in), gcloud is authenticated, and the
# project below has billing enabled.
set -euo pipefail

if [ -z "${1:-}" ] && [ -z "${GCP_PROJECT_ID:-}" ]; then
  echo "Usage: $0 <gcp-project-id>   (or set GCP_PROJECT_ID)" >&2
  exit 1
fi

PROJECT_ID="${1:-$GCP_PROJECT_ID}"
REGION="us-central1"
REPO_NAME="realtime-transaction-scoring"
IMAGE_TAG="1.0.0"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/scorer:${IMAGE_TAG}"
SERVICE_NAME="realtime-transaction-scorer"
BQ_DATASET="realtime_scoring"
TOPIC_NAME="transaction-events"
SUBSCRIPTION_NAME="transaction-events-push-sub"
PUSH_SA="pubsub-push-invoker"
PUSH_SA_EMAIL="${PUSH_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

cd "$(dirname "$0")/.."

echo "== 1. Artifact Registry repo =="
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker --location="$REGION" \
  --project="$PROJECT_ID" \
  --description="Real-time transaction scoring service images" \
  || echo "  (already exists, continuing)"

echo "== 2. Build and push the image (linux/amd64: Cloud Run rejects arm64) =="
docker build --platform=linux/amd64 -f service/Dockerfile -t "$IMAGE" .
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet --project="$PROJECT_ID"
docker push "$IMAGE"

echo "== 3. Dedicated service account for Pub/Sub push authentication =="
gcloud iam service-accounts create "$PUSH_SA" \
  --display-name="Pub/Sub push auth for $SERVICE_NAME" \
  --project="$PROJECT_ID" \
  || echo "  (already exists, continuing)"

echo "== 4. BigQuery dataset (table is created by the service on first write) =="
bq --project_id="$PROJECT_ID" mk --dataset --location="$REGION" \
  "${PROJECT_ID}:${BQ_DATASET}" \
  || echo "  (already exists, continuing)"

echo "== 5. Deploy to Cloud Run, private (no --allow-unauthenticated): =="
echo "     only Pub/Sub, via the push-auth service account below, may call this."
gcloud run deploy "$SERVICE_NAME" \
  --image="$IMAGE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --no-allow-unauthenticated \
  --memory=512Mi \
  --max-instances=2 \
  --set-env-vars="BQ_DATASET=${BQ_DATASET},BQ_PROJECT=${PROJECT_ID}"

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)')
echo "  Deployed: $SERVICE_URL"

echo "== 6. Let the push-auth SA invoke this specific Cloud Run service =="
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
  --region="$REGION" --project="$PROJECT_ID" \
  --member="serviceAccount:${PUSH_SA_EMAIL}" \
  --role="roles/run.invoker"

echo "== 7. Let Pub/Sub's own service agent mint tokens as the push-auth SA =="
PUBSUB_SERVICE_AGENT=$(gcloud projects get-iam-policy "$PROJECT_ID" \
  --flatten="bindings[].members" --filter="bindings.role:roles/pubsub.serviceAgent" \
  --format="value(bindings.members)" | head -1)
gcloud iam service-accounts add-iam-policy-binding "$PUSH_SA_EMAIL" \
  --project="$PROJECT_ID" \
  --member="$PUBSUB_SERVICE_AGENT" \
  --role="roles/iam.serviceAccountTokenCreator"

echo "== 8. Let the service's own runtime identity write/query BigQuery =="
RUNTIME_SA=$(gcloud run services describe "$SERVICE_NAME" \
  --region="$REGION" --project="$PROJECT_ID" \
  --format='value(spec.template.spec.serviceAccountName)')
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/bigquery.dataEditor" --quiet
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/bigquery.jobUser" --quiet

echo "== 9. Pub/Sub topic =="
gcloud pubsub topics create "$TOPIC_NAME" --project="$PROJECT_ID" \
  || echo "  (already exists, continuing)"

echo "== 10. Push subscription, OIDC-authenticated, pointed at the deployed URL =="
gcloud pubsub subscriptions create "$SUBSCRIPTION_NAME" \
  --topic="$TOPIC_NAME" \
  --project="$PROJECT_ID" \
  --push-endpoint="${SERVICE_URL}/push" \
  --push-auth-service-account="$PUSH_SA_EMAIL" \
  --ack-deadline=30 \
  || echo "  (already exists, continuing)"

echo
echo "Done. Service URL: $SERVICE_URL"
echo "Replay real traffic with:"
echo "  python publisher/replay.py --project $PROJECT_ID --topic $TOPIC_NAME --n 2000"
