#!/usr/bin/env bash
# Usage: PROJECT=my-gcp-project GEMINI_API_KEY=... ./deploy.sh
set -euo pipefail
: "${PROJECT:?set PROJECT}"; REGION=${REGION:-asia-south1}
gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com earthengine.googleapis.com aiplatform.googleapis.com
gcloud run deploy agrinet --source . --region "$REGION" --allow-unauthenticated --memory 1Gi \
  --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY:-},EE_PROJECT=$PROJECT,GOOGLE_CLOUD_PROJECT=$PROJECT"
# One-time: register the Cloud Run service account for Earth Engine at https://code.earthengine.google.com/register
