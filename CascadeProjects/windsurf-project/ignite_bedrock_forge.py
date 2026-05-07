#!/usr/bin/env python3
"""
ignite_bedrock_forge.py — AWS Bedrock Distillation Pipeline
============================================================
Phase 2 of YDN-OMEGA: Teacher-Student distillation via Amazon Bedrock.
Teacher: amazon.nova-pro-v1:0:300k (300B context Nova Pro)
Student: amazon.nova-micro-v1:0:128k (lightweight Nova Micro)
Dataset: s3://wevdev/omega-fuel/weaver_omega_fuel.jsonl
"""

import boto3
import json
import time
import sys
from datetime import datetime

REGION = "us-east-1"
ACCOUNT_ID = "404870839825"

TEACHER_MODEL = "amazon.nova-pro-v1:0"
STUDENT_MODEL = "amazon.nova-micro-v1:0:128k"

S3_BUCKET = "wevdev"
S3_TRAINING_KEY = "omega-fuel/weaver_omega_fuel.jsonl"
S3_OUTPUT_PREFIX = "omega-fuel/output/"

JOB_NAME = f"weaver-omega-distill-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
CUSTOM_MODEL_NAME = "weaver-omega-manifold-v1"

ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/BedrockCustomizationRole"

HYPERPARAMS = {
    "epochCount": "3",
    "batchSize": "4",
    "learningRate": "0.00002",
    "learningRateWarmupSteps": "10",
}


def main():
    bedrock = boto3.client("bedrock", region_name=REGION)

    print(f"{'=' * 60}")
    print(f"  YDN-OMEGA Bedrock Forge — Distillation Pipeline")
    print(f"{'=' * 60}")
    print(f"  Teacher : {TEACHER_MODEL}")
    print(f"  Student : {STUDENT_MODEL}")
    print(f"  Dataset : s3://{S3_BUCKET}/{S3_TRAINING_KEY}")
    print(f"  Job Name: {JOB_NAME}")
    print(f"  Role    : {ROLE_ARN}")
    print()

    # Verify dataset exists in S3
    s3 = boto3.client("s3", region_name=REGION)
    try:
        resp = s3.head_object(Bucket=S3_BUCKET, Key=S3_TRAINING_KEY)
        size_kb = resp["ContentLength"] / 1024
        print(f"  [OK] Dataset verified: {size_kb:.1f} KB in S3")
    except Exception as e:
        print(f"  [FAIL] Dataset not found in S3: {e}")
        sys.exit(1)

    # Create the distillation job
    print(f"\n  Igniting Bedrock Forge...")
    try:
        response = bedrock.create_model_customization_job(
            jobName=JOB_NAME,
            customModelName=CUSTOM_MODEL_NAME,
            roleArn=ROLE_ARN,
            baseModelIdentifier=STUDENT_MODEL,
            customizationType="DISTILLATION",
            customizationConfig={
                "distillationConfig": {
                    "teacherModelConfig": {
                        "teacherModelIdentifier": TEACHER_MODEL,
                        "maxResponseLengthForInference": 4096,
                    }
                }
            },
            trainingDataConfig={
                "s3Uri": f"s3://{S3_BUCKET}/{S3_TRAINING_KEY}"
            },
            outputDataConfig={
                "s3Uri": f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}"
            },
            customModelTags=[
                {"key": "project", "value": "weaver-omega"},
                {"key": "type", "value": "distillation"},
                {"key": "teacher", "value": TEACHER_MODEL},
            ],
        )
        job_arn = response["jobArn"]
        print(f"  [OK] Distillation job created!")
        print(f"  Job ARN: {job_arn}")
    except bedrock.exceptions.ValidationException as e:
        print(f"  [VALIDATION ERROR] {e}")
        sys.exit(1)
    except bedrock.exceptions.AccessDeniedException as e:
        print(f"  [ACCESS DENIED] {e}")
        print(f"\n  The IAM role needs bedrock.amazonaws.com in its trust policy.")
        print(f"  Create a role with:")
        print(f'    Principal: {{"Service": "bedrock.amazonaws.com"}}')
        print(f"    Permissions: s3:GetObject, s3:PutObject on s3://{S3_BUCKET}/*")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        sys.exit(1)

    # Monitor loop
    print(f"\n  Monitoring job status (checking every 60s)...")
    print(f"  {'─' * 50}")

    while True:
        try:
            status_resp = bedrock.get_model_customization_job(jobIdentifier=job_arn)
            status = status_resp["status"]
            now = datetime.utcnow().strftime("%H:%M:%S UTC")

            if status == "InProgress":
                print(f"  [{now}] Status: InProgress — forge active")
                print(f"  [OK] Distillation is running. The forge burns.")
                print(f"\n  Monitor with:")
                print(f"    aws bedrock get-model-customization-job --job-identifier {job_arn} --region {REGION}")
                break
            elif status == "Completed":
                print(f"  [{now}] Status: Completed!")
                output_model = status_resp.get("outputModelArn", "N/A")
                print(f"  Output Model ARN: {output_model}")
                print(f"\n  The forge has spoken. The student carries the teacher's fire.")
                break
            elif status in ("Failed", "Stopped"):
                print(f"  [{now}] Status: {status}")
                failure = status_resp.get("failureMessage", "No details")
                print(f"  Failure: {failure}")
                sys.exit(1)
            else:
                print(f"  [{now}] Status: {status} — waiting...")

        except Exception as e:
            print(f"  [MONITOR ERROR] {e}")

        time.sleep(60)


if __name__ == "__main__":
    main()
