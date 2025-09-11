import json
import os
import boto3

# Match client region to the profile/region you use (default eu-west-2)
region = os.getenv("AWS_REGION", "eu-west-2")
bedrock = boto3.client("bedrock-runtime", region_name=region)

prompt = "You are a cricket expert. Just tell me when RCB will win the IPL?"

# If your account requires an inference profile for Claude 3.7 in this region,
# put the eu-west-2 profile ARN below. Otherwise, prefer the app flow above.
INFERENCE_PROFILE_ARN = os.getenv(
    "BEDROCK_CLAUDE37_PROFILE_ARN",
    "arn:aws:bedrock:eu-west-2:000000000000:inference-profile/eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
)

payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 256,
    "temperature": 0.7,
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
}

response = bedrock.invoke_model(
    body=json.dumps(payload),
    inferenceProfileArn=INFERENCE_PROFILE_ARN,  # comment this if you invoke by modelId instead
    accept="application/json",
    contentType="application/json",
)

resp_body = json.loads(response["body"].read())
print(resp_body["content"][0]["text"])
