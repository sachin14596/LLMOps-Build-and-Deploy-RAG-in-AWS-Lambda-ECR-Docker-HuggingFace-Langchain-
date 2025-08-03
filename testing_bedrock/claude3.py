import json
import boto3

# Region as per your ARN
bedrock = boto3.client("bedrock-runtime", region_name="eu-north-1")

prompt = "You are a cricket expert. Just tell me when RCB will win the IPL?"

# Inference Profile ARN for Claude 3.7 Sonnet (your provided ARN)
INFERENCE_PROFILE_ARN = "arn:aws:bedrock:eu-north-1:137756268106:inference-profile/eu.anthropic.claude-3-7-sonnet-20250219-v1:0"

payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.999,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
}

response = bedrock.invoke_model(
    body=json.dumps(payload),
    inferenceProfileArn=INFERENCE_PROFILE_ARN,  # using ARN
    accept="application/json",
    contentType="application/json",
)

resp_body = json.loads(response["body"].read())
print(resp_body["content"][0]["text"])
