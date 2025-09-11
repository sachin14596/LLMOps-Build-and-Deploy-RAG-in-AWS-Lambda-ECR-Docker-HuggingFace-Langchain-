import os
import json
import boto3

prompt = (
    "An alpinist celebrating at the summit of a tall snowy mountain at sunrise, "
    "ultra-detailed, cinematic, 4k"
)

region = os.getenv("AWS_REGION", "eu-west-2")
bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)

payload = {
    "inputText": prompt,
    "cfgScale": 7,
    "seed": 0,
    "steps": 40,
    "width": 1024,
    "height": 1024,
}

resp = bedrock.invoke_model(
    modelId="stability.stable-image-core-v1:0",
    accept="image/png",
    contentType="application/json",
    body=json.dumps(payload),
)

image_bytes = resp["body"].read()

os.makedirs("output", exist_ok=True)
with open(os.path.join("output", "generated-img.png"), "wb") as f:
    f.write(image_bytes)
