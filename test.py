from google.cloud import aiplatform
from datetime import datetime

PROJECT_ID = "charged-sled-459818-s1"  # ✅ Replace with your actual project ID
REGION = "us-central1"
MODEL_NAME = "gemma-3-1b-it-mg-one-click-deploy"
ENDPOINT_NAME = f"{MODEL_NAME}-endpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}"

aiplatform.init(project=PROJECT_ID, location=REGION)

model = aiplatform.Model.upload(
    display_name=MODEL_NAME,
    model_garden_source_model_name="publishers/google/models/gemma-3-1b-it-mg-one-click-deploy",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/vertex-vision-prediction/pytorch-gpu.1-13:latest",
    serving_container_predict_route="/generate",
    serving_container_health_route="/ping",
    serving_container_ports=[8080],
    serving_container_environment_variables={"MODEL_ID": MODEL_NAME}
)

endpoint = model.deploy(
    deployed_model_display_name=MODEL_NAME,
    machine_type="g2-standard-8",  # ✅ Use this machine type
    accelerator_type="NVIDIA_L4",
    accelerator_count=1
)

print(f"✅ Model deployed successfully.")
print(f"🔗 Endpoint ID: {endpoint.name}")
