from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subscription-id", dest="subscription_id", required=True)
    parser.add_argument("--resource-group", dest="resource_group", required=True)
    parser.add_argument("--workspace", dest="workspace", required=True)
    parser.add_argument("--endpoint-name", dest="endpoint_name", default="diabetes-endpoint")
    parser.add_argument("--deployment-name", dest="deployment_name", default="blue")
    parser.add_argument("--model-name", dest="model_name", default="diabetes-model")

    return parser.parse_args()


def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    credential = AzureCliCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def ensure_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        if endpoint.provisioning_state in ("Failed", "Deleting", "Canceled"):
            print(f"Endpoint '{endpoint_name}' is in state '{endpoint.provisioning_state}'. Deleting and recreating...")
            ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
            raise ValueError("Endpoint deleted, recreating")
        print(f"Check: endpoint {endpoint_name} exists")
        return endpoint
    except ValueError:
        pass
    except Exception:
        pass

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Online endpoint for MLflow diabetes model",
        auth_mode="key",
    )
    return ml_client.begin_create_or_update(endpoint).result()


def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
) -> ManagedOnlineDeployment:
    model = ml_client.models.get(name=model_name, label="latest")
    print(f"Deploying model: {model.name} version {model.version}")

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type="Standard_D2as_v4",
        instance_count=1,
    )

    return ml_client.online_deployments.begin_create_or_update(deployment).result()


def set_traffic_to_deployment(ml_client: MLClient, endpoint_name: str, deployment_name: str) -> None:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.begin_create_or_update(endpoint).result()


def main() -> None:
    args = parse_args()

    print("Connecting to Azure Machine Learning workspace...")
    ml_client = get_ml_client(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace=args.workspace,
    )

    print(f"Ensuring online endpoint '{args.endpoint_name}' exists...")
    endpoint = ensure_endpoint(ml_client, args.endpoint_name)
    print(f"Using endpoint: {endpoint.name}")

    print(f"Creating or updating deployment '{args.deployment_name}'...")
    deployment = create_or_update_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint.name,
        deployment_name=args.deployment_name,
        model_name=args.model_name,
    )
    print(f"Deployment state: {deployment.provisioning_state}")

    print("Directing 100% of traffic to the deployment...")
    set_traffic_to_deployment(ml_client, endpoint.name, args.deployment_name)

    endpoint = ml_client.online_endpoints.get(name=endpoint.name)
    print(f"Deployment complete. Scoring URI: {endpoint.scoring_uri}")


if __name__ == "__main__":
    main()
