"""
deploy.py — Kubernetes Deployment Script for Fluidstack GPU Dashboard
=====================================================================
Applies dashboard.yaml (Deployment + Service + NVIDIA device plugin)
to a k8s cluster (Fluidstack managed or minikube). Fetches pod metrics.
"""

import os
import sys
import time
import yaml
from pathlib import Path

try:
    from kubernetes import client, config, utils
    from kubernetes.client.rest import ApiException
except ImportError:
    print("❌ kubernetes package not installed. Run: pip install kubernetes")
    sys.exit(1)

YAML_PATH = Path(__file__).parent / "dashboard.yaml"
NAMESPACE = os.getenv("K8S_NAMESPACE", "default")
CONTEXT = os.getenv("K8S_CONTEXT", "")  # e.g., "fluidstack-cluster" or "minikube"


def load_kube_config():
    """Load kubeconfig, preferring in-cluster if available."""
    try:
        config.load_incluster_config()
        print("✅ Loaded in-cluster kubeconfig")
    except config.ConfigException:
        try:
            if CONTEXT:
                config.load_kube_config(context=CONTEXT)
                print(f"✅ Loaded kubeconfig with context: {CONTEXT}")
            else:
                config.load_kube_config()
                print("✅ Loaded default kubeconfig")
        except Exception as e:
            print(f"❌ Cannot load kubeconfig: {e}")
            sys.exit(1)


def apply_yaml_manifests():
    """Read dashboard.yaml and apply all documents to the cluster."""
    if not YAML_PATH.exists():
        print(f"❌ {YAML_PATH} not found")
        sys.exit(1)

    print(f"\n📄 Applying manifests from {YAML_PATH}...")
    k8s_client = client.ApiClient()

    with open(YAML_PATH) as f:
        docs = list(yaml.safe_load_all(f))

    for i, doc in enumerate(docs):
        if doc is None:
            continue
        kind = doc.get("kind", "Unknown")
        name = doc.get("metadata", {}).get("name", "unnamed")
        try:
            utils.create_from_dict(k8s_client, doc, namespace=NAMESPACE)
            print(f"  ✅ Created {kind}/{name}")
        except ApiException as e:
            if e.status == 409:
                print(f"  ⚠️  {kind}/{name} already exists (skipping)")
            else:
                print(f"  ❌ Failed {kind}/{name}: {e.reason}")
        except Exception as e:
            print(f"  ❌ Failed {kind}/{name}: {e}")


def wait_for_pods(label_selector="app=fluidstack-gpu-dashboard", timeout=120):
    """Wait until at least one pod is Running."""
    v1 = client.CoreV1Api()
    print(f"\n⏳ Waiting for pods ({label_selector}) in namespace '{NAMESPACE}'...")
    start = time.time()
    while time.time() - start < timeout:
        pods = v1.list_namespaced_pod(NAMESPACE, label_selector=label_selector)
        running = [p for p in pods.items if p.status.phase == "Running"]
        if running:
            for p in running:
                print(f"  ✅ Pod {p.metadata.name} is Running")
            return running
        pending = [p for p in pods.items if p.status.phase == "Pending"]
        if pending:
            print(f"  ⏳ {len(pending)} pod(s) Pending...")
        time.sleep(5)
    print(f"  ❌ Timeout after {timeout}s waiting for pods")
    return []


def fetch_pod_metrics():
    """
    Fetch pod resource metrics using the Metrics API (requires metrics-server).
    Falls back gracefully if metrics-server is not available.
    """
    print("\n📊 Fetching pod metrics...")
    custom_api = client.CustomObjectsApi()
    try:
        metrics = custom_api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=NAMESPACE,
            plural="pods",
        )
        for item in metrics.get("items", []):
            pod_name = item["metadata"]["name"]
            for container in item.get("containers", []):
                cpu = container["usage"].get("cpu", "N/A")
                mem = container["usage"].get("memory", "N/A")
                print(f"  📦 {pod_name}/{container['name']}: CPU={cpu}, Memory={mem}")
    except ApiException as e:
        if e.status == 404:
            print("  ⚠️  Metrics API not available (install metrics-server)")
        else:
            print(f"  ❌ Metrics API error: {e.reason}")
    except Exception as e:
        print(f"  ❌ Error fetching metrics: {e}")


def get_service_url():
    """Get the NodePort URL for accessing the dashboard."""
    v1 = client.CoreV1Api()
    try:
        svc = v1.read_namespaced_service("fluidstack-gpu-dashboard-svc", NAMESPACE)
        for port in svc.spec.ports:
            if port.node_port:
                print(f"\n🌐 Dashboard accessible at: http://<node-ip>:{port.node_port}")
                print(f"   For minikube: minikube service fluidstack-gpu-dashboard-svc --url")
                return port.node_port
    except ApiException:
        print("  ⚠️  Service not found")
    return None


def main():
    print("=" * 60)
    print("⚡ Fluidstack GPU Dashboard — Kubernetes Deployer")
    print("=" * 60)

    load_kube_config()
    apply_yaml_manifests()
    pods = wait_for_pods()
    if pods:
        fetch_pod_metrics()
        get_service_url()

    print("\n✅ Deployment complete!")
    print("   Backend API:   port 8000 (container)")
    print("   Streamlit UI:  port 8501 (container)")
    print("   Prometheus:    http://<host>:8000/prometheus")


if __name__ == "__main__":
    main()
