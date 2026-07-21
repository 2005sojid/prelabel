"""Register Prelabel with CVAT as an auto-annotation model.

CVAT will not call a plain URL: it lists whatever is registered in Nuclio, and a
Nuclio function has to declare the labels it can produce. Those labels come from
the model loaded in Prelabel *right now*, so this reads them from a live server
and deploys a matching function.

    python serverless/cvat/deploy.py
    python serverless/cvat/deploy.py --prelabel-url http://127.0.0.1:8000
    python serverless/cvat/deploy.py --delete

Deployment goes through the Nuclio dashboard's HTTP API rather than ``nuctl``.
The dashboard runs inside a Linux container with Docker access and works
identically on every host; the Windows ``nuctl`` binary cannot deploy to the
local platform at all -- it shells out to ``/bin/sh``.

Re-run after loading a different model: the label list is baked into the
function's metadata, and that is what CVAT shows you.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HANDLER = HERE / "nuclio" / "main.py"

FUNCTION_NAME = "prelabel"
NAMESPACE = "nuclio"
CVAT_PROJECT = "cvat"

#: CVAT shape kinds, by Prelabel task.
SHAPE_BY_TASK = {"segment": "polygon", "pose": "points", "obb": "rectangle", "classify": "tag"}

READY_TIMEOUT = 300


def request_json(url: str, method: str = "GET", payload: dict | None = None, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body.strip() else {}


def model_info(base_url: str) -> dict:
    try:
        return request_json(f"{base_url}/api/cvat/info", timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            sys.exit("No model is loaded in Prelabel. Load one, then run this again.")
        sys.exit(f"Prelabel returned HTTP {exc.code} from /api/cvat/info")
    except urllib.error.URLError as exc:
        sys.exit(f"Cannot reach Prelabel at {base_url}: {exc.reason}")


def build_spec(info: dict, container_url: str, timeout: int, network: str) -> dict:
    """The function body Nuclio's dashboard accepts, with CVAT's metadata on it."""
    default_shape = SHAPE_BY_TASK.get(info["task"], "rectangle")
    labels = [
        {"id": entry["id"], "name": entry["name"], "type": entry.get("type", default_shape)}
        for entry in info["spec"]
    ]
    source = base64.b64encode(HANDLER.read_bytes()).decode("ascii")

    return {
        "metadata": {
            "name": FUNCTION_NAME,
            "namespace": NAMESPACE,
            "labels": {"nuclio.io/project-name": CVAT_PROJECT},
            "annotations": {
                "name": f"Prelabel ({info['name']})",
                "version": "2",
                "type": "detector",
                "framework": info["framework"],
                # CVAT parses this as JSON to build its label list.
                "spec": json.dumps(labels),
            },
        },
        "spec": {
            "description": "Prelabel proxy - inference runs in the Prelabel server, not here",
            "handler": "main:handler",
            "runtime": "python:3.11",
            "eventTimeout": f"{timeout}s",
            "env": [
                {"name": "PRELABEL_URL", "value": container_url},
                {"name": "PRELABEL_TIMEOUT", "value": str(timeout)},
            ],
            "build": {
                "functionSourceCode": source,
                "baseImage": "python:3.11-slim",
                "noBaseImagesPull": False,
            },
            "triggers": {
                "myHttpTrigger": {
                    "kind": "http",
                    "maxWorkers": 1,
                    "workerAvailabilityTimeoutMilliseconds": 10000,
                    # CVAT posts whole frames as base64; the default cap is far
                    # too small for anything above a thumbnail.
                    "attributes": {"maxRequestBodySize": 33554432},
                }
            },
            "platform": {
                "attributes": {
                    "restartPolicy": {"name": "always", "maximumRetryCount": 3},
                    "mountMode": "volume",
                    # Without this the function lands on Docker's default bridge
                    # while the Nuclio dashboard sits on CVAT's own network, and
                    # every invocation dies with an i/o timeout the dashboard
                    # reports as a plain 500.
                    "network": network,
                }
            },
        },
    }


def existing(dashboard: str) -> dict | None:
    try:
        return request_json(f"{dashboard}/api/functions/{FUNCTION_NAME}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def wait_ready(dashboard: str) -> str:
    deadline = time.time() + READY_TIMEOUT
    state = "unknown"
    while time.time() < deadline:
        function = existing(dashboard) or {}
        state = (function.get("status") or {}).get("state", "unknown")
        if state in ("ready", "error", "unhealthy"):
            return state
        print(f"  building... ({state})", flush=True)
        time.sleep(5)
    return state


def deploy(args) -> None:
    info = model_info(args.prelabel_url.rstrip("/"))
    print(f"Model: {info['name']}  task={info['task']}  labels={len(info['spec'])}")

    spec = build_spec(info, args.container_url.rstrip("/"), args.timeout, args.network)
    dashboard = args.dashboard.rstrip("/")

    method = "PUT" if existing(dashboard) else "POST"
    print(f"{'Updating' if method == 'PUT' else 'Creating'} Nuclio function '{FUNCTION_NAME}'...")
    url = f"{dashboard}/api/functions" + (f"/{FUNCTION_NAME}" if method == "PUT" else "")
    try:
        request_json(url, method=method, payload=spec)
    except urllib.error.HTTPError as exc:
        sys.exit(f"Nuclio refused the function: HTTP {exc.code}\n{exc.read().decode('utf-8')[:500]}")
    except urllib.error.URLError as exc:
        sys.exit(f"Cannot reach the Nuclio dashboard at {dashboard}: {exc.reason}")

    state = wait_ready(dashboard)
    if state != "ready":
        sys.exit(f"Function ended in state '{state}'. Check: docker logs nuclio")

    print()
    print("Ready. In CVAT: open a task, then Actions -> Automatic annotation, and pick")
    print(f"'Prelabel ({info['name']})'.")


def delete(args) -> None:
    dashboard = args.dashboard.rstrip("/")
    if not existing(dashboard):
        print(f"No function '{FUNCTION_NAME}' to remove.")
        return
    # Nuclio deletes by body, not by path: DELETE /api/functions/<name> is a 405.
    request_json(f"{dashboard}/api/functions", method="DELETE",
                 payload={"metadata": {"name": FUNCTION_NAME, "namespace": NAMESPACE}})
    print(f"Removed '{FUNCTION_NAME}' from Nuclio.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prelabel-url", default="http://127.0.0.1:8000",
                        help="Prelabel as reached from this machine (default: %(default)s)")
    parser.add_argument("--container-url", default="http://host.docker.internal:8000",
                        help="Prelabel as reached from inside the function container "
                             "(default: %(default)s)")
    parser.add_argument("--dashboard", default="http://localhost:8070",
                        help="Nuclio dashboard (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds a single inference may take (default: %(default)s)")
    parser.add_argument("--network", default="cvat_cvat",
                        help="Docker network the function joins. Must be the one the Nuclio "
                             "dashboard is on, or it cannot reach the function "
                             "(default: %(default)s)")
    parser.add_argument("--delete", action="store_true", help="remove the function and exit")
    args = parser.parse_args()

    delete(args) if args.delete else deploy(args)


if __name__ == "__main__":
    main()
