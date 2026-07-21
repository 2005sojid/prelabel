"""Nuclio function that lets CVAT call Prelabel.

CVAT does not accept a plain URL for an auto-annotation model: it discovers
models through Nuclio and calls whatever functions are registered there. So the
piece that makes "press Annotate in CVAT" work is this — a function CVAT can
find, which forwards the request to a Prelabel server and returns its answer
unchanged.

Deliberately a *proxy* and nothing more. The model, the GPU and the weights stay
in Prelabel; this container carries no ML dependencies, builds in seconds, and
never needs rebuilding when the model changes.
"""

import json
import os
import urllib.error
import urllib.request

#: Where Prelabel is listening, as seen *from inside this container*.
#: On Docker Desktop the host is reachable as host.docker.internal; on Linux,
#: run the container with --add-host=host.docker.internal:host-gateway, or point
#: this at the container name if Prelabel itself runs in Docker.
PRELABEL_URL = os.environ.get("PRELABEL_URL", "http://host.docker.internal:8000").rstrip("/")

#: Prelabel loads a model once and holds it, so a request is a forward pass, not
#: a load. Still generous: a CPU-only host running a large model is slow.
TIMEOUT_SECONDS = float(os.environ.get("PRELABEL_TIMEOUT", "300"))

#: Optional shared token, when the Prelabel server has PL_AUTH_TOKEN set.
AUTH_TOKEN = os.environ.get("PRELABEL_TOKEN", "").strip()


def init_context(context):
    context.logger.info(f"Prelabel proxy ready, forwarding to {PRELABEL_URL}")
    context.user_data.url = PRELABEL_URL


def _post(path, payload):
    request = urllib.request.Request(
        f"{PRELABEL_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if AUTH_TOKEN:
        request.add_header("Authorization", f"Bearer {AUTH_TOKEN}")
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def handler(context, event):
    """Forward one CVAT auto-annotation request to Prelabel.

    CVAT sends ``{"image": "<base64>", "threshold": <float>}`` and expects a flat
    JSON array of shapes in pixel coordinates. Prelabel speaks exactly that, so
    the body passes through untouched.
    """
    body = event.body
    if isinstance(body, (bytes, bytearray)):
        body = json.loads(body.decode("utf-8"))
    elif isinstance(body, str):
        body = json.loads(body)

    image = body.get("image")
    if not image:
        return context.Response(
            body=json.dumps({"error": "no image in request"}),
            content_type="application/json",
            status_code=400,
        )

    payload = {"image": image, "threshold": float(body.get("threshold", 0.5))}

    try:
        shapes = _post("/api/cvat/invoke", payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        context.logger.warn(f"Prelabel returned {exc.code}: {detail}")
        return context.Response(
            body=json.dumps({"error": f"Prelabel returned {exc.code}: {detail}"}),
            content_type="application/json",
            status_code=502,
        )
    except urllib.error.URLError as exc:
        # By far the most common failure: Prelabel is not running, or is not
        # reachable from inside the container. Say which, because "connection
        # refused" from a serverless function is otherwise a long afternoon.
        context.logger.warn(f"Cannot reach Prelabel at {PRELABEL_URL}: {exc.reason}")
        return context.Response(
            body=json.dumps({
                "error": f"Cannot reach Prelabel at {PRELABEL_URL} ({exc.reason}). "
                         "Is the server running, and is PRELABEL_URL correct for this container?"
            }),
            content_type="application/json",
            status_code=502,
        )

    context.logger.info(f"Prelabel returned {len(shapes)} shapes")
    return context.Response(
        body=json.dumps(shapes),
        content_type="application/json",
        status_code=200,
    )
