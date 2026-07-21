/**
 * Dataset exporters.
 *
 * Each produces a ready-to-import ZIP containing the images *and* the labels,
 * laid out the way the target tool expects. The COCO family is used for the
 * detection-shaped tasks because it is the de-facto interchange format that
 * CVAT, Label Studio and Roboflow all read.
 */

import { confidence, store } from "../state.js";
import { doneItems, visibleFor } from "../batch/items.js";
import { saveZip } from "./zip.js";

const IMAGE_ROOT = "images/default/";

const round2 = (value) => Math.round(value * 100) / 100;

/** Strip characters that are unsafe in a path, keeping letters of any script. */
const safeName = (name) => name.replace(/[\x00-\x1f<>:"/\\|?*]+/g, "_").replace(/^\.+/, "_") || "file";

async function fileBytes(file) {
  return new Uint8Array(await file.arrayBuffer());
}

/**
 * Class name → index, preferring the model's own ordering so exported ids match
 * the model's. Classes the model didn't declare are appended.
 */
export function classIndex() {
  const index = new Map();
  (store.model?.class_names ? Object.values(store.model.class_names) : []).forEach((name, position) => {
    if (!index.has(name)) index.set(name, position);
  });
  let next = index.size;
  for (const item of doneItems()) {
    for (const detection of item.detections) {
      if (!index.has(detection.class_name)) index.set(detection.class_name, next++);
    }
  }
  return index;
}

const orderedNames = (index) => [...index.entries()].sort((a, b) => a[1] - b[1]).map(([name]) => name);

/** Shared skeleton for the COCO exporters. */
async function buildCoco({ withSegmentation = false, withKeypoints = false } = {}) {
  const index = classIndex();
  const threshold = confidence();
  const items = doneItems();

  const keypointCount = withKeypoints
    ? Math.max(0, ...items.flatMap((item) => item.detections.map((d) => (d.keypoints || []).length)))
    : 0;

  const categories = orderedNames(index).map((name, position) => ({
    id: position + 1, // COCO ids are 1-based
    name,
    supercategory: "",
    ...(withKeypoints
      ? { keypoints: Array.from({ length: keypointCount }, (_, i) => `kp${i}`), skeleton: [] }
      : {}),
  }));

  const images = [];
  const annotations = [];
  const files = [];
  let annotationId = 1;

  for (const [position, item] of items.entries()) {
    const name = safeName(item.path);
    const imageId = position + 1;
    images.push({ id: imageId, file_name: name, width: item.width, height: item.height });
    files.push({ name: IMAGE_ROOT + name, data: await fileBytes(item.file) });

    for (const detection of visibleFor(item, threshold)) {
      if (!detection.box) continue;
      const [x1, y1, x2, y2] = detection.box;
      const annotation = {
        id: annotationId++,
        image_id: imageId,
        category_id: index.get(detection.class_name) + 1,
        bbox: [round2(x1), round2(y1), round2(x2 - x1), round2(y2 - y1)],
        area: round2((x2 - x1) * (y2 - y1)),
        iscrowd: 0,
      };
      if (withSegmentation) {
        annotation.segmentation = detection.mask ? [detection.mask.flat().map(round2)] : [];
      } else if (!withKeypoints) {
        annotation.segmentation = [];
      }
      if (withKeypoints) {
        const flattened = [];
        let visible = 0;
        for (const point of detection.keypoints || []) {
          const visibility = point[2] > 0.5 ? 2 : point[2] > 0 ? 1 : 0;
          if (visibility) visible += 1;
          flattened.push(round2(point[0]), round2(point[1]), visibility);
        }
        annotation.keypoints = flattened;
        annotation.num_keypoints = visible;
      }
      annotations.push(annotation);
    }
  }

  return {
    document: {
      info: { description: "Prelabel auto-annotations", date_created: new Date().toISOString() },
      licenses: [{ id: 1, name: "", url: "" }],
      images,
      annotations,
      categories,
    },
    files,
  };
}

function withJson(files, path, document) {
  const bytes = new TextEncoder().encode(JSON.stringify(document));
  return [...files, { name: path, data: bytes }];
}

export async function exportCoco({ withSegmentation = false } = {}) {
  const { document, files } = await buildCoco({ withSegmentation });
  saveZip(withJson(files, "annotations/instances_default.json", document), "coco_dataset.zip");
}

export async function exportCocoKeypoints() {
  const { document, files } = await buildCoco({ withKeypoints: true });
  saveZip(withJson(files, "annotations/person_keypoints_default.json", document), "coco_keypoints_dataset.zip");
}

/** Classification: one folder per predicted class. */
export async function exportImageNet() {
  const threshold = confidence();
  const files = [];
  for (const item of doneItems()) {
    const top = visibleFor(item, threshold)[0];
    const folder = safeName(top ? top.class_name : "unknown");
    files.push({ name: `${folder}/${safeName(item.name)}`, data: await fileBytes(item.file) });
  }
  saveZip(files, "imagenet_dataset.zip");
}

/** Task → the exporters offered for it. */
export const EXPORTERS = {
  detect: [{ label: "COCO 1.0", sub: "images + instances_default.json", run: () => exportCoco() }],
  segment: [{ label: "COCO 1.0", sub: "images + polygon segmentation", run: () => exportCoco({ withSegmentation: true }) }],
  pose: [{ label: "COCO Keypoints 1.0", sub: "images + keypoints json", run: exportCocoKeypoints }],
  obb: [{ label: "COCO 1.0", sub: "axis-aligned boxes json", run: () => exportCoco() }],
  classify: [{ label: "ImageNet", sub: "images grouped into class folders", run: exportImageNet }],
};
