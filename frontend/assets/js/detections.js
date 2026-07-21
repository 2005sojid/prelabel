/**
 * Canvas drawing for detections.
 *
 * Shared by every surface that renders results — the single-image viewport, the
 * gallery thumbnails, the lightbox and the live webcam — so a box looks the same
 * everywhere and there is one place to change how results are drawn.
 *
 * All coordinates are in *image* pixels. Callers set up the transform; `scale`
 * is passed in only so line widths and text stay visually constant while zoomed.
 */

import { colorFor, colorForAlpha } from "./colors.js";

const MASK_ALPHA = 0.25;
const KEYPOINT_THRESHOLD = 0.3;
const SKELETON = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4],
];
const POSE_LINE = "#00ffa2";
const POSE_JOINT = "#ff8c00";

/** Draw every detection at the context's current transform. */
export function drawDetections(ctx, detections, scale) {
  for (const detection of detections) {
    const color = colorFor(detection.class_name);
    if (detection.mask) drawMask(ctx, detection.mask, detection.class_name, scale);
    if (detection.box) drawBox(ctx, detection, color, scale);
    if (detection.keypoints) drawPose(ctx, detection.keypoints, scale);
  }
}

function drawMask(ctx, polygon, className, scale) {
  if (!polygon.length) return;
  ctx.beginPath();
  polygon.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = colorForAlpha(className, MASK_ALPHA);
  ctx.fill();
  ctx.strokeStyle = colorFor(className);
  ctx.lineWidth = 2 / scale;
  ctx.stroke();
}

function drawBox(ctx, detection, color, scale) {
  const [x1, y1, x2, y2] = detection.box;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 / scale;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`;
  ctx.font = `${13 / scale}px system-ui`;
  const width = ctx.measureText(label).width + 8 / scale;
  const height = 18 / scale;
  // Keep the caption on-screen when the box is flush against the top edge.
  const top = y1 - height < 0 ? y1 : y1 - height;

  ctx.fillStyle = color;
  ctx.fillRect(x1, top, width, height);
  ctx.fillStyle = "#000";
  ctx.fillText(label, x1 + 4 / scale, top + 13 / scale);
}

function drawPose(ctx, keypoints, scale) {
  ctx.strokeStyle = POSE_LINE;
  ctx.lineWidth = 2 / scale;
  for (const [from, to] of SKELETON) {
    const a = keypoints[from];
    const b = keypoints[to];
    if (!a || !b || a[2] <= KEYPOINT_THRESHOLD || b[2] <= KEYPOINT_THRESHOLD) continue;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }

  ctx.fillStyle = POSE_JOINT;
  for (const point of keypoints) {
    if (point[2] <= KEYPOINT_THRESHOLD) continue;
    ctx.beginPath();
    ctx.arc(point[0], point[1], 3 / scale, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Classification has nothing to attach a box to, so the top labels are stamped
 * in the corner — in *screen* space, at the untransformed context.
 */
export function drawClassification(ctx, detections) {
  ctx.font = "bold 18px system-ui";
  let y = 30;
  for (const detection of detections.slice(0, 5)) {
    const label = `${detection.class_name}  ${(detection.confidence * 100).toFixed(1)}%`;
    ctx.fillStyle = "rgba(8,10,14,.8)";
    ctx.fillRect(14, y - 18, ctx.measureText(label).width + 16, 26);
    ctx.fillStyle = POSE_LINE;
    ctx.fillText(label, 22, y);
    y += 32;
  }
}

/** Fit `image` inside `width` x `height`, returning the transform to apply. */
export function fitTransform(width, height, imageWidth, imageHeight) {
  const scale = Math.min(width / imageWidth, height / imageHeight);
  return {
    scale,
    offsetX: (width - imageWidth * scale) / 2,
    offsetY: (height - imageHeight * scale) / 2,
  };
}

/** Detections at or above the current confidence threshold. */
export const visibleDetections = (detections, threshold) =>
  (detections || []).filter((d) => (d.confidence ?? 1) >= threshold);
