import argparse
import pathlib
from datetime import datetime

from plotly.offline import get_plotlyjs


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Camera Candidate Bank Designer</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #172033;
      --muted: #5d6b82;
      --line: #d5d8e2;
      --accent: #0f766e;
      --accent-2: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f6f1e8 0%, #eef3f8 100%);
      color: var(--ink);
    }
    .page {
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      padding: 20px;
      border-right: 1px solid var(--line);
      background: rgba(255, 253, 248, 0.95);
      overflow-y: auto;
    }
    .main {
      padding: 20px;
      display: grid;
      gap: 16px;
      overflow-y: auto;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 26px;
    }
    .lead {
      margin: 0 0 16px 0;
      color: var(--muted);
      line-height: 1.4;
    }
    .group {
      margin-bottom: 16px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
    }
    .group h2 {
      margin: 0 0 12px 0;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }
    .field {
      margin-bottom: 10px;
    }
    .field label {
      display: block;
      margin-bottom: 5px;
      font-size: 13px;
      color: var(--muted);
    }
    .field input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      color: var(--ink);
      background: white;
    }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 12px;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 10px 14px;
      font-size: 14px;
      cursor: pointer;
      color: white;
      background: var(--accent);
    }
    button.secondary {
      background: #475569;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: var(--panel);
    }
    .stat .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .stat .value {
      margin-top: 6px;
      font-size: 24px;
      font-weight: 700;
    }
    .plot-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .plot-card {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 253, 248, 0.92);
      overflow: hidden;
      min-height: 420px;
    }
    .plot-card.wide {
      grid-column: 1 / -1;
    }
    .plot-title {
      padding: 12px 16px 0 16px;
      font-size: 15px;
      font-weight: 600;
    }
    .note {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .reason-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .reason-item {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: #fff;
    }
    @media (max-width: 1100px) {
      .page { grid-template-columns: 1fr; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .plot-grid { grid-template-columns: 1fr; }
    }
  </style>
  <script>__PLOTLY_JS__</script>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <h1>Candidate Bank Designer</h1>
      <p class="lead">Tune the moving-camera bank live, then inspect the filtered and unfiltered pose space in 3D and in the image plane.</p>

      <div class="group">
        <h2>Bank Levels</h2>
        <div class="field"><label for="radiusLevels">Radius Levels (m)</label><input id="radiusLevels" value="0.5, 1.0, 2.0, 5.0, 10.0" /></div>
        <div class="field"><label for="azimuthLevels">Azimuth Levels (deg)</label><input id="azimuthLevels" value="-30, -15, 0, 15, 30" /></div>
        <div class="field"><label for="elevationLevels">Elevation Levels (deg)</label><input id="elevationLevels" value="-25, 0, 25" /></div>
        <div class="field"><label for="rollLevels">Roll Levels (deg)</label><input id="rollLevels" value="0, 90" /></div>
        <div class="field"><label for="yawLevels">Yaw Perturb Levels (deg)</label><input id="yawLevels" value="-5, 0, 5" /></div>
        <div class="field"><label for="pitchLevels">Pitch Perturb Levels (deg)</label><input id="pitchLevels" value="-5, 0, 5" /></div>
        <div class="field"><label for="aimPointMode">Aim Point Mode</label><input id="aimPointMode" value="board_anchors" /></div>
      </div>

      <div class="group">
        <h2>Board + Camera</h2>
        <div class="field"><label for="boardRows">Board Rows</label><input id="boardRows" type="number" value="9" /></div>
        <div class="field"><label for="boardCols">Board Cols</label><input id="boardCols" type="number" value="9" /></div>
        <div class="field"><label for="boardSquareSize">Board Square Size (m)</label><input id="boardSquareSize" type="number" step="0.001" value="0.125" /></div>
        <div class="field"><label for="width">Image Width (px)</label><input id="width" type="number" value="1280" /></div>
        <div class="field"><label for="height">Image Height (px)</label><input id="height" type="number" value="960" /></div>
        <div class="field"><label for="fx">fx</label><input id="fx" type="number" value="820" /></div>
        <div class="field"><label for="fy">fy</label><input id="fy" type="number" value="815" /></div>
      </div>

      <div class="group">
        <h2>Validity Filters</h2>
        <div class="field"><label for="minArea">Min Target Area Fraction</label><input id="minArea" type="number" step="0.001" value="0.02" /></div>
        <div class="field"><label for="maxArea">Max Target Area Fraction</label><input id="maxArea" type="number" step="0.001" value="0.55" /></div>
        <div class="field"><label for="maxSlant">Max Slant (deg)</label><input id="maxSlant" type="number" step="0.1" value="65" /></div>
        <div class="field"><label for="minSpread">Min Corner Spread (px)</label><input id="minSpread" type="number" step="0.1" value="8" /></div>
      </div>

      <div class="group">
        <h2>Render Limits</h2>
        <div class="field"><label for="maxValidAxes">Max Kept Axes</label><input id="maxValidAxes" type="number" value="250" /></div>
        <div class="field"><label for="maxInvalidAxes">Max Filtered Axes</label><input id="maxInvalidAxes" type="number" value="100" /></div>
      </div>

      <div class="actions">
        <button id="updateBtn">Update Plots</button>
        <button id="densePresetBtn" class="secondary">Dense Preset</button>
      </div>
      <div class="note">
        The filter is geometric only: all corners visible, area fraction in range, slant below threshold, and minimum projected corner spread above threshold.
      </div>
    </aside>

    <main class="main">
      <section class="summary">
        <div class="stat"><div class="label">Total Candidates</div><div class="value" id="totalCount">0</div></div>
        <div class="stat"><div class="label">Valid Candidates</div><div class="value" id="validCount">0</div></div>
        <div class="stat"><div class="label">Valid Fraction</div><div class="value" id="validFraction">0.00</div></div>
        <div class="stat"><div class="label">Rendered Axes</div><div class="value" id="renderedCount">0</div></div>
      </section>

      <section class="group">
        <h2>Rejection Counts</h2>
        <div class="reason-list">
          <div class="reason-item">All corners visible failed: <strong id="reasonVisible">0</strong></div>
          <div class="reason-item">Area fraction failed: <strong id="reasonArea">0</strong></div>
          <div class="reason-item">Slant failed: <strong id="reasonSlant">0</strong></div>
          <div class="reason-item">Corner spread failed: <strong id="reasonSpread">0</strong></div>
        </div>
      </section>

      <section class="plot-grid">
        <div class="plot-card">
          <div class="plot-title">Unfiltered 3D Bank</div>
          <div id="plot3dUnfiltered" style="height: 460px;"></div>
        </div>
        <div class="plot-card">
          <div class="plot-title">Filtered 3D Bank</div>
          <div id="plot3dFiltered" style="height: 460px;"></div>
        </div>
        <div class="plot-card">
          <div class="plot-title">Projected Corners Before Filtering</div>
          <div id="plot2dUnfiltered" style="height: 420px;"></div>
        </div>
        <div class="plot-card">
          <div class="plot-title">Projected Corners After Filtering</div>
          <div id="plot2dFiltered" style="height: 420px;"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const byId = (id) => document.getElementById(id);

    function parseNumberList(id) {
      const raw = byId(id).value.trim();
      if (!raw) return [];
      return raw.split(",").map(v => Number(v.trim())).filter(v => Number.isFinite(v));
    }

    function normalize(v) {
      const n = Math.hypot(v[0], v[1], v[2]);
      if (n < 1e-12) return [0, 0, 0];
      return [v[0] / n, v[1] / n, v[2] / n];
    }

    function cross(a, b) {
      return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
      ];
    }

    function dot(a, b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    function transpose3(R) {
      return [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]],
      ];
    }

    function matmul3(A, B) {
      const out = Array.from({ length: 3 }, () => [0, 0, 0]);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          out[i][j] = 0;
          for (let k = 0; k < 3; k++) out[i][j] += A[i][k] * B[k][j];
        }
      }
      return out;
    }

    function matvec3(A, v) {
      return [
        A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
        A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
        A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
      ];
    }

    function rotationX(rad) {
      const c = Math.cos(rad), s = Math.sin(rad);
      return [[1,0,0],[0,c,-s],[0,s,c]];
    }
    function rotationY(rad) {
      const c = Math.cos(rad), s = Math.sin(rad);
      return [[c,0,s],[0,1,0],[-s,0,c]];
    }
    function rotationZ(rad) {
      const c = Math.cos(rad), s = Math.sin(rad);
      return [[c,-s,0],[s,c,0],[0,0,1]];
    }

    function lookAtRotation(cameraPosition, target, up) {
      let forward = normalize([target[0]-cameraPosition[0], target[1]-cameraPosition[1], target[2]-cameraPosition[2]]);
      let right = cross(forward, up);
      if (Math.hypot(...right) < 1e-9) {
        right = cross(forward, [0,1,0]);
        if (Math.hypot(...right) < 1e-9) right = cross(forward, [1,0,0]);
      }
      right = normalize(right);
      const cameraDown = normalize(cross(forward, right));
      return [
        [right[0], cameraDown[0], forward[0]],
        [right[1], cameraDown[1], forward[1]],
        [right[2], cameraDown[2], forward[2]],
      ];
    }

    function createCheckerboardPoints(rows, cols, squareSize) {
      const points = [];
      const xOff = 0.5 * (cols - 1) * squareSize;
      const yOff = 0.5 * (rows - 1) * squareSize;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          points.push([c * squareSize - xOff, r * squareSize - yOff, 0]);
        }
      }
      return points;
    }

    function createAimPoints(targetPoints, mode) {
      const xs = targetPoints.map(p => p[0]);
      const ys = targetPoints.map(p => p[1]);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const midX = 0.5 * (minX + maxX), midY = 0.5 * (minY + maxY);
      const anchors = {
        center: [midX, midY, 0],
        top_left: [minX, maxY, 0],
        top_center: [midX, maxY, 0],
        top_right: [maxX, maxY, 0],
        center_left: [minX, midY, 0],
        center_right: [maxX, midY, 0],
        bottom_left: [minX, minY, 0],
        bottom_center: [midX, minY, 0],
        bottom_right: [maxX, minY, 0],
      };
      if (mode === "center") return [{ label: "center", point: anchors.center }];
      if (mode === "board_anchors") {
        return [
          "center",
          "top_left",
          "top_center",
          "top_right",
          "center_left",
          "center_right",
          "bottom_left",
          "bottom_center",
          "bottom_right",
        ].map(label => ({ label, point: anchors[label] }));
      }
      return [{ label: "center", point: anchors.center }];
    }

    function projectPoints(pointsWorld, rotationWC, translationWC, intrinsics) {
      const [fx, fy, cx, cy] = intrinsics;
      const rotationCW = transpose3(rotationWC);
      return pointsWorld.map(p => {
        const diff = [p[0] - translationWC[0], p[1] - translationWC[1], p[2] - translationWC[2]];
        const pc = matvec3(rotationCW, diff);
        if (pc[2] <= 1e-9) return [NaN, NaN];
        return [fx * (pc[0] / pc[2]) + cx, fy * (pc[1] / pc[2]) + cy];
      });
    }

    function computeAreaFraction(measurements, width, height) {
      const valid = measurements.filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
      if (!valid.length) return 0;
      const us = valid.map(p => p[0]);
      const vs = valid.map(p => p[1]);
      const bboxW = Math.max(...us) - Math.min(...us);
      const bboxH = Math.max(...vs) - Math.min(...vs);
      return (bboxW * bboxH) / (width * height);
    }

    function computeMinCornerSpread(measurements, rows, cols) {
      let best = Infinity;
      const at = (r, c) => measurements[r * cols + c];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols - 1; c++) {
          const p0 = at(r, c), p1 = at(r, c + 1);
          if (Number.isFinite(p0[0]) && Number.isFinite(p1[0])) {
            best = Math.min(best, Math.hypot(p1[0] - p0[0], p1[1] - p0[1]));
          }
        }
      }
      for (let r = 0; r < rows - 1; r++) {
        for (let c = 0; c < cols; c++) {
          const p0 = at(r, c), p1 = at(r + 1, c);
          if (Number.isFinite(p0[0]) && Number.isFinite(p1[0])) {
            best = Math.min(best, Math.hypot(p1[0] - p0[0], p1[1] - p0[1]));
          }
        }
      }
      return Number.isFinite(best) ? best : 0;
    }

    function strideSample(indices, maxCount) {
      if (maxCount == null || indices.length <= maxCount) return indices.slice();
      const step = Math.max(1, Math.floor(indices.length / maxCount));
      const out = [];
      for (let i = 0; i < indices.length && out.length < maxCount; i += step) out.push(indices[i]);
      return out;
    }

    function buildCandidateBank(config) {
      const target = [0, 0, 0];
      const up = [0, 0, 1];
      const targetPoints = createCheckerboardPoints(config.boardRows, config.boardCols, config.boardSquareSize);
      const aimPoints = createAimPoints(targetPoints, config.aimPointMode);
      const bank = [];

      for (const radius of config.radiusLevels) {
        for (const azimuthDeg of config.azimuthLevels) {
          const azimuth = azimuthDeg * Math.PI / 180;
          for (const elevationDeg of config.elevationLevels) {
            const elevation = elevationDeg * Math.PI / 180;
            const direction = normalize([Math.tan(azimuth), Math.tan(elevation), 1.0]);
            const position = [
              target[0] + radius * direction[0],
              target[1] + radius * direction[1],
              target[2] + radius * direction[2],
            ];
            for (const aim of aimPoints) {
              const baseRotation = lookAtRotation(position, aim.point, up);
              for (const rollDeg of config.rollLevels) {
                for (const yawDeg of config.yawLevels) {
                  for (const pitchDeg of config.pitchLevels) {
                    const local = matmul3(
                      matmul3(rotationZ(rollDeg * Math.PI / 180), rotationY(yawDeg * Math.PI / 180)),
                      rotationX(pitchDeg * Math.PI / 180)
                    );
                    const rotation = matmul3(baseRotation, local);
                    const measurements = projectPoints(targetPoints, rotation, position, config.intrinsics);
                    const allVisible = measurements.every(p =>
                      Number.isFinite(p[0]) &&
                      Number.isFinite(p[1]) &&
                      p[0] >= 0 && p[0] < config.width &&
                      p[1] >= 0 && p[1] < config.height
                    );
                    const areaFraction = computeAreaFraction(measurements, config.width, config.height);
                    const viewingDir = normalize([target[0]-position[0], target[1]-position[1], target[2]-position[2]]);
                    const slantAngleDeg = Math.acos(Math.max(-1, Math.min(1, Math.abs(dot(viewingDir, [0, 0, 1]))))) * 180 / Math.PI;
                    const minSpreadPx = computeMinCornerSpread(measurements, config.boardRows, config.boardCols);
                    const areaOk = areaFraction >= config.minArea && areaFraction <= config.maxArea;
                    const slantOk = slantAngleDeg <= config.maxSlant;
                    const spreadOk = minSpreadPx >= config.minSpread;
                    const valid = allVisible && areaOk && slantOk && spreadOk;
                    bank.push({
                      position,
                      rotation,
                      measurements,
                      valid,
                      radius,
                      azimuthDeg,
                      elevationDeg,
                      rollDeg,
                      yawDeg,
                      pitchDeg,
                      aimLabel: aim.label,
                      allVisible,
                      areaFraction,
                      slantAngleDeg,
                      minSpreadPx,
                    });
                  }
                }
              }
            }
          }
        }
      }
      return { bank, targetPoints };
    }

    function makeAxisLineTrace(candidates, axisIndex, axisLength, color, name, opacity) {
      const xs = [], ys = [], zs = [];
      for (const c of candidates) {
        const origin = c.position;
        const dir = [c.rotation[0][axisIndex], c.rotation[1][axisIndex], c.rotation[2][axisIndex]];
        xs.push(origin[0], origin[0] + axisLength * dir[0], null);
        ys.push(origin[1], origin[1] + axisLength * dir[1], null);
        zs.push(origin[2], origin[2] + axisLength * dir[2], null);
      }
      return {
        type: "scatter3d",
        mode: "lines",
        x: xs, y: ys, z: zs,
        line: { color, width: 5 },
        opacity,
        name,
        hoverinfo: "skip",
      };
    }

    function makeOriginTrace(candidates, color, name, opacity) {
      return {
        type: "scatter3d",
        mode: "markers",
        x: candidates.map(c => c.position[0]),
        y: candidates.map(c => c.position[1]),
        z: candidates.map(c => c.position[2]),
        marker: { size: 3, color, opacity },
        name,
        text: candidates.map(c =>
          `r=${c.radius.toFixed(2)}m<br>az=${c.azimuthDeg.toFixed(1)} deg<br>el=${c.elevationDeg.toFixed(1)} deg<br>roll=${c.rollDeg.toFixed(1)} deg<br>yaw=${c.yawDeg.toFixed(1)} deg<br>pitch=${c.pitchDeg.toFixed(1)} deg<br>aim=${c.aimLabel}<br>area=${c.areaFraction.toFixed(3)}<br>slant=${c.slantAngleDeg.toFixed(1)} deg<br>spread=${c.minSpreadPx.toFixed(1)} px`
        ),
        hovertemplate: "%{text}<extra></extra>",
      };
    }

    function build3DTraces(targetPoints, keptCandidates, filteredCandidates, showFiltered) {
      const traces = [{
        type: "scatter3d",
        mode: "markers",
        x: targetPoints.map(p => p[0]),
        y: targetPoints.map(p => p[1]),
        z: targetPoints.map(p => p[2]),
        marker: { size: 4, color: "#111827" },
        name: "checkerboard",
        hoverinfo: "skip",
      }];
      if (keptCandidates.length) {
        traces.push(makeAxisLineTrace(keptCandidates, 0, 0.45, "#dc2626", "kept x", 0.95));
        traces.push(makeAxisLineTrace(keptCandidates, 1, 0.45, "#2563eb", "kept y", 0.95));
        traces.push(makeAxisLineTrace(keptCandidates, 2, 0.45, "#059669", "kept z", 0.95));
        traces.push(makeOriginTrace(keptCandidates, "#065f46", "kept origins", 0.95));
      }
      if (showFiltered && filteredCandidates.length) {
        traces.push(makeAxisLineTrace(filteredCandidates, 0, 0.32, "#dc2626", "filtered x", 0.25));
        traces.push(makeAxisLineTrace(filteredCandidates, 1, 0.32, "#2563eb", "filtered y", 0.25));
        traces.push(makeAxisLineTrace(filteredCandidates, 2, 0.32, "#059669", "filtered z", 0.25));
        traces.push(makeOriginTrace(filteredCandidates, "#64748b", "filtered origins", 0.45));
      }
      return traces;
    }

    function buildCornerScatter(candidates, color, name) {
      const xs = [], ys = [];
      for (const c of candidates) {
        for (const p of c.measurements) {
          if (Number.isFinite(p[0]) && Number.isFinite(p[1])) {
            xs.push(p[0]);
            ys.push(p[1]);
          }
        }
      }
      return {
        type: "scatter",
        mode: "markers",
        x: xs,
        y: ys,
        marker: { size: 4, color, opacity: 0.28 },
        name,
      };
    }

    function updateSummary(data, renderedValid, renderedInvalid) {
      const total = data.bank.length;
      const valid = data.bank.filter(c => c.valid).length;
      byId("totalCount").textContent = String(total);
      byId("validCount").textContent = String(valid);
      byId("validFraction").textContent = total ? (valid / total).toFixed(3) : "0.000";
      byId("renderedCount").textContent = String(renderedValid + renderedInvalid);

      let failVisible = 0, failArea = 0, failSlant = 0, failSpread = 0;
      for (const c of data.bank) {
        if (!c.allVisible) failVisible += 1;
        if (!(c.areaFraction >= currentConfig.minArea && c.areaFraction <= currentConfig.maxArea)) failArea += 1;
        if (!(c.slantAngleDeg <= currentConfig.maxSlant)) failSlant += 1;
        if (!(c.minSpreadPx >= currentConfig.minSpread)) failSpread += 1;
      }
      byId("reasonVisible").textContent = String(failVisible);
      byId("reasonArea").textContent = String(failArea);
      byId("reasonSlant").textContent = String(failSlant);
      byId("reasonSpread").textContent = String(failSpread);
    }

    function readConfig() {
      const width = Number(byId("width").value);
      const height = Number(byId("height").value);
      return {
        radiusLevels: parseNumberList("radiusLevels"),
        azimuthLevels: parseNumberList("azimuthLevels"),
        elevationLevels: parseNumberList("elevationLevels"),
        rollLevels: parseNumberList("rollLevels"),
        yawLevels: parseNumberList("yawLevels"),
        pitchLevels: parseNumberList("pitchLevels"),
        aimPointMode: byId("aimPointMode").value.trim() || "center",
        boardRows: Number(byId("boardRows").value),
        boardCols: Number(byId("boardCols").value),
        boardSquareSize: Number(byId("boardSquareSize").value),
        width,
        height,
        intrinsics: [
          Number(byId("fx").value),
          Number(byId("fy").value),
          width / 2.0,
          height / 2.0,
        ],
        minArea: Number(byId("minArea").value),
        maxArea: Number(byId("maxArea").value),
        maxSlant: Number(byId("maxSlant").value),
        minSpread: Number(byId("minSpread").value),
        maxValidAxes: Number(byId("maxValidAxes").value),
        maxInvalidAxes: Number(byId("maxInvalidAxes").value),
      };
    }

    function draw() {
      currentConfig = readConfig();
      const data = buildCandidateBank(currentConfig);
      const validIndices = data.bank.map((c, i) => c.valid ? i : -1).filter(i => i >= 0);
      const invalidIndices = data.bank.map((c, i) => !c.valid ? i : -1).filter(i => i >= 0);
      const sampledValid = strideSample(validIndices, currentConfig.maxValidAxes).map(i => data.bank[i]);
      const sampledInvalid = strideSample(invalidIndices, currentConfig.maxInvalidAxes).map(i => data.bank[i]);

      updateSummary(data, sampledValid.length, sampledInvalid.length);

      Plotly.react("plot3dUnfiltered", build3DTraces(data.targetPoints, sampledValid, [], false), {
        margin: {l: 0, r: 0, t: 10, b: 0},
        legend: {orientation: "h"},
        scene: {xaxis: {title: "x"}, yaxis: {title: "y"}, zaxis: {title: "z"}, aspectmode: "data"},
      }, {responsive: true, displaylogo: false});

      Plotly.react("plot3dFiltered", build3DTraces(data.targetPoints, sampledValid, sampledInvalid, true), {
        margin: {l: 0, r: 0, t: 10, b: 0},
        legend: {orientation: "h"},
        scene: {xaxis: {title: "x"}, yaxis: {title: "y"}, zaxis: {title: "z"}, aspectmode: "data"},
      }, {responsive: true, displaylogo: false});

      const allCandidates = data.bank;
      const keptCandidates = data.bank.filter(c => c.valid);
      Plotly.react("plot2dUnfiltered", [buildCornerScatter(allCandidates, "#0f766e", "all projected corners")], {
        margin: {l: 50, r: 10, t: 10, b: 45},
        xaxis: {title: "u (px)", range: [0, currentConfig.width]},
        yaxis: {title: "v (px)", range: [currentConfig.height, 0]},
      }, {responsive: true, displaylogo: false});

      Plotly.react("plot2dFiltered", [buildCornerScatter(keptCandidates, "#b45309", "kept projected corners")], {
        margin: {l: 50, r: 10, t: 10, b: 45},
        xaxis: {title: "u (px)", range: [0, currentConfig.width]},
        yaxis: {title: "v (px)", range: [currentConfig.height, 0]},
      }, {responsive: true, displaylogo: false});
    }

    let currentConfig = null;
    byId("updateBtn").addEventListener("click", draw);
    byId("densePresetBtn").addEventListener("click", () => {
      byId("radiusLevels").value = "0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0";
      byId("elevationLevels").value = "-25, -10, 0, 10, 25";
      byId("rollLevels").value = "0, 45, 90, 135";
      byId("yawLevels").value = "-10, -5, 0, 5, 10";
      byId("pitchLevels").value = "-10, -5, 0, 5, 10";
      byId("aimPointMode").value = "board_anchors";
      byId("maxValidAxes").value = "800";
      byId("maxInvalidAxes").value = "250";
      draw();
    });

    draw();
  </script>
</body>
</html>
"""


def create_output_path(base_dir: pathlib.Path) -> pathlib.Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"camera_candidate_bank_designer_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir / "camera_candidate_bank_designer.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a standalone live HTML tool for camera candidate-bank design.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results"),
        help="directory where the generated HTML tool will be saved",
    )
    args = parser.parse_args()

    output_path = create_output_path(pathlib.Path(args.output_dir))
    html = HTML_TEMPLATE.replace("__PLOTLY_JS__", get_plotlyjs())
    output_path.write_text(html, encoding="utf-8")
    print(f"Interactive designer saved to: {output_path}")
