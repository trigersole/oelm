import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import { FEATURE_ORDER } from './data/featureOrder.js';


// ===============================================================
// CONFIG
// ===============================================================
// -- Hugging Face Space API endpoint ------------------------------
// Set this to your Hugging Face Docker Space public URL after deploying:
const HF_SPACE_URL = window.OELM_CONFIG?.HF_SPACE_URL || '';
const API_BASE_URL = window.OELM_CONFIG?.API_BASE_URL || HF_SPACE_URL;

const SUPABASE_URL = window.OELM_CONFIG?.SUPABASE_URL || '';
const SUPABASE_KEY = window.OELM_CONFIG?.SUPABASE_KEY || '';

const CAPTURE_FPS    = 5;
const WINDOW_SECONDS = 10;
const WINDOW_FRAMES  = CAPTURE_FPS * WINDOW_SECONDS;  // 50
const MODEL_VERSION  = 'xgb-binary-v1';
const SINGAPORE_TZ   = 'Asia/Singapore';
const GAZE_AOI_CONFIG = {
  name: 'Whole screen AOI',
  minX: 0,
  maxX: 1,
  minY: 0,
  maxY: 1,
  invertHorizontal: true,
  eyeWeight: 0.8,
  headWeight: 0.2,
  maxAbsYaw: 0.55,
  maxAbsPitch: 0.35,
  maxAbsRoll: 0.55,
  minTrackingCoverage: 0.6,
  attentiveRatio: 0.7,
  partialRatio: 0.4,
};
const SHOW_WEBCAM_PANEL = false;
const SHOW_DEBUG_PANEL = true;
const ADMIN_USER_ID = window.OELM_CONFIG?.ADMIN_USER_ID || '';
const ADMIN_PASSWORD = window.OELM_CONFIG?.ADMIN_PASSWORD || '';
const MANUAL_OVERRIDE_TABLE = 'manual_overrides';
const PAUSE_REFLECTION_AOI_TABLE = 'pause_reflection_aoi';
const MANUAL_OVERRIDE_CHANGE_THRESHOLD_PERCENT = 20;
const LABEL_SPLIT_MIN_BUCKET_MS = 1 * 1000;

const ACTIVITY_TYPES = {
  WITHOUT_EDIT: 'without_edit',
  WITH_EDIT: 'with_edit',
  WITH_EDIT_JUSTIFICATION: 'with_edit_justification',
};

const ACTIVITY_TYPE_LABELS = {
  [ACTIVITY_TYPES.WITHOUT_EDIT]: 'No Edit',
  [ACTIVITY_TYPES.WITH_EDIT]: 'With Edit',
  [ACTIVITY_TYPES.WITH_EDIT_JUSTIFICATION]: 'With Edit + Text',
};

function getEditPolicyByType(activityType) {
  if (activityType === ACTIVITY_TYPES.WITHOUT_EDIT) {
    return { canEdit: false, requireTextJustification: false };
  }
  if (activityType === ACTIVITY_TYPES.WITH_EDIT) {
    return { canEdit: true, requireTextJustification: false };
  }
  return { canEdit: true, requireTextJustification: true };
}

function generateAccessCode(length = 8) {
  const chars = 'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let out = '';
  for (let i = 0; i < length; i++) out += chars[Math.floor(Math.random() * chars.length)];
  return out;
}

function formatSingaporeTime(value, options = {}) {
  return new Date(value).toLocaleString([], { timeZone: SINGAPORE_TZ, ...options });
}

function singaporeDayKey(value) {
  return formatSingaporeTime(value, { year: 'numeric', month: '2-digit', day: '2-digit' });
}

function csvEscape(value) {
  const text = value === null || value === undefined ? '' : String(value);
  return `"${text.replace(/"/g, '""')}"`;
}

function downloadCsv(filename, rows) {
  const csv = rows.map(row => row.map(csvEscape).join(',')).join('\r\n');
  const blob = new Blob([`\uFEFF${csv}`], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function safeFilenamePart(value) {
  return String(value || 'all')
    .trim()
    .replace(/[^a-z0-9_-]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    || 'all';
}

// ===============================================================
// BLENDSHAPE_NAMES
// ===============================================================
const BLENDSHAPE_NAMES = [
  '_neutral',
  'browDownLeft','browDownRight',
  'browInnerUp',
  'browOuterUpLeft','browOuterUpRight',
  'cheekPuff',
  'cheekSquintLeft','cheekSquintRight',
  'eyeBlinkLeft','eyeBlinkRight',
  'eyeLookDownLeft','eyeLookDownRight',
  'eyeLookInLeft','eyeLookInRight',
  'eyeLookOutLeft','eyeLookOutRight',
  'eyeLookUpLeft','eyeLookUpRight',
  'eyeSquintLeft','eyeSquintRight',
  'eyeWideLeft','eyeWideRight',
  'jawForward',
  'jawLeft','jawRight',
  'jawOpen',
  'mouthClose',
  'mouthDimpleLeft','mouthDimpleRight',
  'mouthFrownLeft','mouthFrownRight',
  'mouthFunnel',
  'mouthLeft','mouthRight',
  'mouthLowerDownLeft','mouthLowerDownRight',
  'mouthPressLeft','mouthPressRight',
  'mouthPucker',
  'mouthRollLower','mouthRollUpper',
  'mouthShrugLower','mouthShrugUpper',
  'mouthSmileLeft','mouthSmileRight',
  'mouthStretchLeft','mouthStretchRight',
  'mouthUpperUpLeft','mouthUpperUpRight',
  'noseSneerLeft','noseSneerRight',
];

const LABEL_COLS   = ['Boredom','Engagement','Confusion','Frustration'];
const LABEL_COLORS = { Boredom:'#6B7280', Engagement:'#10B981', Confusion:'#F59E0B', Frustration:'#EF4444' };
const LABEL_EMOJIS = { Boredom:'', Engagement:'', Confusion:'', Frustration:'' };
const LEVEL_LABELS = ['Low','High'];
const LEVEL_COLORS = ['#38BDF8', '#F97316'];
const LEVEL_COUNT = LEVEL_LABELS.length;

const PLOTLY_STATIC_CONFIG = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
  doubleClick: false,
  editable: false,
  showTips: false,
  staticPlot: true,
};

const PLOTLY_EDIT_CONFIG = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
  doubleClick: false,
  editable: false,
  showTips: false,
  staticPlot: false,
};

function svgIcon(className, viewBox, children) {
  return React.createElement('svg', {
    className,
    viewBox,
    fill: 'none',
    xmlns: 'http://www.w3.org/2000/svg',
    'aria-hidden': 'true',
  }, children);
}

function chartMarkIcon(iconClass = 'chart-title-icon') {
  return svgIcon(`svg-icon ${iconClass}`, '0 0 24 24', [
    React.createElement('rect', { key:'frame', x:'3.5', y:'3.5', width:'17', height:'17', rx:'3' }),
    React.createElement('path', { key:'bar1', d:'M8 16v-4' }),
    React.createElement('path', { key:'bar2', d:'M12 16V8' }),
    React.createElement('path', { key:'bar3', d:'M16 16v-6' }),
    React.createElement('path', { key:'base', d:'M7 16h10' }),
  ]);
}

function panelTitle(text, iconClass = 'chart-title-icon') {
  return React.createElement('span', { className:'panel-title' },
    chartMarkIcon(iconClass),
    text,
  );
}

// ===============================================================
// extractBlendshapes()
// ===============================================================
function extractBlendshapes(result) {
  if (!result.faceBlendshapes?.length) return null;
  const cats = result.faceBlendshapes[0].categories;
  const out = {};
  for (const cat of cats) {
    out[cat.categoryName] = parseFloat(cat.score.toFixed(5));
  }
  return out;
}

// ===============================================================
// FEATURE_ORDER - 364 keys from feature_order.pkl
// ===============================================================

// ===============================================================
// aggregateWindow()
// ===============================================================
function aggregateWindow(frames) {
  if (!frames.length) return null;
  const result = {};
  for (const name of BLENDSHAPE_NAMES) {
    const vals = frames.map(f => f[name] ?? 0);
    const n = vals.length;
    const mean = vals.reduce((s, v) => s + v, 0) / n;
    const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    let min = vals[0], max = vals[0];
    for (const v of vals) { if (v < min) min = v; if (v > max) max = v; }
    const sorted = [...vals].sort((a, b) => a - b);
    const mid = n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[Math.floor(n / 2)];
    const skew = std === 0 ? 0 : vals.reduce((s, v) => s + ((v - mean) / std) ** 3, 0) / n;
    const kurt = std === 0 ? 0 : (vals.reduce((s, v) => s + ((v - mean) / std) ** 4, 0) / n) - 3;
    result[`${name}_mean`]   = parseFloat(mean.toFixed(5));
    result[`${name}_std`]    = parseFloat(std.toFixed(5));
    result[`${name}_min`]    = parseFloat(min.toFixed(5));
    result[`${name}_max`]    = parseFloat(max.toFixed(5));
    result[`${name}_median`] = parseFloat(mid.toFixed(5));
    result[`${name}_skew`]   = parseFloat(skew.toFixed(5));
    result[`${name}_kurt`]   = parseFloat(kurt.toFixed(5));
  }
  return result;
}

function averageLandmark(landmarks, indices) {
  const pts = indices.map(i => landmarks[i]).filter(Boolean);
  if (!pts.length) return null;
  return {
    x: pts.reduce((sum, p) => sum + p.x, 0) / pts.length,
    y: pts.reduce((sum, p) => sum + p.y, 0) / pts.length,
    z: pts.reduce((sum, p) => sum + (p.z || 0), 0) / pts.length,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function safeRatio(value, min, max) {
  const span = max - min;
  if (!Number.isFinite(span) || Math.abs(span) < 1e-6) return null;
  return clamp((value - min) / span, 0, 1);
}

function estimateEyeRatios(landmarks, irisIndices, cornerA, cornerB, top, bottom) {
  const iris = averageLandmark(landmarks, irisIndices);
  const a = landmarks[cornerA], b = landmarks[cornerB], t = landmarks[top], bt = landmarks[bottom];
  if (!iris || !a || !b || !t || !bt) return null;

  const x = safeRatio(iris.x, Math.min(a.x, b.x), Math.max(a.x, b.x));
  const y = safeRatio(iris.y, Math.min(t.y, bt.y), Math.max(t.y, bt.y));
  if (x === null || y === null) return null;
  return { x, y };
}

function estimateGazeAOI(landmarks) {
  if (!landmarks || landmarks.length < 478) return null;

  const leftEye = estimateEyeRatios(landmarks, [468, 469, 470, 471, 472], 33, 133, 159, 145);
  const rightEye = estimateEyeRatios(landmarks, [473, 474, 475, 476, 477], 362, 263, 386, 374);
  if (!leftEye || !rightEye) return null;

  const rawGazeX = clamp((leftEye.x + rightEye.x) / 2, 0, 1);
  const gazeY = clamp((leftEye.y + rightEye.y) / 2, 0, 1);

  const nose = landmarks[1];
  const chin = landmarks[152];
  const leftOuterEye = landmarks[33];
  const rightOuterEye = landmarks[263];
  const leftMouth = landmarks[61];
  const rightMouth = landmarks[291];
  if (!nose || !chin || !leftOuterEye || !rightOuterEye || !leftMouth || !rightMouth) return null;

  const eyeCenter = {
    x: (leftOuterEye.x + rightOuterEye.x) / 2,
    y: (leftOuterEye.y + rightOuterEye.y) / 2,
  };
  const mouthCenter = {
    x: (leftMouth.x + rightMouth.x) / 2,
    y: (leftMouth.y + rightMouth.y) / 2,
  };
  const eyeDist = Math.hypot(rightOuterEye.x - leftOuterEye.x, rightOuterEye.y - leftOuterEye.y) || 1;
  const faceHeight = Math.hypot(mouthCenter.x - eyeCenter.x, mouthCenter.y - eyeCenter.y) || 1;

  const headYaw = clamp((nose.x - eyeCenter.x) / eyeDist, -1.5, 1.5);
  const headPitch = clamp(((nose.y - eyeCenter.y) / faceHeight) - 0.5, -1.5, 1.5);
  const headRoll = Math.atan2(rightOuterEye.y - leftOuterEye.y, rightOuterEye.x - leftOuterEye.x);

  const rawHeadX = clamp(0.5 + headYaw, 0, 1);
  const rawHeadY = clamp(0.5 + headPitch, 0, 1);
  const screenGazeX = GAZE_AOI_CONFIG.invertHorizontal ? 1 - rawGazeX : rawGazeX;
  const screenHeadX = GAZE_AOI_CONFIG.invertHorizontal ? 1 - rawHeadX : rawHeadX;
  const combinedGazeX = clamp(
    (screenGazeX * GAZE_AOI_CONFIG.eyeWeight) +
    (screenHeadX * GAZE_AOI_CONFIG.headWeight),
    0,
    1,
  );
  const combinedGazeY = clamp(
    (gazeY * GAZE_AOI_CONFIG.eyeWeight) +
    (rawHeadY * GAZE_AOI_CONFIG.headWeight),
    0,
    1,
  );

  const headInside =
    Math.abs(headYaw) <= GAZE_AOI_CONFIG.maxAbsYaw &&
    Math.abs(headPitch) <= GAZE_AOI_CONFIG.maxAbsPitch &&
    Math.abs(headRoll) <= GAZE_AOI_CONFIG.maxAbsRoll;
  const gazeInside =
    combinedGazeX >= GAZE_AOI_CONFIG.minX &&
    combinedGazeX <= GAZE_AOI_CONFIG.maxX &&
    combinedGazeY >= GAZE_AOI_CONFIG.minY &&
    combinedGazeY <= GAZE_AOI_CONFIG.maxY;
  const insideAOI = gazeInside && headInside;
  const horizontalMargin = Math.min(
    safeRatio(combinedGazeX, GAZE_AOI_CONFIG.minX, GAZE_AOI_CONFIG.maxX) ?? 0,
    safeRatio(GAZE_AOI_CONFIG.maxX - combinedGazeX, 0, GAZE_AOI_CONFIG.maxX - GAZE_AOI_CONFIG.minX) ?? 0,
  );
  const verticalMargin = Math.min(
    safeRatio(combinedGazeY, GAZE_AOI_CONFIG.minY, GAZE_AOI_CONFIG.maxY) ?? 0,
    safeRatio(GAZE_AOI_CONFIG.maxY - combinedGazeY, 0, GAZE_AOI_CONFIG.maxY - GAZE_AOI_CONFIG.minY) ?? 0,
  );
  const gazeMargin = clamp(Math.min(horizontalMargin, verticalMargin) * 2, 0, 1);
  const headPenalty = Math.max(
    Math.abs(headYaw) / GAZE_AOI_CONFIG.maxAbsYaw,
    Math.abs(headPitch) / GAZE_AOI_CONFIG.maxAbsPitch,
    Math.abs(headRoll) / GAZE_AOI_CONFIG.maxAbsRoll,
  );

  return {
    aoi_name: GAZE_AOI_CONFIG.name,
    aoi_bounds: {
      min_x: GAZE_AOI_CONFIG.minX,
      max_x: GAZE_AOI_CONFIG.maxX,
      min_y: GAZE_AOI_CONFIG.minY,
      max_y: GAZE_AOI_CONFIG.maxY,
    },
    horizontal_mapping: GAZE_AOI_CONFIG.invertHorizontal ? 'camera_to_screen_inverted' : 'camera_to_screen_direct',
    inside_aoi: insideAOI,
    raw_gaze_x: parseFloat(rawGazeX.toFixed(5)),
    screen_gaze_x: parseFloat(screenGazeX.toFixed(5)),
    head_screen_x: parseFloat(screenHeadX.toFixed(5)),
    combined_gaze_x: parseFloat(combinedGazeX.toFixed(5)),
    combined_gaze_y: parseFloat(combinedGazeY.toFixed(5)),
    gaze_x: parseFloat(combinedGazeX.toFixed(5)),
    gaze_y: parseFloat(combinedGazeY.toFixed(5)),
    head_yaw: parseFloat(headYaw.toFixed(5)),
    head_pitch: parseFloat(headPitch.toFixed(5)),
    head_roll: parseFloat(headRoll.toFixed(5)),
    confidence: parseFloat(clamp((insideAOI ? 0.55 : 0.35) + gazeMargin * 0.35 - headPenalty * 0.2, 0, 1).toFixed(5)),
  };
}

function summarizeGazeWindow(samples) {
  const valid = samples.filter(Boolean);
  const totalFrames = samples.length;
  const validFrames = valid.length;
  const insideFrames = valid.filter(g => g.inside_aoi).length;
  const outsideFrames = validFrames - insideFrames;
  const trackingCoverage = totalFrames ? validFrames / totalFrames : 0;
  const insideRatio = validFrames ? insideFrames / validFrames : null;
  const avg = key => validFrames
    ? parseFloat((valid.reduce((sum, g) => sum + (Number(g[key]) || 0), 0) / validFrames).toFixed(5))
    : null;
  let attentionStatus = 'WAITING';
  if (totalFrames) {
    if (trackingCoverage < GAZE_AOI_CONFIG.minTrackingCoverage) {
      attentionStatus = 'INSUFFICIENT_TRACKING';
    } else if (insideRatio >= GAZE_AOI_CONFIG.attentiveRatio) {
      attentionStatus = 'VISUALLY_ATTENTIVE';
    } else if (insideRatio >= GAZE_AOI_CONFIG.partialRatio) {
      attentionStatus = 'PARTIALLY_ATTENTIVE';
    } else {
      attentionStatus = 'LOOKING_AWAY';
    }
  }

  return {
    aoi_name: GAZE_AOI_CONFIG.name,
    aoi_bounds: {
      min_x: GAZE_AOI_CONFIG.minX,
      max_x: GAZE_AOI_CONFIG.maxX,
      min_y: GAZE_AOI_CONFIG.minY,
      max_y: GAZE_AOI_CONFIG.maxY,
    },
    horizontal_mapping: GAZE_AOI_CONFIG.invertHorizontal ? 'camera_to_screen_inverted' : 'camera_to_screen_direct',
    total_gaze_frames: totalFrames,
    valid_gaze_frames: validFrames,
    inside_aoi_frames: insideFrames,
    outside_aoi_frames: outsideFrames,
    tracking_coverage: parseFloat(trackingCoverage.toFixed(5)),
    inside_aoi_ratio: insideRatio === null ? null : parseFloat(insideRatio.toFixed(5)),
    outside_aoi_ratio: validFrames ? parseFloat((outsideFrames / validFrames).toFixed(5)) : null,
    attention_status: attentionStatus,
    avg_raw_gaze_x: avg('raw_gaze_x'),
    avg_screen_gaze_x: avg('screen_gaze_x'),
    avg_head_screen_x: avg('head_screen_x'),
    avg_combined_gaze_x: avg('combined_gaze_x'),
    avg_combined_gaze_y: avg('combined_gaze_y'),
    avg_gaze_x: avg('gaze_x'),
    avg_gaze_y: avg('gaze_y'),
    avg_head_yaw: avg('head_yaw'),
    avg_head_pitch: avg('head_pitch'),
    avg_head_roll: avg('head_roll'),
  };
}

function attentionStatusLabel(status) {
  return {
    WAITING: 'Collecting gaze samples',
    INSUFFICIENT_TRACKING: 'Insufficient tracking',
    VISUALLY_ATTENTIVE: 'Looking at screen',
    PARTIALLY_ATTENTIVE: 'Partially looking at screen',
    LOOKING_AWAY: 'Looking away',
  }[status] || 'Collecting gaze samples';
}

function emptyPauseAOITracker() {
  return {
    pauseNumber: null,
    startedAt: null,
    startedWallClock: null,
    lastSampleAt: null,
    insideMs: 0,
    validFrames: 0,
    totalFrames: 0,
    finalized: true,
  };
}

// ===============================================================
// Inference client - Hugging Face Space / FastAPI
// ===============================================================
async function classifyWindow(aggregatedFeatures) {
  if (!API_BASE_URL) {
    console.warn('[API] API_BASE_URL is not set.');
    return null;
  }
  const orderedFeatures = {};
  for (const k of FEATURE_ORDER) orderedFeatures[k] = aggregatedFeatures[k] ?? 0;

  // Support both FastAPI (/predict) and HF Gradio Space APIs.
  const attempts = [
    {
      url: `${API_BASE_URL}/predict`,
      body: { agg_features: orderedFeatures },
    },
    {
      url: `${API_BASE_URL}/api/predict`,
      body: { agg_features: orderedFeatures },
    },
    {
      url: `${API_BASE_URL}/api/predict`,
      body: { data: [orderedFeatures] },
    },
  ];

  try {
    for (const attempt of attempts) {
      const res = await fetch(attempt.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(attempt.body),
      });

      if (!res.ok) {
        const text = await res.text();
        console.warn('[API] HTTP error:', attempt.url, res.status, text.slice(0, 200));
        continue;
      }

      const payload = await res.json();

      // FastAPI returns prediction object directly.
      if (payload && payload.Boredom && payload.Engagement && payload.Confusion && payload.Frustration) {
        return payload;
      }

      // Some Gradio variants return { data: [predictionObject] }.
      if (
        payload &&
        Array.isArray(payload.data) &&
        payload.data[0] &&
        payload.data[0].Boredom &&
        payload.data[0].Engagement &&
        payload.data[0].Confusion &&
        payload.data[0].Frustration
      ) {
        return payload.data[0];
      }

      // If response shape differs, keep trying fallbacks.
      console.warn('[API] Unexpected response shape from', attempt.url, payload);
    }

    return null;
  } catch (err) {
    console.error('[API] classify error:', err);
    return null;
  }
}

// ===============================================================
// SUPABASE HELPERS
// ===============================================================
async function sbRequest(table, method, body, params, options = {}) {
  if (!SUPABASE_URL || !SUPABASE_KEY) return null;
  let url = `${SUPABASE_URL}/rest/v1/${table}`;
  if (params) url += '?' + new URLSearchParams(params).toString();
  const extraHeaders = options.headers || {};
  const preferHeader = options.prefer
    ? `${options.prefer},return=representation`
    : 'return=representation';
  const res = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      'apikey': SUPABASE_KEY,
      'Authorization': `Bearer ${SUPABASE_KEY}`,
      'Prefer': preferHeader,
      ...extraHeaders,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Supabase ${res.status}: ${t.slice(0, 200)}`);
  }
  if (res.status === 204) return null;
  const contentType = res.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) return null;
  return res.json();
}

function normalizeCohortRow(row) {
  return {
    id: row.cohort_id,
    access_code: row.access_code,
    created_at: row.created_at,
    activity_type: row.activity_type,
    type: row.activity_type,
    description: row.task_description || '',
  };
}

async function createSession(label = '', cohortId = '', userId = '', activityType = '') {
  const sessionUuid = (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : null;
  const payload = {
    label,
    participant_id: userId,
    cohort_id: cohortId,
    activity_type: activityType,
  };
  if (sessionUuid) {
    payload.session_id = sessionUuid;
  }

  const data = await sbRequest('sessions', 'POST', payload);
  return data?.[0]?.session_id ?? data?.[0]?.id ?? null;
}

async function insertPrediction(sessionUuid, userId, prediction, aggFeatures, pauseAndReflectNumber, cohortId = '', activityType = '') {
  if (!sessionUuid) return;
  const labelsFlat = {}, probsFlat = {};
  let maxConf = 0;
  for (const lbl of LABEL_COLS) {
    if (!prediction[lbl]) continue;
    labelsFlat[lbl] = prediction[lbl].label;
    probsFlat[lbl]  = prediction[lbl].probabilities;
    const conf = prediction[lbl].probabilities[prediction[lbl].label] ?? 0;
    if (conf > maxConf) maxConf = conf;
  }
  const payload = {
    session_id: sessionUuid,
    participant_id: userId || null,
    cohort_id: cohortId || null,
    activity_type: activityType || null,
    pause_and_reflect_number: pauseAndReflectNumber ?? null,
    label: JSON.stringify(labelsFlat),
    confidence: parseFloat(maxConf.toFixed(5)),
    probabilities: probsFlat,
    model_version: MODEL_VERSION,
    raw_data: aggFeatures ?? null,
  };
  await sbRequest('emotion_predictions', 'POST', payload);
}

async function savePauseReflectionAOI(metric) {
  if (!metric?.session_id || !metric?.pause_and_reflect_number) return;
  await sbRequest(
    PAUSE_REFLECTION_AOI_TABLE,
    'POST',
    metric,
    { on_conflict: 'session_id,pause_and_reflect_number' },
    { prefer: 'resolution=merge-duplicates' },
  );
}

async function savePauseReflectionAOIWithRetry(metric, maxAttempts = 3) {
  let lastError = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      await savePauseReflectionAOI(metric);
      return;
    } catch (error) {
      lastError = error;
      if (attempt < maxAttempts) {
        await new Promise(resolve => window.setTimeout(resolve, 400 * (2 ** (attempt - 1))));
      }
    }
  }
  throw lastError;
}

// -- Save manual override to Supabase ---------------------------
// Stores every saved manual edit as a history row. Only the latest edit for the
// same session/cohort/participant/pause bucket/emotion bucket stays status=true.
// If the table doesn't exist yet it'll throw - gracefully caught in the UI.
async function saveManualOverride(
  sessionUuid,
  cohortId,
  bucketLabel,
  labelCol,
  levelPercentages,
  originalPredictions,
  pauseAndReflectNumber,
  justification,
  userId,
) {
  const maxChange = Math.max(...(levelPercentages || []).map((v, i) => Math.abs((Number(v) || 0) - (Number(originalPredictions?.[i]) || 0))), 0);
  const isMajorChange = maxChange > MANUAL_OVERRIDE_CHANGE_THRESHOLD_PERCENT;
  const isMinorChange = maxChange <= MANUAL_OVERRIDE_CHANGE_THRESHOLD_PERCENT;
  const wordsEdited = (justification || '').trim()
    ? (justification || '').trim().split(/\s+/).length
    : 0;

  const overrideAt = new Date().toISOString();
  const pauseFilter = (pauseAndReflectNumber === null || pauseAndReflectNumber === undefined)
    ? 'is.null'
    : `eq.${pauseAndReflectNumber}`;

  const nextPayload = {
    session_id:          sessionUuid,
    cohort_id:            cohortId,
    participant_id:             userId || null,
    pause_and_reflect_number: pauseAndReflectNumber ?? null,
    bucket_label:        bucketLabel,
    label_col:           labelCol,
    level_percentages:   levelPercentages,  // array [low%, high%]
    original_predictions: originalPredictions,
    is_major_change:     isMajorChange,
    is_minor_change:     isMinorChange,
    words_edited:        wordsEdited,
    overridden_at:       overrideAt,
    justification:       justification || null,
    status:              true,
  };

  // Deactivate previously active row(s) for the same bucket/label/session (+ pause bucket).
  await sbRequest(MANUAL_OVERRIDE_TABLE, 'PATCH', {
    status: false,
  }, {
    session_id: `eq.${sessionUuid}`,
    cohort_id: `eq.${cohortId}`,
    participant_id: userId ? `eq.${userId}` : 'is.null',
    bucket_label: `eq.${bucketLabel}`,
    label_col: `eq.${labelCol}`,
    pause_and_reflect_number: pauseFilter,
    status: 'eq.true',
  });

  try {
    return await sbRequest(MANUAL_OVERRIDE_TABLE, 'POST', nextPayload);
  } catch (err) {
    const msg = String(err || '');
    if (!msg.includes('23505') && !/duplicate/i.test(msg)) throw err;

    throw new Error(
      'Manual override history is blocked by an old Supabase unique constraint. Run docs/supabase-manual-overrides-history.sql, then try saving again.',
    );
  }
}

async function clearManualOverride(sessionUuid, cohortId, bucketLabel, labelCol) {
  if (!sessionUuid) return null;
  return sbRequest(MANUAL_OVERRIDE_TABLE, 'PATCH', {
    status: false,
  }, {
    session_id: `eq.${sessionUuid}`,
    cohort_id: `eq.${cohortId}`,
    bucket_label: `eq.${bucketLabel}`,
    label_col: `eq.${labelCol}`,
    status: 'eq.true',
  });
}

async function getManualOverrides(sessionUuid, cohortId) {
  if (!sessionUuid) return [];
  return sbRequest(MANUAL_OVERRIDE_TABLE, 'GET', null, {
    select: 'bucket_label,label_col,level_percentages',
    session_id: `eq.${sessionUuid}`,
    cohort_id: `eq.${cohortId}`,
    status: 'eq.true',
    order: 'overridden_at.desc',
  });
}

async function fetchCohorts() {
  const rows = await sbRequest('login_credentials', 'GET', null, {
    select: 'cohort_id,access_code,created_at,activity_type,task_description,active',
    order: 'created_at.desc',
    limit: '500',
  }) || [];
  return rows.map(normalizeCohortRow);
}

async function createCohortCredential(group) {
  return sbRequest('login_credentials', 'POST', {
    cohort_id: group.id,
    access_code: group.access_code,
    created_at: group.created_at,
    activity_type: group.activity_type,
    task_description: group.description,
    active: true,
  });
}

async function getManualOverrideCount(sessionUuid, cohortId, userId) {
  if (!sessionUuid) return 0;
  const filters = {
    select: 'overridden_at',
    session_id: `eq.${sessionUuid}`,
    cohort_id: `eq.${cohortId}`,
    order: 'overridden_at.desc',
    limit: '10000',
  };
  filters.participant_id = userId ? `eq.${userId}` : 'is.null';
  const rows = await sbRequest(MANUAL_OVERRIDE_TABLE, 'GET', null, filters) || [];
  return Array.isArray(rows) ? rows.length : 0;
}

async function fetchAdminSessionOverview() {
  const sessions = await sbRequest('sessions', 'GET', null, {
    select: 'id,session_id,label,created_at,cohort_id,activity_type,participant_id',
    order: 'created_at.desc',
    limit: '500',
  }) || [];

  const predictionRows = await sbRequest('emotion_predictions', 'GET', null, {
    select: 'session_id,participant_id,cohort_id,activity_type,pause_and_reflect_number,predicted_at',
    order: 'predicted_at.desc',
    limit: '20000',
  }) || [];

  const overrideRows = await sbRequest(MANUAL_OVERRIDE_TABLE, 'GET', null, {
    select: 'session_id,cohort_id,participant_id,pause_and_reflect_number,bucket_label,label_col,is_minor_change,is_major_change,words_edited,justification,overridden_at,status',
    order: 'overridden_at.desc',
    limit: '10000',
  }) || [];

  let aoiRows = [];
  try {
    aoiRows = await sbRequest(PAUSE_REFLECTION_AOI_TABLE, 'GET', null, {
      select: 'session_id,cohort_id,participant_id,pause_and_reflect_number,total_pause_ms,inside_aoi_ms,aoi_percent,valid_gaze_frames,total_gaze_frames,paused_at,resumed_at',
      order: 'paused_at.desc',
      limit: '10000',
    }) || [];
  } catch (error) {
    console.warn('[Supabase] Pause/reflection AOI metrics are unavailable:', error);
  }

  const defaultEditMetrics = () => ({
    manual_override_count: 0,
    max_pause_and_reflect_number: 0,
    par_metrics: {},
    aoi_metrics: {},
    justifications: [],
  });

  const metricsBySession = {};
  const sessionMetaById = {};

  const normalizeIdentity = (sessionIdRaw, userIdRaw) => {
    const sessionId = String(sessionIdRaw || '').trim();
    const userId = String(userIdRaw || '').trim();
    if (sessionId && userId) return `sid:${sessionId}|uid:${userId}`;
    if (sessionId) return `sid:${sessionId}`;
    if (userId) return `uid:${userId}`;
    return null;
  };

  for (const row of sessions) {
    const sessionId = row?.session_id ?? row?.id;
    const identityKey = normalizeIdentity(sessionId, row?.participant_id);
    if (!identityKey) continue;
    sessionMetaById[identityKey] = {
      id: String(sessionId || identityKey),
      label: row.label || '',
      created_at: row.created_at || null,
      cohort_id: row.cohort_id || 'unassigned',
      activity_type: row.activity_type || '',
      participant_id: row.participant_id || '',
    };
  }

  for (const row of predictionRows) {
    const identityKey = normalizeIdentity(row?.session_id, row?.participant_id);
    if (!identityKey) continue;
    if (!metricsBySession[identityKey]) metricsBySession[identityKey] = defaultEditMetrics();

    if (!sessionMetaById[identityKey]) {
      sessionMetaById[identityKey] = {
        id: String(row?.session_id || identityKey),
        label: '',
        created_at: row.predicted_at || null,
        cohort_id: row.cohort_id || 'unassigned',
        activity_type: row.activity_type || '',
        participant_id: row.participant_id || '',
      };
    }

    const m = metricsBySession[identityKey];
    const segmentNumber = Number(row.pause_and_reflect_number ?? null);
    if (!Number.isFinite(segmentNumber) || segmentNumber <= 0) continue;

    if (segmentNumber > m.max_pause_and_reflect_number) {
      m.max_pause_and_reflect_number = segmentNumber;
    }
  }

  for (const row of overrideRows) {
    const identityKey = normalizeIdentity(row?.session_id, row?.participant_id);
    if (!identityKey) continue;
    if (!metricsBySession[identityKey]) metricsBySession[identityKey] = defaultEditMetrics();

    if (!sessionMetaById[identityKey]) {
      sessionMetaById[identityKey] = {
        id: String(row?.session_id || identityKey),
        label: '',
        created_at: row.overridden_at || null,
        cohort_id: row.cohort_id || 'unassigned',
        activity_type: '',
        participant_id: row.participant_id || '',
      };
    }

    const m = metricsBySession[identityKey];
    const segmentNumber = Number(row.pause_and_reflect_number ?? null);
    if (!Number.isFinite(segmentNumber) || segmentNumber <= 0) continue;

    if (segmentNumber > m.max_pause_and_reflect_number) {
      m.max_pause_and_reflect_number = segmentNumber;
    }

    const bucketKey = String(segmentNumber);
    if (!m.par_metrics[bucketKey]) {
      m.par_metrics[bucketKey] = { minor: 0, major: 0, words: 0 };
    }

    const isMinor = row.is_minor_change === true;
    const isMajor = row.is_major_change === true || (!isMinor);
    const words = Math.max(0, Number(row.words_edited) || 0);

    if (isMinor) m.par_metrics[bucketKey].minor += 1;
    if (isMajor) m.par_metrics[bucketKey].major += 1;
    m.par_metrics[bucketKey].words += words;

    const justification = String(row.justification || '').trim();
    if (row?.status === true && justification) {
      m.justifications.push({
        reflection: segmentNumber,
        bucket_label: row.bucket_label || '',
        label_col: row.label_col || '',
        words_edited: words,
        overridden_at: row.overridden_at || null,
        text: justification,
      });
    }

    m.manual_override_count += 1;
  }

  for (const row of aoiRows) {
    const identityKey = normalizeIdentity(row?.session_id, row?.participant_id);
    if (!identityKey) continue;
    if (!metricsBySession[identityKey]) metricsBySession[identityKey] = defaultEditMetrics();

    if (!sessionMetaById[identityKey]) {
      sessionMetaById[identityKey] = {
        id: String(row?.session_id || identityKey),
        label: '',
        created_at: row.paused_at || null,
        cohort_id: row.cohort_id || 'unassigned',
        activity_type: '',
        participant_id: row.participant_id || '',
      };
    }

    const reflectionNumber = Number(row.pause_and_reflect_number);
    if (!Number.isFinite(reflectionNumber) || reflectionNumber <= 0) continue;
    const m = metricsBySession[identityKey];
    m.max_pause_and_reflect_number = Math.max(m.max_pause_and_reflect_number, reflectionNumber);
    m.aoi_metrics[String(reflectionNumber)] = {
      percent: Math.max(0, Math.min(100, Number(row.aoi_percent) || 0)),
      total_pause_ms: Math.max(0, Number(row.total_pause_ms) || 0),
      inside_aoi_ms: Math.max(0, Number(row.inside_aoi_ms) || 0),
      valid_gaze_frames: Math.max(0, Number(row.valid_gaze_frames) || 0),
      total_gaze_frames: Math.max(0, Number(row.total_gaze_frames) || 0),
    };
  }

  const sessionIds = new Set([
    ...Object.keys(sessionMetaById),
    ...Object.keys(metricsBySession),
  ]);

  const normalizedSessions = [...sessionIds].map(identityKey => {
    const meta = sessionMetaById[identityKey] || {
      id: identityKey,
      label: '',
      created_at: null,
      cohort_id: 'unassigned',
      activity_type: '',
      participant_id: '',
    };
    const metrics = metricsBySession[identityKey] || defaultEditMetrics();
    return {
      ...meta,
      ...metrics,
      edit_count: metrics.manual_override_count,
    };
  });

  normalizedSessions.sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0));

  return { sessions: normalizedSessions, metricsBySession };
}

// ===============================================================
// USER ACTION LOGGER
// ===============================================================
async function logEvent(sessionId, cohortId, eventName, eventData = {}, userId = null) {
  try {
    const now = new Date().toISOString();
    const { vlearn_url, learning_url } = eventData || {};
    const modernPayload = {
      session_id: sessionId ?? null,
      cohort_id: cohortId ?? null,
      participant_id: userId ?? eventData?.participant_id ?? null,
      event_name: eventName,
      event_data: {},
      vlearn_url: vlearn_url ?? learning_url ?? null,
      created_at: now,
    };
    return await sbRequest('logs', 'POST', modernPayload);
  } catch (e) {
    console.warn('[logEvent]', eventName, e);
  }
}

// ===============================================================
// TIMELINE CHART (Plotly)
// ===============================================================
function TimelineChart({ history, visibleLabels }) {
  const divRef = useRef(null);
  useEffect(() => {
    if (!divRef.current) return;
    if (!history.length) { Plotly.purge?.(divRef.current); return; }
    const df = [...history].sort((a,b) => new Date(a.predicted_at)-new Date(b.predicted_at));
    const times = df.map(r => r.predicted_at);
    const timesMs = times.map(t => new Date(t).getTime());
    const firstDay = singaporeDayKey(timesMs[0]);
    const hasMultipleDays = timesMs.some(t => singaporeDayKey(t) !== firstDay);

    const fmtAxisTime = t => {
      if (hasMultipleDays) {
        return formatSingaporeTime(t, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
      }
      return formatSingaporeTime(t, { hour: '2-digit', minute: '2-digit' });
    };
    const fmtHoverTime = t => {
      return formatSingaporeTime(t, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
    };

    const tickvals = [];
    const ticktext = [];
    const startMs = timesMs[0];
    const endMs = timesMs[timesMs.length - 1];
    const spanMs = Math.max(1, endMs - startMs);
    const targetTicks = 8;
    const roughStep = spanMs / Math.max(1, targetTicks - 1);
    const stepCandidates = [
      1000, 2000, 5000, 10000, 15000, 30000,
      60000, 120000, 300000, 600000, 900000, 1800000,
      3600000, 7200000, 21600000, 43200000, 86400000,
    ];
    let tickStep = stepCandidates[stepCandidates.length - 1];
    for (const candidate of stepCandidates) {
      if (candidate >= roughStep) {
        tickStep = candidate;
        break;
      }
    }

    tickvals.push(startMs);
    for (let t = startMs + tickStep; t < endMs; t += tickStep) tickvals.push(t);
    if (endMs !== startMs) tickvals.push(endMs);
    for (const tv of tickvals) ticktext.push(fmtAxisTime(tv));

    const mode = df.length < 2 ? 'markers' : 'lines+markers';
    const segmentOrder = [...new Set(df.map(r => (Number.isFinite(r.segment_id) ? r.segment_id : 1)))];
    const traces = LABEL_COLS
      .filter(lbl => (visibleLabels?.[lbl] ?? true) && df.some(r => r[lbl] !== undefined))
      .flatMap(lbl => {
        const perSegment = [];
        for (let si = 0; si < segmentOrder.length; si++) {
          const segId = segmentOrder[si];
          const idxs = [];
          for (let i = 0; i < df.length; i++) {
            const rowSeg = Number.isFinite(df[i].segment_id) ? df[i].segment_id : 1;
            if (rowSeg === segId && df[i][lbl] !== undefined) idxs.push(i);
          }
          if (!idxs.length) continue;
          perSegment.push({
            x: idxs.map(i => timesMs[i]),
            y: idxs.map(i => df[i][lbl]),
            customdata: idxs.map(i => ({
              level: LEVEL_LABELS[df[i][lbl]],
              ts: fmtHoverTime(timesMs[i]),
            })),
            mode,
            connectgaps: false,
            name: lbl,
            showlegend: si === 0,
            line: { color: LABEL_COLORS[lbl], width: 3 },
            marker: { size: 7, color: LABEL_COLORS[lbl] },
            hovertemplate: `<b>${lbl}</b><br>%{customdata.ts}<br>level: %{customdata.level}<extra></extra>`,
          });
        }
        return perSegment;
      });
    const xAxisRange = timesMs.length > 1 ? [startMs, endMs] : [startMs - 1000, startMs + 1000];
    Plotly.react(divRef.current, traces, {
      margin: { l:122,r:10,t:10,b:70 }, height:420,
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      xaxis: {
        type: 'linear',
        range: xAxisRange,
        fixedrange: true,
        showgrid:true,
        gridcolor:'rgba(80,80,100,0.2)',
        showline:true,
        linecolor:'#666',
        linewidth:2,
        zeroline:false,
        zerolinecolor:'#e2e8f0',
        zerolinewidth:2,
        color:'#aaa',
        tickmode:'array',
        tickvals,
        ticktext,
        tickangle:-45,
        automargin:true,
        tickfont:{size:9},
        title:{ text: 'Time', font:{size:11,color:'#aaa'} },
      },
      yaxis: {
        fixedrange: true,
        range:[-0.5,LEVEL_COUNT - 0.5],
        tickmode:'array',
        tickvals:LEVEL_LABELS.map((_, index) => index),
        ticktext:LEVEL_LABELS,
        showgrid:true, gridcolor:'rgba(80,80,100,0.2)', color:'#aaa',
        automargin:true,
        showline:true,
        linecolor:'#666',
        linewidth:2,
        zeroline:false,
        zerolinecolor:'#9ca3af',
        zerolinewidth:2,
        tickfont:{size:12,color:'#ccc'},
        title:{text:'Intensity Level',font:{size:13,color:'#ccc'},standoff:24},
      },
      legend: { orientation:'h', yanchor:'bottom', y:1.12, xanchor:'center', x:0.5, font:{size:11,color:'#ccc'}, itemclick:false, itemdoubleclick:false },
      font: { size:11, color:'#ccc' },
      hovermode: 'x unified',
      hoverlabel: { bgcolor:'rgba(20,20,30,0.95)', bordercolor:'#444', font:{color:'#fff'} },
      dragmode: false,
    }, PLOTLY_STATIC_CONFIG);
  }, [history, visibleLabels]);

  if (!history.length) {
    return React.createElement('div', { className:'chart-placeholder' },
      React.createElement('div', null,
        'Timeline appears after the first ', React.createElement('b',null,`${WINDOW_SECONDS}s`), ' window',
        React.createElement('br'),
        React.createElement('span', {style:{fontSize:'0.7rem'}}, `(${WINDOW_FRAMES} frames at ${CAPTURE_FPS} FPS)`),
      )
    );
  }
  return React.createElement('div', { ref:divRef, style:{width:'100%'} });
}

function computeOverallLevelPercentages(history) {
  const result = {};
  for (const label of LABEL_COLS) {
    const counts = Array(LEVEL_COUNT).fill(0);
    for (const row of history) {
      const level = row[label];
      if (level !== undefined && level !== null && level >= 0 && level < LEVEL_COUNT) counts[level] += 1;
    }
    const total = counts.reduce((s, x) => s + x, 0) || 0;
    result[label] = counts.map(c => total ? +(c / total * 100).toFixed(0) : 0);
  }
  return result;
}

function OverallSplitChart({ history }) {
  const divRef = useRef(null);
  useEffect(() => {
    if (!divRef.current) return;
    if (!history.length) { Plotly.purge?.(divRef.current); return; }
    const levelPercentages = computeOverallLevelPercentages(history);
    const traces = LEVEL_LABELS.map((lvl, li) => ({
      x: LABEL_COLS,
      y: LABEL_COLS.map(lbl => levelPercentages[lbl][li]),
      name: lvl, type: 'bar',
      marker: { color: LEVEL_COLORS[li] },
      text: LABEL_COLS.map(lbl => { const pct = levelPercentages[lbl][li]; return pct > 0 ? `${pct}%` : ''; }),
      textposition: 'inside',
      hovertemplate: `<b>%{x}</b><br>${lvl}: %{y}%<extra></extra>`,
    }));
    Plotly.react(divRef.current, traces, {
      margin:{l:82,r:10,t:30,b:40}, height:260,
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      barmode:'stack',
      xaxis: { title:{ text: 'Emotions', font:{size:11,color:'#aaa'} }, tickfont:{color:'#ccc'} },
      yaxis: { fixedrange:true, range:[0,100], ticksuffix:'%', showline:true, linecolor:'#666', linewidth:2, gridcolor:'rgba(80,80,100,0.2)', color:'#aaa', automargin:true, title:{text:'Percentage of Time (%)',font:{size:12,color:'#ccc'},standoff:20} },
      font:{size:11,color:'#ccc'},
      legend:{ orientation:'h', yanchor:'bottom', y:1.08, xanchor:'center', x:0.5, font:{size:10,color:'#ccc'}, itemclick:false, itemdoubleclick:false },
      dragmode:false,
    }, PLOTLY_STATIC_CONFIG);
  }, [history]);
  if (!history.length) return React.createElement('div', { className:'chart-placeholder' }, 'Overall split will appear after predictions');
  return React.createElement('div', { ref:divRef, style:{width:'100%'} });
}

function bucketLevelPercentages(history, label, bucketMinutes = 5, editableSegmentId = null) {
  const ms = bucketMinutes * 60 * 1000;
  const rawRows = (history || [])
    .map(r => ({
      activeElapsedMs: Number.isFinite(r.active_elapsed_ms) ? r.active_elapsed_ms : null,
      ts: new Date(r.predicted_at).getTime(),
      segmentStartTs: Number.isFinite(r.segment_start_ts) ? r.segment_start_ts : null,
      level: r[label],
      segmentId: Number.isFinite(r.segment_id) ? r.segment_id : 1,
    }))
    .filter(r => r.level !== undefined && r.level !== null && !isNaN(r.ts));
  if (!rawRows.length) {
    return {
      labels: [],
      percentagesPerLevel: Array.from({ length: LEVEL_COUNT }, () => []),
      bucketKeys: [],
      editableMask: [],
      bucketDurationMs: [],
      visibleMask: [],
    };
  }

  const pad = v => String(v).padStart(2, '0');
  const fmtClock = epochMs => {
    const parts = formatSingaporeTime(epochMs, {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).split(':');
    return `${pad(parts[0])}:${pad(parts[1])}`;
  };

  const uniqueSegmentIds = [...new Set(rawRows.map(r => r.segmentId))].sort((a, b) => a - b);
  const labels = [];
  const bucketKeys = [];
  const editableMask = [];
  const bucketDurationMs = [];
  const visibleMask = [];
  const counts = [];

  uniqueSegmentIds.forEach((segmentId, segIdx) => {
    const segmentRows = rawRows.filter(r => r.segmentId === segmentId).sort((a, b) => a.ts - b.ts);
    if (!segmentRows.length) return;

    // Start from the first prediction in this segment.
    const segmentStart = segmentRows[0].ts;
    const segmentEnd = segmentRows[segmentRows.length - 1].ts;
    const firstBoundary = Math.ceil(segmentStart / ms) * ms;

    const boundaries = [segmentStart];
    for (let t = firstBoundary; t <= segmentEnd; t += ms) boundaries.push(t);
    if (boundaries[boundaries.length - 1] < segmentEnd) boundaries.push(segmentEnd);
    const dedupedBoundaries = boundaries.filter((v, i) => i === 0 || v !== boundaries[i - 1]);

    const localBucketCount = dedupedBoundaries.length - 1;
    if (localBucketCount < 1) return;

    const localCounts = Array.from(
      { length: localBucketCount },
      () => Array(LEVEL_COUNT).fill(0),
    );
    for (const r of segmentRows) {
      if (r.level < 0 || r.level >= LEVEL_COUNT) continue;
      let idx = localBucketCount - 1;
      for (let i = 0; i < localBucketCount; i++) {
        if (r.ts >= dedupedBoundaries[i] && r.ts < dedupedBoundaries[i + 1]) {
          idx = i;
          break;
        }
      }
      if (r.ts === dedupedBoundaries[localBucketCount]) idx = localBucketCount - 1;
      localCounts[idx][r.level] += 1;
    }

    for (let i = 0; i < localBucketCount; i++) {
      const start = dedupedBoundaries[i];
      const end = dedupedBoundaries[i + 1];
      labels.push(`${fmtClock(start)}-${fmtClock(end)}`);
      bucketKeys.push(`seg${segmentId}:${start}-${end}`);
      editableMask.push(editableSegmentId !== null && segmentId === editableSegmentId);
      const durationMs = Math.max(1000, end - start);
      bucketDurationMs.push(durationMs);
      visibleMask.push(durationMs >= LABEL_SPLIT_MIN_BUCKET_MS);
      counts.push(localCounts[i]);
    }
  });

  const percentagesPerLevel = Array.from({ length: LEVEL_COUNT }, () => []);
  for (const c of counts) {
    const total = c.reduce((s, x) => s + x, 0) || 0;
    for (let i = 0; i < LEVEL_COUNT; i++) percentagesPerLevel[i].push(total ? +(c[i] / total * 100).toFixed(0) : 0);
  }
  return { labels, percentagesPerLevel, bucketKeys, editableMask, bucketDurationMs, visibleMask };
}

// ===============================================================
// BAR EDIT PANEL - shown on bar click
// ===============================================================
function BarEditPanel({ label, bucketLabel, bucketKey, initialValues, currentValues, onSave, onClose, sessionId, cohortId, pauseAndReflectNumber, requireTextJustification, userId }) {
  // initialValues: one percentage per binary level, summing to 100.
  const getStartValues = () => {
    if (Array.isArray(currentValues) && currentValues.length === LEVEL_COUNT) return [...currentValues];
    return [...initialValues];
  };
  const [vals, setVals] = useState(getStartValues);
  const [justificationText, setJustificationText] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');

  useEffect(() => {
    setVals(getStartValues());
    setJustificationText('');
    setSaveMsg('');
  }, [bucketKey, label, currentValues, initialValues]);

  const sum = vals.reduce((s, v) => s + v, 0);
  const sumOk = Math.abs(sum - 100) <= 1; // allow 1% rounding tolerance
  const startValues = getStartValues();
  const maxChangeFromStart = Math.max(...vals.map((v, i) => Math.abs(v - (startValues[i] ?? 0))));
  const hasChanged = maxChangeFromStart > 0;
  const minRequiredWords = 5;
  const enteredWords = justificationText.trim() ? justificationText.trim().split(/\s+/).length : 0;
  const textRequirementMet = !requireTextJustification || !hasChanged || enteredWords >= minRequiredWords;

  // When a slider changes: adjust the last "free" other slider to keep sum = 100
  function handleChange(idx, newVal) {
    const clamped = Math.max(0, Math.min(100, newVal));
    const next = [...vals];
    next[idx] = clamped;
    // Distribute the remainder proportionally across other sliders
    const others = next.map((v, i) => i === idx ? null : v);
    const otherSum = others.reduce((s, v) => s + (v ?? 0), 0);
    const remainder = 100 - clamped;
    if (otherSum === 0) {
      // Spread evenly
      const share = Math.round(remainder / Math.max(1, LEVEL_COUNT - 1));
      let leftover = remainder;
      const otherIdxs = LEVEL_LABELS.map((_, i) => i).filter(i => i !== idx);
      for (let i = 0; i < LEVEL_COUNT; i++) {
        if (i === idx) continue;
        if (leftover <= 0) { next[i] = 0; continue; }
        const v = i === otherIdxs[otherIdxs.length - 1] ? leftover : share;
        next[i] = Math.min(v, leftover);
        leftover -= next[i];
      }
    } else {
      let assigned = 0;
      const otherIdxs = LEVEL_LABELS.map((_, i) => i).filter(i => i !== idx);
      for (let j = 0; j < otherIdxs.length; j++) {
        const i = otherIdxs[j];
        if (j === otherIdxs.length - 1) {
          next[i] = Math.max(0, remainder - assigned);
        } else {
          const proportion = vals[i] / (otherSum || 1);
          const share = Math.round(remainder * proportion);
          next[i] = Math.max(0, share);
          assigned += next[i];
        }
      }
    }
    setVals(next);
  }

  async function handleSave() {
    setSaving(true);
    setSaveMsg('');
    if (requireTextJustification && hasChanged && enteredWords < minRequiredWords) {
      setSaveMsg(`Warning: Please enter at least ${minRequiredWords} words explaining your edit.`);
      setSaving(false);
      return;
    }
    try {
      await saveManualOverride(
        sessionId,
        cohortId,
        bucketKey,
        label,
        vals,
        initialValues,
        pauseAndReflectNumber,
        justificationText,
        userId,
      );
      await logEvent(sessionId, cohortId, 'changes_made', {
        label,
        bucket_key: bucketKey,
        new_values: vals,
        justification: justificationText,
      }, userId);
      setSaveMsg('');
      onSave(vals);
    } catch (err) {
      setSaveMsg('Warning: ' + String(err));
    } finally {
      setSaving(false);
    }
  }



  const e = React.createElement;
  return e('div', { className:'bar-edit-overlay' },
    e('h4', null,
      `Edit - ${label} - ${bucketLabel}`,
      e('button', { onClick: onClose, title:'Close' }, 'x')
    ),
    e('div', { className:'edit-level-row', style:{ marginBottom:'0.6rem', paddingBottom:'0.4rem', borderBottom:'1px solid #2d3350' } },
      e('div', { style:{ width:'16px' } }),
      e('span', { style:{ width:'48px' } }),
      e('div', { style:{ flex:1 } }),
      e('span', { style:{ width:'36px', textAlign:'right', fontFamily:'var(--mono)', fontSize:'0.68rem', color:'var(--muted)', textTransform:'uppercase', letterSpacing:'0.04em', opacity:0.6 } }, 'AI'),
      e('span', { style:{ width:'36px', textAlign:'right', fontFamily:'var(--mono)', fontSize:'0.68rem', color:'var(--text)', textTransform:'uppercase', letterSpacing:'0.04em' } }, 'You'),
    ),
    LEVEL_LABELS.map((lvl, i) =>
      e('div', { key:lvl, className:'edit-level-row' },
        e('div', { className:'edit-level-dot', style:{ background: LEVEL_COLORS[i] } }),
        e('span', { className:'edit-level-name' }, lvl),
        e('div', { className:'edit-level-slider-container' },
          e('div', { className:'edit-level-slider-stack' },
            e('input', {
              type:'range', min:0, max:100, step:1,
              className:'edit-level-slider-ai',
              value: initialValues[i],
              disabled: true,
            }),
            e('input', {
              type:'range', min:0, max:100, step:1,
              className:'edit-level-slider-edit',
              value: vals[i],
              style: { '--thumb-color': LEVEL_COLORS[i] },
              onChange: ev => handleChange(i, parseInt(ev.target.value, 10)),
            }),
          ),
        ),
        e('span', { className:'edit-level-val-orig' }, `${initialValues[i]}%`),
        e('span', { className:'edit-level-val' }, `${vals[i]}%`),
      )
    ),
    e('div', { className:'edit-sum-row' },
      e('span', { style:{color:'var(--muted)'} }, 'Sum'),
      e('span', { className:`edit-sum-val${sumOk ? '' : ' bad'}` }, `${sum}% ${sumOk ? 'OK' : '- must be 100%'}`),
    ),
    requireTextJustification ? e('div', { style:{ marginTop:'0.55rem' } },
      e('div', {
        style:{
          fontFamily:'var(--mono)',
          fontSize:'0.66rem',
          color:'var(--subtext)',
          marginBottom:'0.3rem',
        }
      },
      hasChanged
        ? `Required explanation: ${minRequiredWords}+ words (${enteredWords} entered)`
        : 'No slider change detected'
      ),
      e('textarea', {
        value: justificationText,
        onChange: ev => setJustificationText(ev.target.value),
        placeholder:'In 1–2 sentences, briefly explain the reasoning behind your adjustment (minimum 5 words).',
        rows: 3,
        style:{
          width:'100%',
          resize:'vertical',
          minHeight:'62px',
          background:'#0f1220',
          color:'var(--text)',
          border:`1px solid ${hasChanged && !textRequirementMet ? 'var(--red)' : 'var(--border)'}`,
          borderRadius:'6px',
          padding:'0.45rem 0.5rem',
          fontFamily:'var(--sans)',
          fontSize:'0.74rem',
          outline:'none',
        }
      })
    ) : null,
    e('div', { className:'edit-actions' },
      e('button', { className:'btn-cancel', onClick:onClose }, 'Cancel'),
      e('button', {
        className:'btn-save',
        disabled: !hasChanged || !sumOk || saving || !textRequirementMet,
        onClick: handleSave,
      }, saving ? 'Saving...' : 'Save'),
    ),
    saveMsg ? e('div', { className:'edit-saving-msg' }, saveMsg) : null,
  );
}

// ===============================================================
// LabelSplitCharts - with bar click -> edit overlay
// ===============================================================
function LabelSplitCharts({ history, sessionId, cohortId, overrides, setOverrides, editableSegmentId, pauseAndReflectNumber, canEdit, requireTextJustification, userId, onManualEditSaved }) {
  const chartRefs    = useRef([]);
  // editState: { labelIdx, bucketIdx, bucketLabel, initialValues }
  const [editState, setEditState] = useState(null);

  // Build chart data for label at idx
  function getChartData(label) {
    const base = bucketLevelPercentages(history, label, 5, editableSegmentId);
    // Apply any manual overrides
    const ppl = base.percentagesPerLevel.map(arr => [...arr]);
    base.bucketKeys.forEach((bucketKey, bi) => {
      const key = `${label}:${bucketKey}`;
      if (overrides[key]) {
        for (let li = 0; li < LEVEL_COUNT; li++) ppl[li][bi] = overrides[key][li];
      }
    });
    return { ...base, percentagesPerLevel: ppl };
  }

  function renderChart(label, idx, chartData) {
    const div = chartRefs.current[idx];
    if (!div) return;
    const { labels, percentagesPerLevel, editableMask, bucketDurationMs, visibleMask } = chartData;
    if (!labels.length) { Plotly.purge?.(div); return; }

    const durationUnit = 5 * 60 * 1000;
    const normalizedWidths = bucketDurationMs.map(d => d / durationUnit);
    const xPositions = [];
    let cursor = 0;
    for (let i = 0; i < normalizedWidths.length; i++) {
      const w = normalizedWidths[i];
      xPositions.push(cursor + w / 2);
      cursor += w;
    }

    const tickvals = xPositions;
    const ticktext = labels.map((l, i) => visibleMask?.[i] ? l : '');
    const lockAnnotations = labels
      .map((lbl, bi) => {
        if (!visibleMask?.[bi]) return null;
        if (editableMask?.[bi]) return null;
        return {
          x: xPositions[bi],
          y: 100,
          xref: 'x',
          yref: 'y',
          text: '\uD83D\uDD12',
          showarrow: false,
          yshift: 10,
          font: { size: 16 },
        };
      })
      .filter(Boolean);

    const traces = LEVEL_LABELS.map((lvl, li) => ({
      x: xPositions,
      y: labels.map((lbl, bi) => (visibleMask?.[bi] ? percentagesPerLevel[li][bi] : null)),
      width: normalizedWidths.map(w => w * 0.92),
      customdata: labels,
      name: lvl, type: 'bar',
      marker: {
        color: LEVEL_COLORS[li],
        line: { color: 'rgba(0,0,0,0)', width: 0 },
        opacity: labels.map((lbl, bi) => {
          if (!visibleMask?.[bi]) return 0;
          return 1;
        }),
      },
      text: labels.map((lbl, bi) => (visibleMask?.[bi] ? `${percentagesPerLevel[li][bi]}%` : '')),
      textposition: 'auto',
      textfont: {
        color: li === 1 ? '#0b1220' : '#f8fafc',
        size: 10,
      },
      hovertemplate: `<b>${label} - ${lvl}</b><br>%{customdata}: %{y}%<extra></extra>`,
    }));

    Plotly.react(div, traces, {
      margin:{l:82,r:10,t:30,b:60}, height:220,
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      barmode:'stack',
      xaxis:{
        title:{ text: '5 min windows', font:{size:11,color:'#aaa'} },
        fixedrange:true,
        tickfont:{color:'#ccc'},
        tickangle:-45,
        automargin:true,
        tickmode:'array',
        tickvals,
        ticktext,
      },
      yaxis:{ fixedrange:true, range:[0,100], ticksuffix:'%', showline:true, linecolor:'#666', linewidth:2, gridcolor:'rgba(80,80,100,0.2)', color:'#aaa', automargin:true, title:{text:'Percentage of Time (%)',font:{size:11,color:'#ccc'},standoff:16} },
      font:{size:11, color:'#ccc'},
      legend:{ itemclick:false, itemdoubleclick:false },
      annotations: lockAnnotations,
      dragmode:false,
    }, PLOTLY_EDIT_CONFIG);

    // Attach click handler (remove previous first to avoid stacking on re-renders)
    div.removeAllListeners('plotly_click');
    div.on('plotly_click', (data) => {
      if (!canEdit) return;
      const pt = data.points?.[0];
      if (!pt) return;
      const bucketIdx   = pt.pointIndex;
      const baseChartData = bucketLevelPercentages(history, label, 5, editableSegmentId);
      const chartData2 = getChartData(label);
      const bucketLabel = chartData2.labels[bucketIdx];
      if (!chartData2.visibleMask?.[bucketIdx]) return;
      if (!chartData2.editableMask[bucketIdx]) return;
      const bucketKey = chartData2.bucketKeys[bucketIdx];
      const initialValues = LEVEL_LABELS.map((_, li) => baseChartData.percentagesPerLevel?.[li]?.[bucketIdx] ?? 0);
      const currentValues = LEVEL_LABELS.map((_, li) => chartData2.percentagesPerLevel?.[li]?.[bucketIdx] ?? 0);
      setEditState({ labelIdx: idx, label, bucketIdx, bucketLabel, bucketKey, initialValues, currentValues });
    });
  }

  useEffect(() => {
    LABEL_COLS.forEach((label, idx) => renderChart(label, idx, getChartData(label)));
  }, [history, overrides, editableSegmentId, canEdit]);

  function handleSave(newVals) {
    if (!editState) return;
    const { label, bucketKey } = editState;
    const key = `${label}:${bucketKey}`;
    setOverrides(prev => ({ ...prev, [key]: newVals }));
    onManualEditSaved?.();
    setEditState(null);
  }

  const e = React.createElement;
  return e('div', null,
    e('div', {
      style:{display:'grid', gridTemplateColumns:'repeat(2,minmax(0,1fr))', gap:'0.75rem', marginBottom:'0.75rem'}
    },
      LABEL_COLS.map((label, idx) => {
        const isEditing = editState?.labelIdx === idx;
        return e('div', { key:label, style:{ position:'relative' } },
          e('div', { className:'panel', style:{padding:'0.6rem'} },
            e('div', { className:'panel-header' }, panelTitle(`${label} split`)),
            e('div', { style:{fontSize:'0.72rem', color:'#9ba3c4', marginTop:'0.15rem', padding:'0 0.1rem'} }, '5 Min Windows'),
            canEdit
              ? e('div', { className:'chart-click-hint' }, 'Click a bar to contest AI predictions you disagree with')
              : e('div', { className:'chart-click-hint' }, 'Editing disabled for your assigned cohort'),
            e('div', {
              ref: el => chartRefs.current[idx] = el,
              style:{
                width:'100%',
                minHeight:'160px',
                marginTop:'0.5rem',
                cursor: canEdit ? 'pointer' : 'default',
              },
            }),
          ),
          // Edit overlay - positioned above the card
          isEditing ? e('div', {
            style:{ position:'absolute', top:0, left:'50%', transform:'translateX(-50%)', zIndex:30, width:'290px' }
          },
            e(BarEditPanel, {
              label,
              bucketLabel: editState.bucketLabel,
              bucketKey: editState.bucketKey,
              initialValues: editState.initialValues,
              currentValues: editState.currentValues,
              sessionId,
              cohortId,
              pauseAndReflectNumber,
              requireTextJustification,
              userId,
              onSave: handleSave,
              onClose: () => setEditState(null),
            })
          ) : null,
        );
      })
    )
  );
}

// ===============================================================
// MAIN APP
// ===============================================================
function App() {
  const [contentUrl,   setContentUrl]   = useState('');
  const [running,      setRunning]      = useState(false);
  const [paused,       setPaused]       = useState(false);
  const [userId,       setUserId]       = useState('');
  const [cohortAccessCodeInput, setCohortAccessCodeInput] = useState('');
  const [authError,    setAuthError]    = useState('');
  const [authReady,    setAuthReady]    = useState(false);
  const [isAdminMode,  setIsAdminMode]  = useState(false);
  const [cohortIds, setCohorts] = useState([]);
  const [activeCohort,  setActiveCohort]  = useState(null);
  const [newCohortId, setNewCohortId] = useState('');
  const [newActivityType, setNewActivityType] = useState(ACTIVITY_TYPES.WITHOUT_EDIT);
  const [newTaskDescription, setNewTaskDescription] = useState('');
  const [adminInfoMsg, setAdminInfoMsg] = useState('');
  const [adminError,   setAdminError]   = useState('');
  const [adminLoading, setAdminLoading] = useState(false);
  const [adminSessions, setAdminSessions] = useState([]);
  const [selectedAdminCohortId, setSelectedAdminCohortId] = useState(null);
  const [adminTextModal, setAdminTextModal] = useState(null);
  const [history,      setHistory]      = useState([]);
  const [currentPreds, setCurrentPreds] = useState(null);
  const [frameCount,   setFrameCount]   = useState(0);
  const [windowCount,  setWindowCount]  = useState(0);
  const [sessionId,    setSessionId]    = useState(null);
  const [lastPredTime, setLastPredTime] = useState(null);
  const [lastError,    setLastError]    = useState(null);
  const [noFace,       setNoFace]       = useState(false);
  const [cameraReady,  setCameraReady]  = useState(false);
  const [monitorMode,  setMonitorMode]  = useState(null);
  const [monitorNotice, setMonitorNotice] = useState('');
  const [processingStatus, setProcessingStatus] = useState('Waiting');
  const [bufferLen,    setBufferLen]    = useState(0);
  const [modelReady,   setModelReady]   = useState(false);
  const [configWarn,   setConfigWarn]   = useState('');
  const [checkedLabels, setCheckedLabels] = useState({Boredom:false, Engagement:false, Confusion:false, Frustration:false});
  const [overrides,    setOverrides]    = useState({});
  const [manualEditCount, setManualEditCount] = useState(0);
  const [currentPauseAOIPercent, setCurrentPauseAOIPercent] = useState(null);

  const videoRef     = useRef(null);
  const canvasRef    = useRef(null);
  const mainVideoRef = useRef(null);
  const mainCanvasRef= useRef(null);
  const landmarkerRef= useRef(null);
  const streamRef    = useRef(null);
  const frameBuffer  = useRef([]);
  const gazeBuffer   = useRef([]);
  const pauseAOIRef  = useRef(emptyPauseAOITracker());
  const currentGazeAOIRef = useRef(null);
  const gazeStatsRef = useRef({
    insideRatio: null,
    outsideRatio: null,
    trackingCoverage: 0,
    validFrames: 0,
    totalFrames: 0,
    status: 'WAITING',
  });
  const throttleRef  = useRef(0);
  const processingGenerationRef = useRef(0);
  const runningRef   = useRef(false);
  const pausedRef    = useRef(false);
  const rafRef       = useRef(null);
  const processingWindowRef = useRef(window);
  const monitorWindowRef = useRef(null);
  const monitorElementsRef = useRef(null);
  const monitorClosingRef = useRef(false);
  const scheduleFrameRef = useRef(callback => window.requestAnimationFrame(callback));
  const cancelFrameRef = useRef(id => window.cancelAnimationFrame(id));
  const pauseActionRef = useRef(() => {});
  const stopActionRef = useRef(() => {});
  const monitorClosedActionRef = useRef(() => {});
  const drawUtils    = useRef(null);
  const sessionRef   = useRef(null);
  const userIdRef    = useRef('');
  const activeCohortRef = useRef(null);
  const activeElapsedBeforePauseRef = useRef(0);
  const activeRunStartedAtRef = useRef(null);
  const activeSegmentRef = useRef(1);
  const segmentStartedAtWallClockRef = useRef(Date.now());
  const editPolicy = getEditPolicyByType(activeCohort?.activity_type || activeCohort?.type || ACTIVITY_TYPES.WITHOUT_EDIT);

  useEffect(() => { userIdRef.current = userId; }, [userId]);
  useEffect(() => { activeCohortRef.current = activeCohort; }, [activeCohort]);

  useEffect(() => {
    if (!paused) return undefined;
    let secondFrame = null;
    const firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => {
        document.querySelectorAll('.reflection-layout .js-plotly-plot').forEach(plot => {
          Plotly.Plots?.resize?.(plot);
        });
      });
    });
    return () => {
      window.cancelAnimationFrame(firstFrame);
      if (secondFrame !== null) window.cancelAnimationFrame(secondFrame);
    };
  }, [paused]);

  const refreshAdminOverview = useCallback(async () => {
    setAdminLoading(true);
    setAdminError('');
    try {
      const snapshot = await fetchAdminSessionOverview();
      setAdminSessions(snapshot.sessions || []);
    } catch (e) {
      setAdminError('Could not load admin session analytics: ' + String(e));
      setAdminSessions([]);
    } finally {
      setAdminLoading(false);
    }
  }, []);

  const refreshSessionGroups = useCallback(async () => {
    try {
      const rows = await fetchCohorts();
      setCohorts(rows);
    } catch (e) {
      setCohorts([]);
      setAdminError('Could not load login credentials: ' + String(e));
    }
  }, []);

  useEffect(() => {
    refreshSessionGroups();
  }, [refreshSessionGroups]);

  useEffect(() => {
    if (isAdminMode) {
      refreshAdminOverview();
      refreshSessionGroups();
    }
  }, [isAdminMode, refreshAdminOverview, refreshSessionGroups]);

  useEffect(() => {
    const missing = [];
    if (!SUPABASE_URL)     missing.push('SUPABASE_URL');
    if (!SUPABASE_KEY)     missing.push('SUPABASE_KEY');
    if (!API_BASE_URL)     missing.push('HF_SPACE_URL');
    if (missing.length) setConfigWarn(`Not configured: ${missing.join(', ')} - edit the CONFIG block`);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!API_BASE_URL) return;
      try {
        const zeroFeatures = {};
        for (const k of FEATURE_ORDER) zeroFeatures[k] = 0;
        const result = await classifyWindow(zeroFeatures);
        if (!cancelled) {
          if (result) {
            setCurrentPreds(result);
            setLastError(null);
          } else {
            setLastError('HF Space API returned no prediction during the startup check.');
          }
        }
      } catch (e) {
        if (!cancelled) setLastError('[HF Space] startup check failed: ' + String(e));
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        const lm = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU',
          },
          outputFaceBlendshapes: true,
          runningMode: 'VIDEO',
          numFaces: 1,
        });
        if (!cancelled) { landmarkerRef.current = lm; setModelReady(true); }
      } catch(e) {
        if (!cancelled) setLastError('MediaPipe load failed: ' + e.message);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const handleCreateCohort = useCallback(async () => {
    const requestedId = (newCohortId || '').trim();
    if (!requestedId) {
      setAdminError('Cohort ID is required.');
      return;
    }
    if (cohortIds.some(g => String(g.id).toLowerCase() === requestedId.toLowerCase())) {
      setAdminError('Cohort ID already exists. Please use a unique value.');
      return;
    }
    const now = new Date().toISOString();
    const group = {
      id: requestedId,
      activity_type: newActivityType,
      type: newActivityType,
      access_code: generateAccessCode(),
      description: (newTaskDescription || '').trim(),
      created_at: now,
    };
    try {
      setAdminError('');
      await createCohortCredential(group);
      await refreshSessionGroups();
      setNewCohortId('');
      setNewTaskDescription('');
    } catch (e) {
      setAdminError('Failed to create login credential: ' + String(e));
    }
  }, [newCohortId, newActivityType, newTaskDescription, refreshSessionGroups, cohortIds]);

  const handleEnterWithAccessCode = useCallback(async () => {
    const uid = (userId || '').trim();
    const enteredAccessCode = (cohortAccessCodeInput || '').trim();
    if (!uid) {
      setAuthError('Participant ID is required.');
      return;
    }
    if (!/^\d+$/.test(uid)) {
      setAuthError('Participant ID must contain numbers only.');
      return;
    }
    if (!enteredAccessCode) {
      setAuthError('Access code is required.');
      return;
    }
    if (ADMIN_USER_ID && ADMIN_PASSWORD && uid === ADMIN_USER_ID && enteredAccessCode === ADMIN_PASSWORD) {
      setAuthError('');
      setAuthReady(false);
      setIsAdminMode(true);
      setActiveCohort(null);
      setSessionId(null);
      sessionRef.current = null;
      await Promise.all([refreshAdminOverview(), refreshSessionGroups()]);
      return;
    }
    const group = cohortIds.find(g => (g.access_code || '').trim() === enteredAccessCode);
    if (!group) {
      setAuthError('Invalid access code. Access denied.');
      return;
    }
    try {
      const newSessionId = await createSession(uid, group.id, uid, group.activity_type || group.type || '');
      setSessionId(newSessionId);
      sessionRef.current = newSessionId;
      await logEvent(newSessionId, group.id, 'session_login_success', {}, uid);
    } catch (e) {
      setAuthError('Login succeeded but session row creation failed: ' + String(e));
      return;
    }
    setActiveCohort(group);
    setIsAdminMode(false);
    setAuthError('');
    setAuthReady(true);
  }, [userId, cohortAccessCodeInput, cohortIds, refreshAdminOverview, refreshSessionGroups]);

  const restoreMainProcessingSurface = useCallback(() => {
    processingWindowRef.current = window;
    scheduleFrameRef.current = callback => window.requestAnimationFrame(callback);
    cancelFrameRef.current = id => window.cancelAnimationFrame(id);
    videoRef.current = mainVideoRef.current;
    canvasRef.current = mainCanvasRef.current;
    if (mainVideoRef.current && streamRef.current) {
      mainVideoRef.current.srcObject = streamRef.current;
      mainVideoRef.current.play().catch(() => {});
    }
    if (mainCanvasRef.current) {
      drawUtils.current = new DrawingUtils(mainCanvasRef.current.getContext('2d'));
    }
  }, []);

  const closeMonitorWindow = useCallback(() => {
    const monitorWindow = monitorWindowRef.current;
    monitorClosingRef.current = true;
    monitorWindowRef.current = null;
    monitorElementsRef.current = null;
    setMonitorMode(null);
    restoreMainProcessingSurface();
    if (monitorWindow && !monitorWindow.closed) monitorWindow.close();
    window.setTimeout(() => { monitorClosingRef.current = false; }, 0);
  }, [restoreMainProcessingSurface]);

  const buildMonitorWindow = useCallback((monitorWindow, mode) => {
    const doc = monitorWindow.document;
    doc.title = 'OELM';
    doc.documentElement.style.background = '#0a0c12';
    doc.body.replaceChildren();

    const style = doc.createElement('style');
    style.textContent = `
      * { box-sizing:border-box; }
      html, body { width:100%; height:100%; overflow:hidden; }
      body { margin:0; background:#0a0c12; color:#e2e8f0; font-family:Arial,sans-serif; }
      .monitor { width:100%; height:100%; display:flex; align-items:center; justify-content:center; background:linear-gradient(145deg,#111827,#0d111b); }
      .heading { display:flex; align-items:center; gap:5px; font-size:10px; font-weight:700; letter-spacing:.03em; }
      .dot { width:6px; height:6px; border-radius:50%; background:#6b7280; flex:none; }
      .dot.live { background:#10b981; box-shadow:0 0 5px rgba(16,185,129,.8); }
      .dot.paused { background:#f59e0b; box-shadow:0 0 5px rgba(245,158,11,.65); }
      .processing-media { position:fixed; left:0; bottom:0; width:2px; height:2px; opacity:.01; overflow:hidden; pointer-events:none; }
      .processing-media video, .processing-media canvas { position:absolute; width:2px; height:2px; }
    `;
    doc.head.appendChild(style);

    const shell = doc.createElement('main');
    shell.className = 'monitor';
    const heading = doc.createElement('div');
    heading.className = 'heading';
    const dot = doc.createElement('span');
    dot.className = 'dot';
    const title = doc.createElement('span');
    title.textContent = 'OELM';
    heading.append(dot, title);

    const media = doc.createElement('div');
    media.className = 'processing-media';
    const processingVideo = document.createElement('video');
    processingVideo.muted = true;
    processingVideo.autoplay = true;
    processingVideo.playsInline = true;
    const processingCanvas = document.createElement('canvas');
    media.append(processingVideo, processingCanvas);
    shell.append(heading, media);
    doc.body.appendChild(shell);

    monitorWindowRef.current = monitorWindow;
    processingWindowRef.current = monitorWindow;
    scheduleFrameRef.current = callback => monitorWindow.requestAnimationFrame(callback);
    cancelFrameRef.current = id => monitorWindow.cancelAnimationFrame(id);
    videoRef.current = processingVideo;
    canvasRef.current = processingCanvas;
    drawUtils.current = new DrawingUtils(processingCanvas.getContext('2d'));
    if (streamRef.current) {
      processingVideo.srcObject = streamRef.current;
      processingVideo.play().catch(() => {});
    }
    monitorElementsRef.current = { dot };
    setMonitorMode(mode);
    setMonitorNotice('');

    monitorWindow.addEventListener('pagehide', () => {
      if (monitorWindowRef.current === monitorWindow && !monitorClosingRef.current) {
        monitorClosedActionRef.current();
      }
    }, { once:true });
  }, []);

  const openMonitorWindow = useCallback(async () => {
    const existing = monitorWindowRef.current;
    if (existing && !existing.closed) {
      existing.focus();
      return true;
    }

    try {
      if ('documentPictureInPicture' in window) {
        const pipWindow = await window.documentPictureInPicture.requestWindow({
          width: 80,
          height: 30,
          disallowReturnToOpener: true,
          preferInitialWindowPlacement: true,
        });
        buildMonitorWindow(pipWindow, 'picture-in-picture');
        try {
          pipWindow.moveTo?.(
            Math.max(4, window.screen.availWidth - pipWindow.outerWidth - 4),
            4,
          );
        } catch (_) {}
        return true;
      }

      const popupWidth = 100;
      const popupLeft = Math.max(4, window.screen.availWidth - popupWidth - 4);
      const popup = window.open(
        '',
        'oelm_monitor_window',
        `popup=yes,width=${popupWidth},height=55,left=${popupLeft},top=4,resizable=yes`,
      );
      if (!popup) {
        setLastError('The monitoring window was blocked. Allow pop-ups for this site and try again.');
        return false;
      }
      buildMonitorWindow(popup, 'popup');
      popup.focus();
      return true;
    } catch (error) {
      setLastError('Could not open the floating monitor: ' + (error?.message || String(error)));
      return false;
    }
  }, [buildMonitorWindow]);

  const processFrame = useCallback(async () => {
    const frameGeneration = processingGenerationRef.current;
    const emotionMode = runningRef.current && !pausedRef.current;
    const gazeMode = pausedRef.current;
    if (!emotionMode && !gazeMode) return;
    const video = videoRef.current, canvas = canvasRef.current, landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker) {
      setProcessingStatus('Waiting for processor');
      rafRef.current = scheduleFrameRef.current(processFrame); return;
    }
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      setProcessingStatus('Waiting for camera frames');
      rafRef.current = scheduleFrameRef.current(processFrame); return;
    }
    const now = performance.now();
    if (now - throttleRef.current < 1000 / CAPTURE_FPS) {
      rafRef.current = scheduleFrameRef.current(processFrame); return;
    }
    throttleRef.current = now;
    let result;
    try { result = landmarker.detectForVideo(video, now); }
    catch (error) {
      const message = error?.message || String(error);
      setProcessingStatus('Processing error');
      setLastError(`MediaPipe frame processing failed: ${message}`);
      rafRef.current = scheduleFrameRef.current(processFrame); return;
    }

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (result.faceLandmarks?.length && drawUtils.current) {
      for (const lms of result.faceLandmarks)
        drawUtils.current.drawConnectors(lms, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color:'#0f4', lineWidth:0.5, opacity:0.2 });
    }

    if (gazeMode) {
      const gaze = estimateGazeAOI(result.faceLandmarks?.[0]);
      const pauseTracker = pauseAOIRef.current;
      if (pauseTracker.startedAt !== null && !pauseTracker.finalized) {
        const rawDeltaMs = Math.max(0, now - (pauseTracker.lastSampleAt ?? pauseTracker.startedAt));
        // Credit only observed, regularly sampled time inside the AOI. Long gaps
        // and missing-face samples remain part of total pause time but add no AOI time.
        const creditedDeltaMs = Math.min(rawDeltaMs, (1000 / CAPTURE_FPS) * 2);
        if (gaze?.inside_aoi) pauseTracker.insideMs += creditedDeltaMs;
        pauseTracker.lastSampleAt = now;
        pauseTracker.totalFrames += 1;
        if (gaze) pauseTracker.validFrames += 1;
        const totalPauseMs = Math.max(1, now - pauseTracker.startedAt);
        setCurrentPauseAOIPercent(Math.min(100, pauseTracker.insideMs / totalPauseMs * 100));
      }
      gazeBuffer.current.push(gaze);
      currentGazeAOIRef.current = gaze;
      setNoFace(!gaze);
      setProcessingStatus(gaze ? 'Checking dashboard attention' : 'Looking for gaze');
      setFrameCount(c => c + 1);
      setBufferLen(gazeBuffer.current.length);

      if (gazeBuffer.current.length >= WINDOW_FRAMES) {
        const gazeSnapshot = gazeBuffer.current.slice(0, WINDOW_FRAMES);
        gazeBuffer.current = gazeBuffer.current.slice(WINDOW_FRAMES);
        const gazeSummary = summarizeGazeWindow(gazeSnapshot);
        gazeStatsRef.current = {
          insideRatio: gazeSummary.inside_aoi_ratio,
          outsideRatio: gazeSummary.outside_aoi_ratio,
          trackingCoverage: gazeSummary.tracking_coverage,
          validFrames: gazeSummary.valid_gaze_frames,
          totalFrames: gazeSummary.total_gaze_frames,
          status: gazeSummary.attention_status,
        };
        setBufferLen(gazeBuffer.current.length);
      }

      rafRef.current = scheduleFrameRef.current(processFrame); return;
    }

    // Learning mode is emotion-only. Gaze/AOI sampling is intentionally reserved
    // for the Pause and Reflect dashboard so the two validators never overlap.
    const features = extractBlendshapes(result);
    if (features === null) {
      setNoFace(true);
      setProcessingStatus('Looking for face');
      rafRef.current = scheduleFrameRef.current(processFrame); return;
    }
    setNoFace(false);
    setProcessingStatus('Running emotion prediction');
    setLastError(previous => previous?.startsWith('MediaPipe frame processing failed:') ? null : previous);
    setFrameCount(c => c + 1);
    frameBuffer.current.push(features);
    const len = frameBuffer.current.length;
    setBufferLen(len);

    if (len >= WINDOW_FRAMES) {
      const snapshot = frameBuffer.current.slice();
      frameBuffer.current = []; setBufferLen(0);
      const agg = aggregateWindow(snapshot);
      if (!agg) { rafRef.current = scheduleFrameRef.current(processFrame); return; }
      let prediction = null;
      try { prediction = await classifyWindow(agg); }
      catch(e) {
        if (frameGeneration === processingGenerationRef.current) {
          setLastError('[HF Space] ' + String(e));
        }
      }
      // A pause, resume, or stop happened while the API request was in flight.
      // The newly selected processing mode owns the frame loop now.
      if (frameGeneration !== processingGenerationRef.current) return;
      // Ignore a response that arrives after the learner has entered reflection.
      if (prediction && runningRef.current && !pausedRef.current) {
        const currentCohort = activeCohortRef.current;
        const currentUserId = userIdRef.current;
        setCurrentPreds(prediction);
        setWindowCount(c => c + 1);
        setLastPredTime(formatSingaporeTime(Date.now(), { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
        const ts = new Date().toISOString();
        const activeElapsedMs = Math.round(
          activeElapsedBeforePauseRef.current +
          Math.max(0, now - (activeRunStartedAtRef.current ?? now))
        );
        const row = { predicted_at: ts, active_elapsed_ms: activeElapsedMs };
        row.segment_id = activeSegmentRef.current;
        row.pause_and_reflect_number = activeSegmentRef.current;
        row.segment_start_ts = segmentStartedAtWallClockRef.current;
        for (const lbl of LABEL_COLS) if (prediction[lbl] !== undefined) row[lbl] = prediction[lbl].label;
        setHistory(h => [...h.slice(-179), row]);
        insertPrediction(
          sessionRef.current,
          currentUserId,
          prediction,
          agg,
          activeSegmentRef.current,
          currentCohort?.id || null,
          currentCohort?.activity_type || currentCohort?.type || null,
        )
          .catch(e => setLastError('[supabase] insertPrediction: ' + String(e)));
      }
    }
    rafRef.current = scheduleFrameRef.current(processFrame);
  }, []);

  const handleStart = useCallback(async () => {
    if (!modelReady) { setLastError('MediaPipe model still loading.'); return; }
    const monitorOpened = await openMonitorWindow();
    if (!monitorOpened) return;
    setLastError(null); setProcessingStatus('Starting'); setHistory([]); setCurrentPreds(null);
    setFrameCount(0); setWindowCount(0); setBufferLen(0);
    currentGazeAOIRef.current = null;
    gazeStatsRef.current = {
      insideRatio: null,
      outsideRatio: null,
      trackingCoverage: 0,
      validFrames: 0,
      totalFrames: 0,
      status: 'WAITING',
    };
    setOverrides({});
    setManualEditCount(0);
    pauseAOIRef.current = emptyPauseAOITracker();
    setCurrentPauseAOIPercent(null);
    frameBuffer.current = []; gazeBuffer.current = []; throttleRef.current = 0;
    processingGenerationRef.current += 1;
    activeElapsedBeforePauseRef.current = 0;
    activeRunStartedAtRef.current = performance.now();
    activeSegmentRef.current = 1;
    segmentStartedAtWallClockRef.current = Date.now();
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ video:true, audio:false }); }
    catch(e) {
      setLastError('Camera access denied: ' + e.message);
      closeMonitorWindow();
      return;
    }
    streamRef.current = stream;
    videoRef.current.srcObject = stream;
    if (mainVideoRef.current && mainVideoRef.current !== videoRef.current) {
      mainVideoRef.current.srcObject = stream;
      mainVideoRef.current.play().catch(() => {});
    }
    await videoRef.current.play();
    setCameraReady(true);
    if (canvasRef.current) drawUtils.current = new DrawingUtils(canvasRef.current.getContext('2d'));
    if (!sessionRef.current) {
      setLastError('No authenticated session found. Please log in again.');
      stream.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      closeMonitorWindow();
      return;
    }
    await logEvent(sessionRef.current, activeCohort?.id || null, 'session_started', {
      vlearn_url: contentUrl,
    }, userId);
    runningRef.current = true; pausedRef.current = false;
    setRunning(true); setPaused(false);
    rafRef.current = scheduleFrameRef.current(processFrame);
  }, [modelReady, contentUrl, processFrame, userId, activeCohort, openMonitorWindow, closeMonitorWindow]);

  const handleOpenSmallWindow = useCallback(() => {
    openMonitorWindow();
  }, [openMonitorWindow]);

  const beginPauseAOI = useCallback(() => {
    const now = performance.now();
    pauseAOIRef.current = {
      pauseNumber: activeSegmentRef.current,
      startedAt: now,
      startedWallClock: new Date().toISOString(),
      lastSampleAt: now,
      insideMs: 0,
      validFrames: 0,
      totalFrames: 0,
      finalized: false,
    };
    setCurrentPauseAOIPercent(0);
  }, []);

  const finalizePauseAOI = useCallback((endedBy) => {
    const tracker = pauseAOIRef.current;
    if (tracker.startedAt === null || tracker.finalized) return null;
    tracker.finalized = true;
    const totalPauseMs = Math.max(0, performance.now() - tracker.startedAt);
    const insideAOIMs = Math.min(totalPauseMs, Math.max(0, tracker.insideMs));
    const aoiPercent = totalPauseMs > 0 ? insideAOIMs / totalPauseMs * 100 : 0;
    setCurrentPauseAOIPercent(aoiPercent);

    const metric = {
      session_id: sessionRef.current,
      participant_id: userIdRef.current || null,
      cohort_id: activeCohortRef.current?.id || null,
      pause_and_reflect_number: tracker.pauseNumber,
      paused_at: tracker.startedWallClock,
      resumed_at: new Date().toISOString(),
      total_pause_ms: Math.round(totalPauseMs),
      inside_aoi_ms: Math.round(insideAOIMs),
      aoi_percent: Number(aoiPercent.toFixed(2)),
      valid_gaze_frames: tracker.validFrames,
      total_gaze_frames: tracker.totalFrames,
      ended_by: endedBy,
    };
    savePauseReflectionAOIWithRetry(metric).catch(error => {
      setLastError('[Supabase] save Pause and Reflect AOI: ' + String(error));
    });
    return metric;
  }, []);

  const handlePauseToggle = useCallback(() => {
    if (!running && !paused) return;
    if (paused) {
      finalizePauseAOI('resumed');
      activeSegmentRef.current += 1;
      activeRunStartedAtRef.current = performance.now();
      segmentStartedAtWallClockRef.current = Date.now();
      gazeBuffer.current = [];
      currentGazeAOIRef.current = null;
      gazeStatsRef.current = {
        insideRatio: null,
        outsideRatio: null,
        trackingCoverage: 0,
        validFrames: 0,
        totalFrames: 0,
        status: 'WAITING',
      };
      setBufferLen(0);
      pausedRef.current = false; setPaused(false);
      runningRef.current = true;
      setProcessingStatus('Running emotion prediction');
      processingGenerationRef.current += 1;
      if (rafRef.current) cancelFrameRef.current(rafRef.current);
      rafRef.current = scheduleFrameRef.current(processFrame);
      logEvent(sessionRef.current, activeCohort?.id || null, 'session_resumed', {}, userId).catch(() => {});
      return;
    }
    activeElapsedBeforePauseRef.current += Math.max(
      0,
      performance.now() - (activeRunStartedAtRef.current ?? performance.now())
    );
    activeRunStartedAtRef.current = null;
    // Discard a partial emotion window so it cannot span across reflection time.
    frameBuffer.current = [];
    gazeBuffer.current = [];
    setBufferLen(0);
    currentGazeAOIRef.current = null;
    gazeStatsRef.current = {
      insideRatio: null,
      outsideRatio: null,
      trackingCoverage: 0,
      validFrames: 0,
      totalFrames: 0,
      status: 'WAITING',
    };
    beginPauseAOI();
    pausedRef.current = true; setPaused(true);
    runningRef.current = false;
    setProcessingStatus('Starting dashboard gaze check');
    processingGenerationRef.current += 1;
    if (rafRef.current) cancelFrameRef.current(rafRef.current);
    rafRef.current = scheduleFrameRef.current(processFrame);
    logEvent(sessionRef.current, activeCohort?.id || null, 'session_paused', {}, userId).catch(() => {});
  }, [running, paused, processFrame, activeCohort, beginPauseAOI, finalizePauseAOI]);

  const handleStop = useCallback(() => {
    if (pausedRef.current) finalizePauseAOI('stopped');
    if (sessionRef.current) {
      logEvent(sessionRef.current, activeCohort?.id || null, 'session_stopped', {}, userId).catch(() => {});
    }
    runningRef.current = false; pausedRef.current = false;
    processingGenerationRef.current += 1;
    activeElapsedBeforePauseRef.current = 0;
    activeRunStartedAtRef.current = null;
    activeSegmentRef.current = 1;
    segmentStartedAtWallClockRef.current = Date.now();
    setRunning(false); setPaused(false);
    if (rafRef.current) cancelFrameRef.current(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    if (mainVideoRef.current) mainVideoRef.current.srcObject = null;
    setCameraReady(false); setNoFace(false);
    setProcessingStatus('Stopped');
    currentGazeAOIRef.current = null;
    gazeStatsRef.current = {
      insideRatio: null,
      outsideRatio: null,
      trackingCoverage: 0,
      validFrames: 0,
      totalFrames: 0,
      status: 'WAITING',
    };
    gazeBuffer.current = [];
    pauseAOIRef.current = emptyPauseAOITracker();
    setCurrentPauseAOIPercent(null);
    closeMonitorWindow();
  }, [activeCohort, closeMonitorWindow, finalizePauseAOI]);

  pauseActionRef.current = handlePauseToggle;
  stopActionRef.current = handleStop;
  monitorClosedActionRef.current = () => {
    monitorWindowRef.current = null;
    monitorElementsRef.current = null;
    setMonitorMode(null);
    restoreMainProcessingSurface();
    if (runningRef.current && !pausedRef.current) pauseActionRef.current();
    setMonitorNotice('Monitoring paused because the floating monitor was closed. Reopen it, then select Resume.');
    if (sessionRef.current) {
      logEvent(sessionRef.current, activeCohortRef.current?.id || null, 'monitor_window_closed', {}, userIdRef.current).catch(() => {});
    }
  };

  useEffect(() => {
    const elements = monitorElementsRef.current;
    if (!elements) return;
    const monitoring = running && !paused && cameraReady;
    elements.dot.classList.toggle('live', monitoring);
    elements.dot.classList.toggle('paused', paused);
  }, [running, paused, cameraReady, frameCount, monitorMode, processingStatus]);

  const handleLeaveToGateway = useCallback(() => {
    handleStop();
    setAuthReady(false);
    setIsAdminMode(false);
    setSelectedAdminCohortId(null);
    setUserId('');
    setCohortAccessCodeInput('');
    setAuthError('');
    setSessionId(null);
    sessionRef.current = null;
    setActiveCohort(null);
  }, [handleStop]);

  const handleRefreshAdminData = useCallback(async () => {
    await Promise.all([refreshAdminOverview(), refreshSessionGroups()]);
  }, [refreshAdminOverview, refreshSessionGroups]);

  useEffect(() => () => handleStop(), []);

  const injectDummy = useCallback(() => {
    const dummy = {};
    for (const lbl of LABEL_COLS) {
      const label = Math.floor(Math.random() * LEVEL_COUNT);
      const high = Math.random();
      dummy[lbl] = {
        label,
        level: LEVEL_LABELS[label],
        probabilities: { 0:+(1 - high).toFixed(4), 1:+high.toFixed(4) },
      };
    }
    setHistory(h => {
      const lastActiveElapsedMs = h.length
        ? (h[h.length - 1].active_elapsed_ms ?? h.length * WINDOW_SECONDS * 1000)
        : 0;
      const row = {
        predicted_at: new Date().toISOString(),
        active_elapsed_ms: lastActiveElapsedMs + WINDOW_SECONDS * 1000,
        segment_id: activeSegmentRef.current,
        pause_and_reflect_number: activeSegmentRef.current,
        segment_start_ts: segmentStartedAtWallClockRef.current,
      };
      for (const lbl of LABEL_COLS) row[lbl] = dummy[lbl].label;
      return [...h, row];
    });
    setCurrentPreds(dummy);
    setWindowCount(c => c + 1);
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    (async () => {
      try {
        const [rows, savedEditCount] = await Promise.all([
          getManualOverrides(sessionId, activeCohort?.id || null),
          getManualOverrideCount(sessionId, activeCohort?.id || null, userId),
        ]);
        if (cancelled || !Array.isArray(rows)) return;
        const next = {};
        for (const row of rows) {
          const key = `${row.label_col}:${row.bucket_label}`;
          if (next[key]) continue;
          if (Array.isArray(row.level_percentages) && row.level_percentages.length === LEVEL_COUNT) {
            next[key] = row.level_percentages.map(v => Number(v) || 0);
          }
        }
        setOverrides(next);
        setManualEditCount(savedEditCount);
      } catch (e) {
        if (!cancelled) setLastError('[supabase] load overrides: ' + String(e));
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId, activeCohort, userId]);

  const e = React.createElement;

  if (isAdminMode) {
    const buildPauseReflectBuckets = (maxIndex) => {
      const safeMax = Math.max(0, Math.floor(Number(maxIndex) || 0));
      const buckets = [];
      for (let i = 1; i <= safeMax; i++) buckets.push(String(i));
      return buckets;
    };

    const stickyCol = (leftPx, widthPx, header = false) => ({
      left: `${leftPx}px`,
      minWidth: `${widthPx}px`,
      width: `${widthPx}px`,
      maxWidth: `${widthPx}px`,
      zIndex: header ? 10 : 5,
    });

    const getBucketMetric = (metrics, bucketKey) => {
      const row = metrics?.[bucketKey];
      if (!row) return { minor: 0, major: 0, words: 0 };
      return {
        minor: Number(row.minor) || 0,
        major: Number(row.major) || 0,
        words: Number(row.words) || 0,
      };
    };

    const getAOIMetric = (metrics, bucketKey) => {
      const row = metrics?.[bucketKey];
      if (!row) return null;
      return {
        percent: Math.max(0, Math.min(100, Number(row.percent) || 0)),
        total_pause_ms: Math.max(0, Number(row.total_pause_ms) || 0),
        inside_aoi_ms: Math.max(0, Number(row.inside_aoi_ms) || 0),
      };
    };

    const renderReflectionMetricHeader = (prefix, bucket, metricKey, metricLabel) => (
      e('th', { key:`${prefix}-${bucket}-${metricKey}` }, `${metricLabel} (Reflection ${bucket})`)
    );

    const cohortsById = {};
    const activityTypeByGroupId = {};
    for (const g of cohortIds) {
      activityTypeByGroupId[g.id] = g.activity_type || g.type;
      cohortsById[g.id] = {
        id: g.id,
        activity_type: g.activity_type || g.type,
        type: g.activity_type || g.type,
        description: g.description || '',
        sessions: 0,
        edits: 0,
        max_pause_and_reflect_number: 0,
        par_metrics: {},
        aoi_inside_ms_sum: 0,
        aoi_total_ms_sum: 0,
        created_at: g.created_at,
        access_code: g.access_code,
      };
    }
    for (const row of adminSessions) {
      if (!cohortsById[row.cohort_id]) continue;
      const g = cohortsById[row.cohort_id];
      g.sessions += 1;
      g.edits += row.edit_count || 0;
      g.max_pause_and_reflect_number = Math.max(
        g.max_pause_and_reflect_number,
        Number(row.max_pause_and_reflect_number) || 0,
      );

      const rowMetrics = row.par_metrics || {};
      for (const key of Object.keys(rowMetrics)) {
        if (!g.par_metrics[key]) g.par_metrics[key] = { minor: 0, major: 0, words: 0 };
        g.par_metrics[key].minor += Number(rowMetrics[key]?.minor) || 0;
        g.par_metrics[key].major += Number(rowMetrics[key]?.major) || 0;
        g.par_metrics[key].words += Number(rowMetrics[key]?.words) || 0;
      }
      for (const metric of Object.values(row.aoi_metrics || {})) {
        g.aoi_inside_ms_sum += Math.max(0, Number(metric?.inside_aoi_ms) || 0);
        g.aoi_total_ms_sum += Math.max(0, Number(metric?.total_pause_ms) || 0);
      }
    }
    const cohortStats = Object.values(cohortsById).sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    const selectedCohortExists = selectedAdminCohortId && cohortStats.some(g => g.id === selectedAdminCohortId);
    const selectedCohortId = selectedCohortExists ? selectedAdminCohortId : null;
    const sessionsToShow = selectedCohortId
      ? adminSessions.filter(row => row.cohort_id === selectedCohortId)
      : [];

    const cohortMaxPauseReflect = cohortStats.reduce((mx, g) => Math.max(mx, Number(g.max_pause_and_reflect_number) || 0), 0);
    const selectedGroupMaxPauseReflect = selectedCohortId
      ? Math.max(
        Number(cohortStats.find(g => g.id === selectedCohortId)?.max_pause_and_reflect_number) || 0,
        sessionsToShow.reduce((mx, s) => Math.max(mx, Number(s.max_pause_and_reflect_number) || 0), 0),
      )
      : 0;
    const cohortBuckets = buildPauseReflectBuckets(cohortMaxPauseReflect);
    const sessionBuckets = buildPauseReflectBuckets(selectedGroupMaxPauseReflect);
    const getActivityLabelForRow = row => (
      ACTIVITY_TYPE_LABELS[row.activity_type || activityTypeByGroupId[row.cohort_id]]
      || row.activity_type
      || activityTypeByGroupId[row.cohort_id]
      || '-'
    );
    const getUserTextForRow = row => (row.justifications || [])
      .map(item => {
        const parts = [
          `Reflection ${item.reflection || '-'}`,
          item.label_col || '-',
          `${item.words_edited || 0} words`,
          item.overridden_at ? formatSingaporeTime(item.overridden_at, { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' }) : '',
        ].filter(Boolean);
        return `${parts.join(' | ')}: ${item.text || ''}`;
      })
      .join('\n\n');
    const handleDownloadParticipantCsv = () => {
      if (!selectedCohortId || !sessionsToShow.length) return;
      const headers = [
        'Participant ID',
        'Cohort ID',
        'Activity Type',
        'Total Edits Made',
        'User Text',
        ...sessionBuckets.flatMap(bucket => [
          `Minor Edits Made (Reflection ${bucket})`,
          `Major Edits Made (Reflection ${bucket})`,
          `Word Count (Reflection ${bucket})`,
          `% AOI (Reflection ${bucket})`,
        ]),
        'Submission Date',
      ];
      const rows = sessionsToShow.map(row => [
        row.participant_id || '-',
        row.cohort_id || '-',
        getActivityLabelForRow(row),
        row.edit_count || 0,
        getUserTextForRow(row),
        ...sessionBuckets.flatMap(bucket => {
          const m = getBucketMetric(row.par_metrics, bucket);
          const aoi = getAOIMetric(row.aoi_metrics, bucket);
          return [m.minor, m.major, m.words, aoi ? Number(aoi.percent.toFixed(2)) : ''];
        }),
        row.created_at ? formatSingaporeTime(row.created_at, { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' }) : '-',
      ]);
      downloadCsv(
        `participant-reflection-data-${safeFilenamePart(selectedCohortId)}-${new Date().toISOString().slice(0, 10)}.csv`,
        [headers, ...rows],
      );
    };

    return e('div', { className:'gateway-wrap' },
      e('div', { className:'gateway-admin-shell' },
        e('div', { className:'gateway-grid' },
        e('div', { className:'gateway-card admin-controls-card' },
          e('div', { className:'gateway-title' }, 'Cohort Setup'),
          e('div', { className:'gateway-sub' }, 'Create cohorts and track participant reflection activity with reflection edit counts.'),
          e('label', { className:'field-label' }, 'Cohort ID'),
          e('input', {
            className:'url-input',
            type:'text',
            placeholder:'e.g., Date_BATCH_session-type',
            value:newCohortId,
            onChange:ev=>setNewCohortId(ev.target.value),
          }),
          e('label', { className:'field-label' }, 'Activity Type'),
          e('select', {
            className:'url-input',
            value:newActivityType,
            onChange:ev=>setNewActivityType(ev.target.value),
          },
            Object.values(ACTIVITY_TYPES).map(type =>
              e('option', { key:type, value:type }, ACTIVITY_TYPE_LABELS[type])
            )
          ),
          e('label', { className:'field-label' }, 'Cohort Note (optional)'),
          e('input', {
            className:'url-input',
            type:'text',
            placeholder:'e.g., Batch 3 morning session',
            value:newTaskDescription,
            onChange:ev=>setNewTaskDescription(ev.target.value),
          }),
          e('div', { className:'name-row' },
            e('button', { className:'btn-primary', style:{flex:'1 1 100%'}, onClick:handleCreateCohort, disabled:!(newCohortId || '').trim() }, 'Create Cohort + Auto Access Code'),
            e('button', { className:'btn-ghost', style:{flex:'1 1 48%'}, onClick:handleRefreshAdminData }, 'Refresh Data'),
            e('button', { className:'btn-ghost', style:{flex:'1 1 48%'}, onClick:handleLeaveToGateway }, 'Back To Login'),
          ),
          adminInfoMsg ? e('div', { className:'banner warn', style:{marginTop:'0.5rem', marginBottom:0} }, adminInfoMsg) : null,
          adminError ? e('div', { className:'banner error', style:{marginTop:'0.5rem', marginBottom:0} }, adminError) : null,
        ),

        e('div', { className:'gateway-card admin-groups-card' },
          e('div', { className:'gateway-title' }, 'Reflection Cohort Dashboard'),
          e('div', { className:'summary-row' },
            e('div', { className:'summary-box' }, e('div', { className:'k' }, 'Cohorts'), e('div', { className:'v' }, String(cohortIds.length))),
            e('div', { className:'summary-box' }, e('div', { className:'k' }, 'Participant Submissions'), e('div', { className:'v' }, String(adminSessions.length))),
            e('div', { className:'summary-box' }, e('div', { className:'k' }, 'Total Edits Made'), e('div', { className:'v' }, String(adminSessions.reduce((s, r) => s + (r.edit_count || 0), 0)))),
            e('div', { className:'summary-box' }, e('div', { className:'k' }, 'Sync Status'), e('div', { className:'v' }, adminLoading ? 'Loading...' : 'Ready')),
          ),

          e('div', { className:'field-label', style:{marginTop:0} }, 'Reflection Cohort Data'),
          e('div', { className:'table-wrap wide admin-group-table-wrap' },
            e('table', { className:'admin-table wide' },
              e('thead', null,
                e('tr', null,
                  e('th', { className:'sticky-col', style:stickyCol(0, 190, true) }, 'Cohort ID'),
                  e('th', { className:'sticky-col', style:stickyCol(190, 120, true) }, 'Activity Type'),
                  e('th', null, 'Task Description'),
                  e('th', null, 'Access Code'),
                  e('th', null, 'Created On'),
                  e('th', { style:{ minWidth:'110px', width:'110px', maxWidth:'110px' } }, 'Total Edits Made'),
                  e('th', null, 'Average % AOI'),
                  e('th', null, 'Total Participants'),
                )
              ),
              e('tbody', null,
                cohortStats.length
                  ? cohortStats.map(g => e('tr', { key:g.id },
                    e('td', { className:'sticky-col', style:stickyCol(0, 190) },
                      e('button', {
                        type:'button',
                        className:`group-id-btn ${selectedCohortId === g.id ? 'active' : ''}`,
                        onClick:() => setSelectedAdminCohortId(prev => prev === g.id ? null : g.id),
                        title:selectedCohortId === g.id ? 'Click to clear filter' : 'Click to show only this group in Participant Reflection Data table',
                      }, g.id)
                    ),
                    e('td', { className:'sticky-col', style:stickyCol(190, 120) }, ACTIVITY_TYPE_LABELS[g.activity_type] || g.activity_type),
                    e('td', null, g.description || '-'),
                    e('td', null, e('span', { style:{fontFamily:'var(--mono)'} }, g.access_code || '-')),
                    e('td', null, g.created_at ? formatSingaporeTime(g.created_at, { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' }) : '-'),
                    e('td', { style:{ minWidth:'110px', width:'110px', maxWidth:'110px' } }, String(g.edits || 0)),
                    e('td', null, g.aoi_total_ms_sum ? `${(g.aoi_inside_ms_sum / g.aoi_total_ms_sum * 100).toFixed(1)}%` : '-'),
                    e('td', null, String(g.sessions || 0)),
                  ))
                  : e('tr', null, e('td', { colSpan:8, style:{color:'var(--muted)'} }, 'No cohorts created yet.'))
              )
            )
          )
        )
      ),
      e('div', { className:'gateway-card dashboard-card' },
        e('div', { className:'admin-session-section' },
          e('div', { className:'field-label admin-sticky-heading admin-session-heading' },
            e('span', null, selectedCohortId ? `Participant Reflection Data - ${selectedCohortId}` : 'Participant Reflection Data (Select a Cohort ID Above)'),
            e('button', {
              type:'button',
              className:'download-csv-btn',
              onClick:handleDownloadParticipantCsv,
              disabled:!selectedCohortId || !sessionsToShow.length,
              title:selectedCohortId ? 'Download the visible participant table as CSV' : 'Select a cohort first',
            }, 'Download CSV')
          ),
          e('div', { className:'table-wrap wide admin-session-table-wrap' },
            e('table', { className:'admin-table wide admin-session-table' },
              e('thead', null,
                e('tr', null,
                  e('th', { className:'sticky-col', style:stickyCol(0, 170, true) }, 'Participant ID'),
                  e('th', { className:'sticky-col', style:stickyCol(170, 170, true) }, 'Cohort ID'),
                  e('th', { className:'sticky-col', style:stickyCol(340, 120, true) }, 'Activity Type'),
                  e('th', { style:{ minWidth:'120px', width:'120px', maxWidth:'120px' } }, 'Total Edits Made'),
                  e('th', { style:{ minWidth:'115px', width:'115px', maxWidth:'115px' } }, 'User Text'),
                  ...sessionBuckets.flatMap(bucket => [
                    renderReflectionMetricHeader('s', bucket, 'minor', 'Minor Edits Made'),
                    renderReflectionMetricHeader('s', bucket, 'major', 'Major Edits Made'),
                    renderReflectionMetricHeader('s', bucket, 'words', 'Word Count'),
                    renderReflectionMetricHeader('s', bucket, 'aoi', '% AOI'),
                  ]),
                  e('th', null, 'Submission Date'),
                )
              ),
              e('tbody', null,
                sessionsToShow.length
                  ? sessionsToShow.map(row => e('tr', { key:row.id },
                    e('td', { className:'sticky-col', style:stickyCol(0, 170) }, row.participant_id || '-'),
                    e('td', { className:'sticky-col', style:stickyCol(170, 170) }, row.cohort_id || '-'),
                    e('td', { className:'sticky-col', style:stickyCol(340, 120) }, getActivityLabelForRow(row)),
                    e('td', { style:{ minWidth:'120px', width:'120px', maxWidth:'120px' } }, String(row.edit_count || 0)),
                    e('td', { style:{ minWidth:'115px', width:'115px', maxWidth:'115px' } },
                      row.justifications?.length
                        ? e('button', {
                            type:'button',
                            className:'text-view-btn',
                            onClick:() => setAdminTextModal(row),
                            title:'View active text entered by this participant',
                          }, `View (${row.justifications.length})`)
                        : e('span', { style:{ color:'var(--muted)' } }, '-')
                    ),
                    ...sessionBuckets.flatMap(bucket => {
                      const m = getBucketMetric(row.par_metrics, bucket);
                      const aoi = getAOIMetric(row.aoi_metrics, bucket);
                      return [
                        e('td', { key:`s-${row.id}-${bucket}-minor` }, String(m.minor)),
                        e('td', { key:`s-${row.id}-${bucket}-major` }, String(m.major)),
                        e('td', { key:`s-${row.id}-${bucket}-words` }, String(m.words)),
                        e('td', {
                          key:`s-${row.id}-${bucket}-aoi`,
                          title:aoi ? `${Math.round(aoi.inside_aoi_ms / 1000)}s inside AOI / ${Math.round(aoi.total_pause_ms / 1000)}s total pause time` : 'No completed AOI measurement',
                        }, aoi ? `${aoi.percent.toFixed(1)}%` : '-'),
                      ];
                    }),
                    e('td', null, row.created_at ? formatSingaporeTime(row.created_at, { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' }) : '-'),
                  ))
                  : e('tr', null, e('td', { colSpan:6 + (sessionBuckets.length * 4), style:{color:'var(--muted)'} }, selectedCohortId ? (adminLoading ? 'Loading participant data...' : 'No submissions found for this cohort.') : 'Select a cohort to view participant reflections.'))
              )
            )
          )
        )
      ),
      adminTextModal ? e('div', { className:'admin-text-modal', onClick:() => setAdminTextModal(null) },
        e('div', { className:'admin-text-card', onClick:ev=>ev.stopPropagation() },
          e('div', { className:'admin-text-head' },
            e('div', null,
              e('div', { className:'admin-text-title' }, 'Participant Text'),
              e('div', { className:'admin-text-sub' }, `${adminTextModal.participant_id || '-'} | ${adminTextModal.cohort_id || '-'} | ${adminTextModal.justifications?.length || 0} active text entr${adminTextModal.justifications?.length === 1 ? 'y' : 'ies'}`)
            ),
            e('button', { className:'btn-ghost admin-text-close', onClick:() => setAdminTextModal(null) }, 'Close')
          ),
          e('div', { className:'admin-text-list' },
            (adminTextModal.justifications || []).length
              ? adminTextModal.justifications.map((item, idx) =>
                  e('div', { className:'admin-text-item', key:`${item.reflection}-${item.bucket_label}-${item.label_col}-${idx}` },
                    e('div', { className:'admin-text-meta' },
                      e('span', null, `Reflection ${item.reflection}`),
                      e('span', null, item.label_col || '-'),
                      e('span', null, `${item.words_edited || 0} words`),
                      item.overridden_at ? e('span', null, formatSingaporeTime(item.overridden_at, { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' })) : null
                    ),
                    e('div', { className:'admin-text-body' }, item.text)
                  )
                )
              : e('div', { className:'admin-text-empty' }, 'No active text entries for this participant.')
          )
        )
      ) : null
    )
  );
  }

  if (!authReady) {
    return e('div', { className:'gateway-wrap login-gateway-wrap' },
      e('div', { className:'name-card', style:{ width:'min(520px, 92%)' } },
        e('div', { className:'gateway-title' }, 'Participant Login'),
        e('div', { className:'gateway-sub' }, 'Enter your Participant ID and Access Code to login.'),
        e('label', { className:'field-label' }, 'Participant ID'),
        e('input', {
          className:'url-input',
          type:'text',
          inputMode:'numeric',
          pattern:'[0-9]*',
          placeholder:'Enter numeric Participant ID',
          value:userId,
          onChange:ev=>setUserId((ev.target.value || '').replace(/\D+/g, '')),
        }),
        e('label', { className:'field-label' }, 'Access Code'),
        e('input', {
          className:'url-input',
          type:'password',
          placeholder:'Enter your access code',
          value:cohortAccessCodeInput,
          onChange:ev=>setCohortAccessCodeInput(ev.target.value),
        }),
        authError ? e('div', { className:'banner error', style:{marginTop:'0.6rem', marginBottom:0} }, authError) : null,
        e('div', { className:'name-row' },
          e('div', { style:{ marginLeft:'auto' } },
            e('button', {
              className:'btn-primary',
              onClick:handleEnterWithAccessCode,
              disabled:!userId.trim() || !cohortAccessCodeInput.trim(),
            }, 'Login')
          )
        )
      )
    );
  }

  return e('div', { className:'app' },
    e('div', { className:'header' },
      e('div', { className:`status-dot ${running ? 'active' : ''}` }),
      e('h1', null, 'Open ', e('span',null,'Emotional'), ' Learner'),
      userId ? e('div', { style:{ marginLeft:'8px', color:'var(--subtext)', fontFamily:'var(--mono)'} }, `Participant ${userId}`) : null,
      activeCohort?.id ? e('span', { className:'pill', style:{ marginLeft:'0.4rem' } }, `${activeCohort.id} - ${ACTIVITY_TYPE_LABELS[activeCohort.activity_type || activeCohort.type] || activeCohort.activity_type || activeCohort.type}`) : null,
      e('div', { className:'header-actions' },
        e('button', {
          className:`btn-ghost ${monitorMode ? 'lit' : ''}`,
          onClick:handleOpenSmallWindow,
          title:'Open the compact, always-visible OELM floating view',
        }, monitorMode ? 'Floating View Active' : 'Open Floating View'),
        e('button', { className:'btn-ghost', onClick:handleLeaveToGateway }, 'Exit Session'),
      ),
    ),

    configWarn && e('div', { className:'banner warn' }, 'Warning: ' + configWarn),
    lastError  && e('div', { className:'banner error' }, 'Warning: ' + lastError),
    monitorNotice && e('div', { className:'banner warn' }, monitorNotice),

    e('div', { className:'control-bar' },
      e('input', {
        className:'url-input', type:'text',
        placeholder:'Paste a lecture / video URL...',
        value:contentUrl, onChange:ev=>setContentUrl(ev.target.value),
        onBlur: ev => { if (ev.target.value) logEvent(sessionRef.current, activeCohort?.id || null, 'vlearn_link_added', { vlearn_url: ev.target.value }, userId).catch(() => {}); },
        disabled:running || paused,
      }),
      e('div', { className:'controls-row' },
        e('div', null,
          !running && !paused
            ? e('button', { className:'btn-primary session-action-btn', onClick:handleStart, disabled:!modelReady,
                title: modelReady ? 'Start capture' : 'Loading MediaPipe model...' },
                modelReady ? 'Start' : 'Loading...')
            : e('button', { className:'btn-danger session-action-btn', onClick:handleStop }, 'Stop')
        ),
        e('div', { className:'btn-group' },
          e('button', {
            className:`btn-ghost reflect-action-btn ${paused ? 'lit' : ''}`,
            onClick:handlePauseToggle, disabled:!running && !paused,
          }, paused ? 'Resume' : 'Pause and Reflect')
        ),
      ),
    ),

    e('div', {
      className:`main-grid ${paused ? 'reflection-layout' : 'content-layout'}`,
      style:{ gridTemplateColumns:'minmax(0, 1fr)', width:'100%', minWidth:0 },
    },

      !paused ? e('div', { style:{display:'flex',flexDirection:'column',gap:'1rem'} },
        e('div', { className:'panel' },
          e('div', { className:'panel-header' },
            e('span',null,'Content'),
            contentUrl && e('span',{style:{color:'var(--muted)',fontSize:'0.63rem',maxWidth:220,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}},contentUrl),
          ),
          contentUrl && (running || paused)
            ? e('div',{className:'iframe-wrap'}, e('iframe',{src:contentUrl,allow:'autoplay; encrypted-media',allowFullScreen:true,sandbox:'allow-scripts allow-same-origin allow-forms allow-popups'}))
            : e('div',{className:'iframe-placeholder'},
                e('div',{className:'icon'},'Content'),
                e('p',null, contentUrl ? 'Press Start to load content' : 'Paste a URL above, then press Start to load content')
              ),
        ),

        SHOW_WEBCAM_PANEL ? e('div', { className:'panel' },
          e('div', { className:'panel-header' },
            e('span',null,'Face capture'),
            running && e('span',{className:'live-pill'},'LIVE - blendshape dicts only, no video leaves this device'),
          ),
          e('div', { className:'webcam-section' },
            e('div', { className:'webcam-wrap' },
              e('video', {
                ref:node => {
                  mainVideoRef.current = node;
                  if (!monitorWindowRef.current) videoRef.current = node;
                },
                muted:true,
                playsInline:true,
                style:{display:cameraReady?'block':'none'},
              }),
              e('canvas', {
                ref:node => {
                  mainCanvasRef.current = node;
                  if (!monitorWindowRef.current) canvasRef.current = node;
                },
              }),
              !cameraReady && e('div',{className:'webcam-off'}, running ? 'Starting camera...' : 'Camera off'),
              cameraReady && e('div',{className:'cam-badge'}, `${frameCount} frames`),
              noFace && e('div',{className:'no-face-badge'},'No face detected'),
            ),
            e('div', { className:'progress-wrap' },
              e('div',{className:'progress-bg'}, e('div',{className:'progress-fill',style:{width:`${Math.min(bufferLen/WINDOW_FRAMES,1)*100}%`}})),
              e('div',{className:'progress-lbl'}, `Window buffer: ${bufferLen} / ${WINDOW_FRAMES} frames - predicts every ${WINDOW_SECONDS}s`),
            ),
          ),
          e('div', { className:'stats-row' },
            e('div',{className:'stat'}, e('div',{className:'val'},frameCount),    e('div',{className:'key'},'Frames')),
            e('div',{className:'stat'}, e('div',{className:'val'},windowCount),   e('div',{className:'key'},'Windows')),
            e('div',{className:'stat'}, e('div',{className:'val'},history.length),e('div',{className:'key'},'Rows')),
            e('div',{className:'stat'}, e('div',{className:'val'},lastPredTime??'-'),e('div',{className:'key'},'Last pred')),
          ),
        ) : null,
      ) : null,

      e('div', { className:'secondary-column', style:{display:'flex',flexDirection:'column',gap:'1rem',width:'100%',minWidth:0} },

        paused ? e('div', { className:'panel' },
          e('div',{className:'panel-header'},
            panelTitle('Overall Emotional Intensity Distribution', 'chart-title-icon chart-title-icon-wide'),
            e('div', { className:'edit-counter', title:'Time looking inside the screen AOI divided by total current Pause and Reflect time' },
              e('span', null, `Pause & Reflect #${activeSegmentRef.current} AOI`),
              e('span', { className:'count' }, currentPauseAOIPercent === null ? '-' : `${currentPauseAOIPercent.toFixed(1)}%`),
            ),
          ),
          e('div',{className:'timeline-section'}, e(OverallSplitChart,{history})),
        ) : null,

        paused ? e('div', { className:'panel' },
          e('div',{className:'panel-header'},
            panelTitle(`Pause & Reflect #${activeSegmentRef.current} - Emotion Intensity Distribution Across Time Segments (5-Min Intervals)`, 'chart-title-icon chart-title-icon-wide'),
            e('div', { className:'edit-counter', title:'Number of manual edits saved in this session' },
              e('span', null, 'Edits made'),
              e('span', { className:'count' }, String(manualEditCount)),
            ),
          ),
          // -- LabelSplitCharts now receives sessionId --
          e(LabelSplitCharts, {
            history,
            sessionId,
            cohortId: activeCohort?.id || null,
            userId,
            overrides,
            setOverrides,
            editableSegmentId: activeSegmentRef.current,
            pauseAndReflectNumber: activeSegmentRef.current,
            canEdit: editPolicy.canEdit,
            requireTextJustification: editPolicy.requireTextJustification,
            onManualEditSaved: () => setManualEditCount(count => count + 1),
          }),
          e('div', { className:'panel-header', style:{ marginTop:'0.25rem' } }, panelTitle('Emotion Intensity Timeline View', 'chart-title-icon chart-title-icon-wide')),
          e('div', { style:{ display:'flex', gap:'1rem', padding:'0.7rem 1rem', flexWrap:'wrap', alignItems:'center' } },
            LABEL_COLS.map(lbl => e('label', { key:lbl, style:{ display:'flex', alignItems:'center', gap:'0.4rem', cursor:'pointer', color:'#ccc', fontSize:'0.85rem' } },
              e('input', { type:'checkbox', checked:checkedLabels[lbl], onChange:ev=>setCheckedLabels(prev=>({...prev,[lbl]:ev.target.checked})), style:{cursor:'pointer'} }),
              lbl
            )),
            e('div', { style:{ fontSize:'0.75rem', color:'#9ba3c4', fontStyle:'italic', marginLeft:'auto' } }, 'OK Tick to View the Emotional Intensity Timeline')
          ),
          e('div',{className:'timeline-section'}, e(TimelineChart,{history, visibleLabels:checkedLabels})),
          e('hr',{className:'div'}),
        ) : null,

        SHOW_DEBUG_PANEL ? e('details', null,
          e('summary',null,'Debug panel'),
          e('div',{className:'debug-box'},
            e('div',null, e('b',null,'MediaPipe: '), e('span',{className:modelReady?'ok':'err'},String(modelReady))),
            e('div',null, e('b',null,'Camera: '),    e('span',{className:cameraReady?'ok':'err'},String(cameraReady))),
            e('div',null, e('b',null,'Running: '),   e('span',{className:running?'ok':'err'},String(running))),
            e('div',null, e('b',null,'Session ID: '),sessionId??'-'),
            e('div',null, e('b',null,'HF Space API: '), API_BASE_URL ? e('span',{className:'ok'},API_BASE_URL) : e('span',{className:'err'},'not set - edit HF_SPACE_URL')),
            e('div',null, e('b',null,'Supabase: '),   SUPABASE_URL?e('span',{className:'ok'},'configured'):e('span',{className:'err'},'not set')),
            e('div',null, e('b',null,'Buffer: '),     `${bufferLen} / ${WINDOW_FRAMES}`),
            e('div',null, e('b',null,'Eye tracking mode: '), paused ? e('span',{className:'ok'},'active (emotion prediction paused)') : 'inactive'),
            e('div',null, e('b',null,'Whole-screen AOI: '),
              gazeStatsRef.current.insideRatio === null
                ? '-'
                : `${Math.round(gazeStatsRef.current.insideRatio * 100)}% (${gazeStatsRef.current.validFrames}/${gazeStatsRef.current.totalFrames} usable frames)`),
            e('div',null, e('b',null,'Tracking coverage: '), `${Math.round(gazeStatsRef.current.trackingCoverage * 100)}%`),
            e('div',null, e('b',null,'Attention status: '), attentionStatusLabel(gazeStatsRef.current.status)),
            e('div',null, e('b',null,'Live gaze: '), currentGazeAOIRef.current
              ? `x ${currentGazeAOIRef.current.combined_gaze_x}, y ${currentGazeAOIRef.current.combined_gaze_y} - ${currentGazeAOIRef.current.inside_aoi ? 'inside screen' : 'looking away'}`
              : '-'),
            e('div',null, e('b',null,'History rows: '),history.length),
            e('hr',{className:'div'}),
            e('div',{className:'dbg-btns'},
              e('button',{className:'btn-ghost',onClick:injectDummy},'Inject dummy'),
              e('button',{className:'btn-ghost',onClick:()=>{setHistory([]);setCurrentPreds(null);}},'Clear history'),
            ),
          ),
        ) : null,
      ),
    ),
  );
}

createRoot(document.getElementById('root')).render(React.createElement(App));
