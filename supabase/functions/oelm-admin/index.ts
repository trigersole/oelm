import { createClient } from "npm:@supabase/supabase-js@2";

const encoder = new TextEncoder();
const ACTIVE_MODEL_VERSION = "xgb260-static-seed1729-v1";

function json(body: unknown, status = 200, origin = "") {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": origin,
      "access-control-allow-headers": "authorization, apikey, content-type",
      "access-control-allow-methods": "POST, OPTIONS",
      "vary": "Origin",
    },
  });
}

function allowedOrigin(request: Request) {
  const origin = request.headers.get("origin") || "";
  const allowed = (Deno.env.get("OELM_ALLOWED_ORIGINS") || "")
    .split(",")
    .map(value => value.trim())
    .filter(Boolean);
  return allowed.includes(origin) ? origin : "";
}

function getPublishableKey() {
  const configured = Deno.env.get("OELM_SUPABASE_PUBLISHABLE_KEY") || "";
  if (configured) return configured;
  const legacy = Deno.env.get("SUPABASE_ANON_KEY") || "";
  if (legacy) return legacy;
  try {
    const keys = JSON.parse(Deno.env.get("SUPABASE_PUBLISHABLE_KEYS") || "{}");
    return String(keys?.default || Object.values(keys || {})[0] || "");
  } catch {
    return "";
  }
}

function base64url(bytes: Uint8Array) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/g, "");
}

async function hmac(value: string, secret: string) {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  return base64url(new Uint8Array(await crypto.subtle.sign("HMAC", key, encoder.encode(value))));
}

async function safeEqual(left: string, right: string) {
  const leftHash = new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(left)));
  const rightHash = new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(right)));
  let difference = 0;
  for (let index = 0; index < leftHash.length; index++) difference |= leftHash[index] ^ rightHash[index];
  return difference === 0;
}

async function createAdminToken(secret: string) {
  const payload = base64url(encoder.encode(JSON.stringify({ role: "oelm-admin", exp: Date.now() + 60 * 60 * 1000 })));
  return `${payload}.${await hmac(payload, secret)}`;
}

async function readSignedToken(token: string, secret: string) {
  const [payload, signature] = token.split(".");
  if (!payload || !signature || !(await safeEqual(signature, await hmac(payload, secret)))) return null;
  try {
    const normalized = payload.replaceAll("-", "+").replaceAll("_", "/");
    const padded = normalized + "=".repeat((4 - (normalized.length % 4)) % 4);
    const decoded = JSON.parse(atob(padded));
    return Number(decoded?.exp) > Date.now() ? decoded : null;
  } catch {
    return null;
  }
}

async function verifyAdminToken(token: string, secret: string) {
  const decoded = await readSignedToken(token, secret);
  return decoded?.role === "oelm-admin";
}

async function createParticipantToken(
  secret: string,
  participantId: string,
  cohortId: string,
  sessionId: string,
) {
  const body = {
    role: "oelm-participant",
    participant_id: participantId,
    cohort_id: cohortId,
    session_id: sessionId,
    exp: Date.now() + 12 * 60 * 60 * 1000,
  };
  const payload = base64url(encoder.encode(JSON.stringify(body)));
  return `${payload}.${await hmac(payload, secret)}`;
}

function normalizeStoredLabel(value: unknown) {
  if (typeof value !== "string") return value || {};
  try {
    return JSON.parse(value);
  } catch {
    return {};
  }
}

Deno.serve(async request => {
  const origin = allowedOrigin(request);
  if (!origin) return json({ error: "Origin not allowed." }, 403, "null");
  if (request.method === "OPTIONS") return new Response(null, {
    status: 204,
    headers: {
      "access-control-allow-origin": origin,
      "access-control-allow-headers": "authorization, apikey, content-type",
      "access-control-allow-methods": "POST, OPTIONS",
      "vary": "Origin",
    },
  });
  if (request.method !== "POST") return json({ error: "Method not allowed." }, 405, origin);

  const input = await request.json().catch(() => ({}));
  const supabaseUrl = Deno.env.get("SUPABASE_URL") || "";
  if (input?.action === "public_config") {
    const publishableKey = getPublishableKey();
    const hfSpaceUrl = Deno.env.get("OELM_HF_SPACE_URL") || "";
    if (!supabaseUrl || !publishableKey || !hfSpaceUrl) {
      return json({ error: "Public application configuration is incomplete." }, 503, origin);
    }
    return json({
      SUPABASE_URL: supabaseUrl,
      SUPABASE_KEY: publishableKey,
      HF_SPACE_URL: hfSpaceUrl,
      API_BASE_URL: hfSpaceUrl,
      ADMIN_API_URL: `${supabaseUrl.replace(/\/$/, "")}/functions/v1/oelm-admin`,
    }, 200, origin);
  }

  const adminId = Deno.env.get("OELM_ADMIN_USER_ID") || "";
  const adminPassword = Deno.env.get("OELM_ADMIN_PASSWORD") || "";
  const sessionSecret = Deno.env.get("OELM_ADMIN_SESSION_SECRET") || "";
  const supabaseSecret = Deno.env.get("OELM_SUPABASE_SECRET_KEY") || Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";
  if (!sessionSecret || !supabaseUrl || !supabaseSecret) {
    return json({ error: "Application data service is not configured." }, 503, origin);
  }

  const db = createClient(supabaseUrl, supabaseSecret, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  if (input?.action === "participant_login") {
    const participantId = String(input?.participant_id || "").trim();
    const accessCode = String(input?.access_code || "").trim();
    if (!/^\d+$/.test(participantId) || !accessCode) {
      return json({ error: "Participant ID and access code are required." }, 400, origin);
    }

    const { data: cohorts, error: cohortError } = await db.from("login_credentials")
      .select("cohort_id,created_at,activity_type,task_description,active")
      .eq("access_code", accessCode)
      .or("active.eq.true,active.is.null")
      .limit(1);
    if (cohortError) return json({ error: cohortError.message }, 500, origin);
    const cohort = cohorts?.[0];
    if (!cohort?.cohort_id) return json({ error: "Invalid access code. Access denied." }, 401, origin);

    const cohortId = String(cohort.cohort_id);
    const { data: previousSessions, error: sessionLookupError } = await db.from("sessions")
      .select("id,session_id,label,created_at,cohort_id,activity_type,participant_id")
      .eq("cohort_id", cohortId)
      .eq("participant_id", participantId)
      .order("created_at", { ascending: false })
      .limit(1);
    if (sessionLookupError) return json({ error: sessionLookupError.message }, 500, origin);

    let session = previousSessions?.[0] || null;
    const resumed = Boolean(session);
    if (!session) {
      const sessionId = crypto.randomUUID();
      const { data: created, error: createError } = await db.from("sessions").insert({
        session_id: sessionId,
        label: participantId,
        participant_id: participantId,
        cohort_id: cohortId,
        activity_type: cohort.activity_type || "without_edit",
        created_at: new Date().toISOString(),
      }).select("id,session_id,label,created_at,cohort_id,activity_type,participant_id").single();
      if (createError) return json({ error: createError.message }, 500, origin);
      session = created;
    }

    const sessionId = String(session?.session_id || session?.id || "");
    const loadPredictions = async (columns: string) => {
      const rows: Record<string, unknown>[] = [];
      for (let offset = 0; offset < 5000; offset += 1000) {
        const page = await db.from("emotion_predictions")
          .select(columns)
          .eq("session_id", sessionId)
          .eq("model_version", ACTIVE_MODEL_VERSION)
          .order("predicted_at", { ascending: true })
          .range(offset, offset + 999);
        if (page.error) return { data: null, error: page.error };
        rows.push(...(page.data || []));
        if ((page.data || []).length < 1000) break;
      }
      return { data: rows, error: null };
    };

    let predictionQuery = await loadPredictions(
      "predicted_at,label,pause_and_reflect_number,active_elapsed_ms,segment_id,segment_start_ts,model_version",
    );
    // This fallback keeps existing deployments usable while the migration is
    // being applied. Exact elapsed-time restoration starts after the new
    // columns exist.
    if (predictionQuery.error && /active_elapsed_ms|segment_id|segment_start_ts/i.test(predictionQuery.error.message)) {
      predictionQuery = await loadPredictions("predicted_at,label,pause_and_reflect_number,model_version");
    }
    if (predictionQuery.error) return json({ error: predictionQuery.error.message }, 500, origin);

    const predictions = (predictionQuery.data || []).map((row, index) => ({
      predicted_at: row.predicted_at,
      labels: normalizeStoredLabel(row.label),
      pause_and_reflect_number: row.pause_and_reflect_number || row.segment_id || 1,
      segment_id: row.segment_id || row.pause_and_reflect_number || 1,
      active_elapsed_ms: row.active_elapsed_ms || ((index + 1) * 10_000),
      segment_start_ts: row.segment_start_ts ? new Date(String(row.segment_start_ts)).getTime() : null,
    }));

    return json({
      token: await createParticipantToken(sessionSecret, participantId, cohortId, sessionId),
      expires_in: 12 * 60 * 60,
      resumed,
      cohort: {
        cohort_id: cohortId,
        created_at: cohort.created_at,
        activity_type: cohort.activity_type,
        task_description: cohort.task_description,
      },
      session: { ...session, session_id: sessionId },
      predictions,
    }, 200, origin);
  }

  if (input?.action === "save_blendshape_batch") {
    const bearer = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
    const participant = await readSignedToken(bearer, sessionSecret);
    if (participant?.role !== "oelm-participant") {
      return json({ error: "Participant session is invalid or expired." }, 401, origin);
    }

    const values = input?.values;
    const expectedFeatureCount = 52;
    if (!Array.isArray(values) || values.length < 1 || values.length > 300) {
      return json({ error: "A batch must contain between 1 and 300 frames." }, 400, origin);
    }
    const validValues = values.every((frame: unknown) =>
      Array.isArray(frame) && frame.length === expectedFeatureCount &&
      frame.every(value => Number.isFinite(value) && Number(value) >= 0 && Number(value) <= 1)
    );
    if (!validValues) return json({ error: "Blendshape batch values are invalid." }, 400, origin);

    const startedAt = new Date(input?.started_at || "");
    const endedAt = new Date(input?.ended_at || "");
    if (!Number.isFinite(startedAt.getTime()) || !Number.isFinite(endedAt.getTime()) || endedAt < startedAt) {
      return json({ error: "Blendshape batch timestamps are invalid." }, 400, origin);
    }

    const batchId = String(input?.batch_id || "");
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(batchId)) {
      return json({ error: "Blendshape batch ID is invalid." }, 400, origin);
    }

    const { error } = await db.from("raw_blendshape_batches").upsert({
      batch_id: batchId,
      session_id: participant.session_id,
      participant_id: participant.participant_id,
      cohort_id: participant.cohort_id,
      started_at: startedAt.toISOString(),
      ended_at: endedAt.toISOString(),
      capture_fps: Number(input?.capture_fps) || 5,
      frame_count: values.length,
      feature_count: expectedFeatureCount,
      values,
    }, { onConflict: "batch_id", ignoreDuplicates: true });
    if (error) return json({ error: error.message }, 500, origin);
    return json({ saved: true, frame_count: values.length }, 200, origin);
  }

  if (input?.action === "login") {
    if (!adminId || !adminPassword) return json({ error: "Administrator service is not configured." }, 503, origin);
    const validId = await safeEqual(String(input?.admin_id || ""), adminId);
    const validPassword = await safeEqual(String(input?.password || ""), adminPassword);
    if (!validId || !validPassword) return json({ error: "Invalid administrator credentials." }, 401, origin);
    return json({ token: await createAdminToken(sessionSecret), expires_in: 3600 }, 200, origin);
  }

  const bearer = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  if (!(await verifyAdminToken(bearer, sessionSecret))) {
    return json({ error: "Administrator session is invalid or expired." }, 401, origin);
  }

  if (input?.action === "list_cohorts") {
    const { data, error } = await db.from("login_credentials")
      .select("cohort_id,access_code,created_at,activity_type,task_description,active")
      .order("created_at", { ascending: false })
      .limit(500);
    if (error) return json({ error: error.message }, 500, origin);
    return json({ cohorts: data || [] }, 200, origin);
  }

  if (input?.action === "create_cohort") {
    const cohort = input?.cohort || {};
    if (!String(cohort.id || "").trim() || !String(cohort.access_code || "").trim()) {
      return json({ error: "Cohort ID and access code are required." }, 400, origin);
    }
    const { data, error } = await db.from("login_credentials").insert({
      cohort_id: String(cohort.id).trim(),
      access_code: String(cohort.access_code).trim(),
      created_at: cohort.created_at || new Date().toISOString(),
      activity_type: cohort.activity_type || "without_edit",
      task_description: String(cohort.description || "").trim(),
      active: true,
    }).select();
    if (error) return json({ error: error.message }, 500, origin);
    return json({ cohort: data?.[0] || null }, 200, origin);
  }

  if (input?.action === "overview") {
    const [sessions, predictions, overrides, aoi] = await Promise.all([
      db.from("sessions").select("id,session_id,label,created_at,cohort_id,activity_type,participant_id").order("created_at", { ascending: false }).limit(500),
      db.from("emotion_predictions").select("session_id,participant_id,cohort_id,activity_type,pause_and_reflect_number,predicted_at").eq("model_version", ACTIVE_MODEL_VERSION).order("predicted_at", { ascending: false }).limit(20000),
      db.from("manual_overrides").select("session_id,cohort_id,participant_id,pause_and_reflect_number,bucket_label,label_col,is_minor_change,is_major_change,words_edited,justification,overridden_at,status").order("overridden_at", { ascending: false }).limit(10000),
      db.from("pause_reflection_aoi").select("session_id,cohort_id,participant_id,pause_and_reflect_number,total_pause_ms,inside_aoi_ms,aoi_percent,valid_gaze_frames,total_gaze_frames,paused_at,resumed_at").order("paused_at", { ascending: false }).limit(10000),
    ]);
    const failure = sessions.error || predictions.error || overrides.error;
    if (failure) return json({ error: failure.message }, 500, origin);
    return json({
      sessions: sessions.data || [],
      predictions: predictions.data || [],
      overrides: overrides.data || [],
      aoi: aoi.error ? [] : (aoi.data || []),
    }, 200, origin);
  }

  return json({ error: "Unknown administrator action." }, 400, origin);
});
