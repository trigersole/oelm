import { createClient } from "npm:@supabase/supabase-js@2";

const encoder = new TextEncoder();

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

async function verifyAdminToken(token: string, secret: string) {
  const [payload, signature] = token.split(".");
  if (!payload || !signature || !(await safeEqual(signature, await hmac(payload, secret)))) return false;
  try {
    const normalized = payload.replaceAll("-", "+").replaceAll("_", "/");
    const padded = normalized + "=".repeat((4 - (normalized.length % 4)) % 4);
    const decoded = JSON.parse(atob(padded));
    return decoded?.role === "oelm-admin" && Number(decoded?.exp) > Date.now();
  } catch {
    return false;
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
  if (!adminId || !adminPassword || !sessionSecret || !supabaseUrl || !supabaseSecret) {
    return json({ error: "Administrator service is not configured." }, 503, origin);
  }

  if (input?.action === "login") {
    const validId = await safeEqual(String(input?.admin_id || ""), adminId);
    const validPassword = await safeEqual(String(input?.password || ""), adminPassword);
    if (!validId || !validPassword) return json({ error: "Invalid administrator credentials." }, 401, origin);
    return json({ token: await createAdminToken(sessionSecret), expires_in: 3600 }, 200, origin);
  }

  const bearer = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  if (!(await verifyAdminToken(bearer, sessionSecret))) {
    return json({ error: "Administrator session is invalid or expired." }, 401, origin);
  }

  const db = createClient(supabaseUrl, supabaseSecret, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

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
      db.from("emotion_predictions").select("session_id,participant_id,cohort_id,activity_type,pause_and_reflect_number,predicted_at").order("predicted_at", { ascending: false }).limit(20000),
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
