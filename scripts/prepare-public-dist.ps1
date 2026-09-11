$ErrorActionPreference = 'Stop'

$required = @(
  'OELM_SUPABASE_URL',
  'OELM_SUPABASE_PUBLISHABLE_KEY',
  'OELM_HF_SPACE_URL'
)

foreach ($name in $required) {
  if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
    throw "Missing required environment variable: $name"
  }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot 'src'
$distRoot = Join-Path $projectRoot 'dist\src'
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'data') | Out-Null

Copy-Item -LiteralPath (Join-Path $projectRoot 'index.html') -Destination (Join-Path $projectRoot 'dist\index.html') -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot 'app.js') -Destination (Join-Path $distRoot 'app.js') -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot 'styles.css') -Destination (Join-Path $distRoot 'styles.css') -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot 'config.js') -Destination (Join-Path $distRoot 'config.js') -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot 'data\featureOrder.js') -Destination (Join-Path $distRoot 'data\featureOrder.js') -Force

$supabaseUrl = [Environment]::GetEnvironmentVariable('OELM_SUPABASE_URL') | ConvertTo-Json -Compress
$publishableKey = [Environment]::GetEnvironmentVariable('OELM_SUPABASE_PUBLISHABLE_KEY') | ConvertTo-Json -Compress
$hfUrl = [Environment]::GetEnvironmentVariable('OELM_HF_SPACE_URL') | ConvertTo-Json -Compress
$adminApiUrl = ([Environment]::GetEnvironmentVariable('OELM_SUPABASE_URL').TrimEnd('/') + '/functions/v1/oelm-admin') | ConvertTo-Json -Compress

$config = @"
window.OELM_CONFIG = {
  SUPABASE_URL: $supabaseUrl,
  SUPABASE_KEY: $publishableKey,
  HF_SPACE_URL: $hfUrl,
  API_BASE_URL: $hfUrl,
  ADMIN_API_URL: $adminApiUrl,
};
"@

Set-Content -LiteralPath (Join-Path $distRoot 'config.local.js') -Value $config -Encoding utf8NoBOM
Write-Output 'Prepared dist with deployment configuration. dist/src/config.local.js remains ignored by Git.'
