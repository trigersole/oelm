$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot 'src'
$distRoot = Join-Path $projectRoot 'dist\src'
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'data') | Out-Null

Copy-Item -LiteralPath (Join-Path $projectRoot 'index.html') -Destination (Join-Path $projectRoot 'dist\index.html') -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot 'app.js') -Destination (Join-Path $distRoot 'app.js') -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot 'styles.css') -Destination (Join-Path $distRoot 'styles.css') -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot 'config.js') -Destination (Join-Path $distRoot 'config.js') -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot 'data\featureOrder.js') -Destination (Join-Path $distRoot 'data\featureOrder.js') -Force

$localConfig = Join-Path $distRoot 'config.local.js'
if (Test-Path -LiteralPath $localConfig) {
  Remove-Item -LiteralPath $localConfig
}
Write-Output 'Prepared dist. Public runtime values will load from the Supabase Edge Function.'
