# =============================================================================
# Spectral Separability Explorer — Launcher Script
# =============================================================================
# Run from any PowerShell prompt:
#   PS> & "D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox\run.ps1"
#
# Or from the project root:
#   PS> .\run.ps1
#
# What it does:
#   1. Navigates to the project root
#   2. Kills any stale process holding ports 7860..7870
#   3. Activates the virtual environment
#   4. Launches python app.py
# =============================================================================

# ── 1. Resolve project root from this script's location ─────────────────────
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $projectRoot
Write-Host ""
Write-Host "📁 Project root: $projectRoot" -ForegroundColor Cyan

# ── 2. Kill stale processes on Gradio port range ────────────────────────────
Write-Host ""
Write-Host "🧹 Checking ports 7860-7870 for stale processes..." -ForegroundColor Yellow
$killed = 0
foreach ($port in 7860..7870) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
            Where-Object { $_.State -eq 'Listen' }
    if ($conn) {
        $pid_to_kill = $conn.OwningProcess | Select-Object -Unique
        foreach ($p in $pid_to_kill) {
            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host "   ✗ Killing PID $p ($($proc.ProcessName)) on port $port" -ForegroundColor DarkYellow
                Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                $killed++
            }
        }
    }
}
if ($killed -eq 0) {
    Write-Host "   ✅ All ports clean." -ForegroundColor Green
} else {
    Write-Host "   ✅ Killed $killed stale process(es)." -ForegroundColor Green
    Start-Sleep -Milliseconds 500   # give Windows a moment to release the sockets
}

# ── 3. Activate venv ────────────────────────────────────────────────────────
Write-Host ""
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Host "❌ venv not found at $venvActivate" -ForegroundColor Red
    Write-Host "   Run: python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt" -ForegroundColor DarkRed
    exit 1
}
Write-Host "🔧 Activating venv..." -ForegroundColor Cyan
& $venvActivate

# ── 4. Launch the app ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "🚀 Launching Spectral Separability Explorer..." -ForegroundColor Magenta
Write-Host "   (Ctrl+C to stop)" -ForegroundColor DarkGray
Write-Host ""
python app.py
