# Ustad quickstart (Windows / PowerShell).
#
#   .\scripts\quickstart.ps1              set up, then launch the app
#   .\scripts\quickstart.ps1 -Cpu         force the CPU-only torch build
#   .\scripts\quickstart.ps1 -Check       set up, then run the hardware preflight
#   .\scripts\quickstart.ps1 -SkipInstall skip dependency install and just launch
#
# Nothing here reaches the network except pip and (optionally) the one-time model
# downloads. The Ollama daemon is started if it is not already listening; it is never
# stopped or killed, because it may be serving something else.

[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$Check,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

function Say([string]$text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Note([string]$text) { Write-Host "   $text" -ForegroundColor DarkGray }
function Die([string]$text) { Write-Host "`nx  $text" -ForegroundColor Red; exit 1 }

# --- 1. interpreter ------------------------------------------------------------------
# torch publishes cp312 wheels; a 3.13/3.14 interpreter will fail to resolve them, so
# the version is checked here rather than discovered halfway through a 3 GB download.

Say 'Locating Python 3.12'
$python = $null
foreach ($candidate in @(
        @{ Exe = 'py';     Args = @('-3.12') },
        @{ Exe = 'python'; Args = @() },
        @{ Exe = 'python3'; Args = @() })) {
    if (-not (Get-Command $candidate.Exe -ErrorAction SilentlyContinue)) { continue }
    try {
        $version = & $candidate.Exe @($candidate.Args + @('-c', 'import sys;print("%d.%d"%sys.version_info[:2])')) 2>$null
    } catch { continue }
    if ($version -match '^3\.(12|11|10)$') {
        $python = $candidate
        Note "$($candidate.Exe) $($candidate.Args -join ' ') -> Python $version"
        break
    }
}
if (-not $python) {
    Die 'No Python 3.10-3.12 found. Install Python 3.12 (torch has no wheels for 3.13+ yet) and re-run.'
}

# --- 2. virtual environment ----------------------------------------------------------

$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    Say 'Creating .venv'
    & $python.Exe @($python.Args + @('-m', 'venv', $venv))
    if (-not (Test-Path $venvPython)) { Die 'venv creation failed.' }
} else {
    Say 'Reusing existing .venv'
}

# --- 3. dependencies ----------------------------------------------------------------

if (-not $SkipInstall) {
    $hasNvidia = -not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    if ($hasNvidia) {
        try { $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1) }
        catch { $gpu = $null }
    }

    # cu126 rather than a newer CUDA: it is the most recent index that still ships
    # kernels for every arch from Maxwell through Hopper, which includes Turing (sm_75).
    $index = if ($hasNvidia) { 'https://download.pytorch.org/whl/cu126' }
             else { 'https://download.pytorch.org/whl/cpu' }

    Say ('Installing torch from {0}' -f (Split-Path -Leaf $index))
    if ($gpu) { Note "detected: $gpu" } elseif ($Cpu) { Note 'CPU build requested' } else { Note 'no NVIDIA GPU detected; installing the CPU build' }

    & $venvPython -m pip install --disable-pip-version-check --upgrade pip
    & $venvPython -m pip install --disable-pip-version-check torch --index-url $index
    if ($LASTEXITCODE -ne 0) { Die 'torch install failed.' }

    Say 'Installing the rest of the requirements'
    & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Die 'requirements install failed.' }
}

# --- 4. Ollama ----------------------------------------------------------------------

Say 'Checking the Ollama daemon'
$listening = $false
try {
    $response = Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 3 -UseBasicParsing
    $listening = $true
    Note "already running: $($response.Content)"
} catch {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Note 'not listening; starting `ollama serve` in the background'
        Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden | Out-Null
        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Milliseconds 500
            try {
                Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 -UseBasicParsing | Out-Null
                $listening = $true
                Note 'daemon is up'
                break
            } catch { }
        }
        if (-not $listening) { Note 'daemon did not come up in 10s; start it yourself with: ollama serve' }
    } else {
        Note 'ollama is not on PATH. Install it from https://ollama.com, then: ollama pull qwen2.5:3b'
    }
}

if ($listening) {
    $tags = (& $venvPython -c @'
import json, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as r:
        names = [m.get("name", "") for m in json.load(r).get("models", [])]
    print(" ".join(n for n in names if n))
except Exception:
    print("")
'@) 2>$null
    if ($tags.Trim()) {
        Note "teachers available: $tags"
    } else {
        Note 'no models pulled yet. A good 4 GB-friendly teacher: ollama pull qwen2.5:3b'
    }
}

# --- 5. go --------------------------------------------------------------------------

if ($Check) {
    Say 'Running the hardware preflight'
    & $venvPython (Join-Path $root 'scripts\smoke_test.py') --check
    exit $LASTEXITCODE
}

Say 'Starting Ustad on http://127.0.0.1:8177'
Note 'Ctrl+C to stop. Everything is written under .\data'
& $venvPython -m backend.server
