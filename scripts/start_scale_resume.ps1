param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$python = "D:/career-projects/envs/memeguard/Scripts/python.exe"
Push-Location $root
try {
    & $python -u scripts/run_scale_resume.py --checkpoint $Checkpoint --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Resume runner failed: $LASTEXITCODE" }
} finally { Pop-Location }
