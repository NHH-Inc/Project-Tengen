param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$Model = 'yolo11n.pt',
    [int]$Epochs = 100,
    [int]$ImageSize = 960,
    [string]$Device = '0',
    [string]$VenvPath = 'C:\yolo11-venv'
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Venv = [System.IO.Path]::GetFullPath($VenvPath)
$Python = Join-Path $Venv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    py -3.13 -m venv --system-site-packages $Venv
}

& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'Could not update the vision environment packaging tools.' }
& $Python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    throw 'Install CUDA-enabled PyTorch in the YOLO11 environment first; see docs\AUTOMATIC-DATASET.md.'
}
& $Python -m pip install -r (Join-Path $PSScriptRoot 'requirements-yolo.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the YOLO vision dependencies.' }

Push-Location $Repo
try {
    & $Python -m training.train_yolo --data $Dataset --output $Output --model $Model `
        --epochs $Epochs --image-size $ImageSize --device $Device
    if ($LASTEXITCODE -ne 0) { throw "YOLO training failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
