<#
.SYNOPSIS
    One entry point for setting up, running and testing the project.

.DESCRIPTION
    Every path is resolved relative to this script, so it does not matter which folder you are
    standing in when you run it. That is the point: the usual failure is running an ingest
    command from inside web\, which fails with "Could not open requirements file" and leaves a
    junk venv behind.

.EXAMPLE
    .\run.ps1 setup     First-time install. Safe to re-run.
    .\run.ps1 check     Everything CI runs. Do this before pushing.
    .\run.ps1 web       Web app on fixture data. No backend needed.
    .\run.ps1 api       Ingest service only, on :8080.
    .\run.ps1 full      Ingest + web together, wired to each other.
    .\run.ps1 serve     Build the UI and serve everything from ONE port. For competitions.
    .\run.ps1 doctor    What is installed and what is missing.
    .\run.ps1 db-check  Safely test the configured SQLite or Supabase database.
    .\run.ps1 native-setup     Start the Windows C++ dependency setup in the background.
    .\run.ps1 native-progress  Live progress bar for the Windows C++ dependency setup.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'check', 'web', 'api', 'full', 'serve', 'doctor', 'db-check', 'native-setup', 'native-progress', 'help')]
    [string]$Command = 'help'
)

# Node, Python and pytest all write UTF-8, but Windows PowerShell decodes a native program's
# output with the console's legacy code page, so an em dash arrives as three garbled characters
# (this comment stays ASCII: PowerShell 5.1 reads a BOM-less script as ANSI). Decoding as UTF-8
# fixes every tool at once, and it is what the console should have been doing anyway.
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

# NOT 'Stop'. In PowerShell 5.1 anything a native exe writes to stderr becomes an ErrorRecord,
# so a harmless deprecation warning from pytest or uvicorn aborts the whole script even though
# the process exited 0. Exit codes are checked explicitly instead.
$ErrorActionPreference = 'Continue'

# Runs a native command, echoes its output including stderr, and returns the exit code.
function Invoke-Native {
    param([scriptblock]$Block)
    & $Block 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Host $_.ToString() }
        else { Write-Host $_ }
    }
    return $LASTEXITCODE
}

$Repo = $PSScriptRoot
$Web = Join-Path $Repo 'web'
$VenvPy = Join-Path $Repo 'ingest\.venv\Scripts\python.exe'

function Say([string]$text, [string]$colour = 'Cyan') { Write-Host "`n$text" -ForegroundColor $colour }
function Ok([string]$text) { Write-Host "  ok    $text" -ForegroundColor Green }
function Warn([string]$text) { Write-Host "  warn  $text" -ForegroundColor Yellow }
function Bad([string]$text) { Write-Host "  FAIL  $text" -ForegroundColor Red }

function Have([string]$name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Initialize-NativeToolchain {
    # Find the C++ toolchain without making anyone edit PATH by hand.
    #
    # Visual Studio Build Tools ships its own CMake but never puts it on PATH, and vcpkg
    # downloads a second copy. So `cmake` can be missing from the shell on a machine that has
    # two working copies of it, which reads as "install CMake" when nothing needs installing.
    if (-not (Have 'cmake')) {
        $candidates = @()

        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (Test-Path $vswhere) {
            $vsRoot = & $vswhere -latest -products * -property installationPath 2>$null
            foreach ($root in @($vsRoot)) {
                if ($root) {
                    $candidates += (Join-Path $root 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin')
                }
            }
        }

        $candidates += 'C:\Program Files\CMake\bin'

        $vcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { 'C:\vcpkg' }
        $toolsDir = Join-Path $vcpkgRoot 'downloads\tools'
        if (Test-Path $toolsDir) {
            $candidates += (Get-ChildItem $toolsDir -Filter 'cmake-*' -Directory -ErrorAction SilentlyContinue |
                ForEach-Object { Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue } |
                ForEach-Object { Join-Path $_.FullName 'bin' })
        }

        foreach ($dir in $candidates) {
            if ($dir -and (Test-Path (Join-Path $dir 'cmake.exe'))) {
                $env:PATH = "$dir;$env:PATH"
                Ok "found cmake in $dir"
                break
            }
        }
    }

    # vcpkg lives at the conventional root. Only guess when the user has not chosen one.
    if (-not $env:VCPKG_ROOT) {
        $defaultVcpkg = 'C:\vcpkg'
        if (Test-Path (Join-Path $defaultVcpkg 'vcpkg.exe')) {
            $env:VCPKG_ROOT = $defaultVcpkg
            Ok "found vcpkg at $defaultVcpkg"
        }
    }
}

function Require-Venv {
    if (-not (Test-Path $VenvPy)) {
        Bad "No Python venv. Run: .\run.ps1 setup"
        exit 1
    }
}

# --------------------------------------------------------------------------- native progress

function Get-NativeVcpkgInstallDir {
    if (-not $env:VCPKG_ROOT) { return $null }
    # ONNX Runtime's Windows port mishandles a package install path containing spaces.
    # Put derived packages under the conventional C:\vcpkg root even when REPO has spaces.
    return (Join-Path $env:VCPKG_ROOT 'frc-analysis-installed')
}

function Invoke-NativeSetup {
    if (-not $env:VCPKG_ROOT) {
        Bad 'VCPKG_ROOT is not set. Follow the vcpkg step in docs\RUNNING.md first.'
        exit 1
    }
    $vcpkg = Join-Path $env:VCPKG_ROOT 'vcpkg.exe'
    if (-not (Test-Path $vcpkg)) {
        Bad "vcpkg.exe was not found at $vcpkg"
        exit 1
    }
    if (Get-Process -Name vcpkg -ErrorAction SilentlyContinue) {
        Warn 'A vcpkg installation is already running. Use: .\run.ps1 native-progress'
        return
    }

    $installDir = Get-NativeVcpkgInstallDir
    $logPath = Join-Path ([System.IO.Path]::GetTempPath()) 'frc-vcpkg-install.log'
    $errorLogPath = Join-Path ([System.IO.Path]::GetTempPath()) 'frc-vcpkg-install-error.log'
    Remove-Item -LiteralPath $logPath, $errorLogPath -Force -ErrorAction SilentlyContinue
    $process = Start-Process -FilePath $vcpkg -WorkingDirectory $Repo `
        -ArgumentList 'install', '--x-manifest-root=analysis', "--x-install-root=$installDir", '--triplet', 'x64-windows' `
        -RedirectStandardOutput $logPath -RedirectStandardError $errorLogPath -WindowStyle Hidden -PassThru

    Say 'Native dependency setup started.' 'Green'
    Write-Host "    Process: $($process.Id)"
    Write-Host '    Open another PowerShell and run: .\run.ps1 native-progress'
}

function Get-NativeInstallProgress {
    # Parse vcpkg's real package counter rather than guessing from elapsed time.
    $logPath = Join-Path ([System.IO.Path]::GetTempPath()) 'frc-vcpkg-install.log'
    if (-not (Test-Path $logPath)) { return $null }

    $line = Get-Content -LiteralPath $logPath -Tail 250 -ErrorAction SilentlyContinue |
        Where-Object { $_ -match 'Installing\s+(\d+)\/(\d+)\s+' } |
        Select-Object -Last 1
    if (-not $line -or $line -notmatch 'Installing\s+(\d+)\/(\d+)\s+') { return $null }

    return [pscustomobject]@{
        Current = [int]$Matches[1]
        Total = [int]$Matches[2]
        Detail = ($line -replace '^.*Installing\s+\d+\/\d+\s+', '')
        LogPath = $logPath
    }
}

function Invoke-NativeProgress {
    $logPath = Join-Path ([System.IO.Path]::GetTempPath()) 'frc-vcpkg-install.log'
    $sawInstaller = $false

    while ($true) {
        $progress = Get-NativeInstallProgress
        $installer = Get-Process -Name vcpkg -ErrorAction SilentlyContinue
        if ($installer) { $sawInstaller = $true }

        if ($progress) {
            $percent = [math]::Min(100, [math]::Round((100 * $progress.Current) / $progress.Total))
            Write-Progress -Activity 'Installing C++ video-analysis dependencies' `
                -Status "$($progress.Current)/$($progress.Total): $($progress.Detail)" `
                -PercentComplete $percent
        }
        elseif ($installer) {
            Write-Progress -Activity 'Installing C++ video-analysis dependencies' `
                -Status 'Preparing the dependency list...' -PercentComplete 0
        }
        elseif ($sawInstaller) {
            Write-Progress -Activity 'Installing C++ video-analysis dependencies' -Completed
            Say 'Native dependency installer stopped.' 'Green'
            Write-Host "    See the final output at: $logPath"
            Write-Host '    Next: run the CMake build command in docs\RUNNING.md.'
            return
        }
        else {
            Warn 'No vcpkg installer is running and no progress log was found.'
            Write-Host '    Start the Windows native setup first; see docs\RUNNING.md.'
            return
        }

        Start-Sleep -Seconds 2
    }
}

# --------------------------------------------------------------------------- doctor

function Invoke-Doctor {
    Say 'Tooling'
    foreach ($t in @(
        @{ n = 'node';   why = 'web app' },
        @{ n = 'npm';    why = 'web app' },
        @{ n = 'python'; why = 'ingest service' },
        @{ n = 'ffmpeg'; why = 'video download and fixture rendering' },
        @{ n = 'git';    why = 'obviously' },
        @{ n = 'cmake';  why = 'building the C++ locally (optional, CI does it)' },
        @{ n = 'ollama'; why = 'local vision models for labelling (optional)' }
    )) {
        if (Have $t.n) { Ok "$($t.n)  -  $($t.why)" } else { Warn "$($t.n) missing  -  needed for: $($t.why)" }
    }

    Say 'Project'
    if (Test-Path $VenvPy) { Ok 'Python venv' } else { Warn 'Python venv missing  -  run: .\run.ps1 setup' }
    if (Test-Path (Join-Path $Web 'node_modules')) { Ok 'npm packages' } else { Warn 'npm packages missing  -  run: .\run.ps1 setup' }
    if (Test-Path (Join-Path $Repo 'ingest\.env')) { Ok 'ingest\.env' } else { Warn 'ingest\.env missing  -  run: .\run.ps1 setup' }
    if (Test-Path (Join-Path $Web '.env.local')) { Ok 'web\.env.local' } else { Warn 'web\.env.local missing  -  created by: .\run.ps1 full' }

    # Report which secrets are set, never their values.
    $envFile = Join-Path $Repo 'ingest\.env'
    if (Test-Path $envFile) {
        Say 'Secrets (names only, never values)'
        foreach ($key in @('TBA_API_KEY', 'SHEETS_SPREADSHEET_ID', 'GOOGLE_APPLICATION_CREDENTIALS')) {
            $line = Select-String -Path $envFile -Pattern "^\s*$key\s*=\s*(.+)$" -ErrorAction SilentlyContinue
            if ($line) { Ok "$key is set" } else { Warn "$key is empty  -  see docs\RUNNING.md" }
        }
    }

    Say 'Stray files'
    if (Test-Path (Join-Path $Web 'ingest')) {
        Bad 'web\ingest\ exists  -  a venv made from the wrong folder. Delete it:'
        Write-Host '        Remove-Item -Recurse -Force web\ingest' -ForegroundColor Red
    }
    else { Ok 'none' }
}

# --------------------------------------------------------------------------- setup

function Invoke-Setup {
    Say 'Web packages'
    Push-Location $Web
    try { npm install } finally { Pop-Location }
    Ok 'npm install'

    Say 'Python venv'
    if (-not (Test-Path $VenvPy)) {
        Push-Location $Repo
        try { python -m venv ingest\.venv } finally { Pop-Location }
    }
    & $VenvPy -m pip install --quiet --upgrade pip
    & $VenvPy -m pip install --quiet -r (Join-Path $Repo 'ingest\requirements.txt')
    & $VenvPy -m pip install --quiet pytest
    Ok 'ingest dependencies'

    Say 'Config'
    $envFile = Join-Path $Repo 'ingest\.env'
    if (-not (Test-Path $envFile)) {
        Copy-Item (Join-Path $Repo 'ingest\.env.example') $envFile
        Ok 'created ingest\.env  -  open it and add your TBA key'
    }
    else { Ok 'ingest\.env already exists, left alone' }

    if (Test-Path (Join-Path $Web 'ingest')) {
        Remove-Item -Recurse -Force (Join-Path $Web 'ingest')
        Ok 'removed stray web\ingest\'
    }

    Say 'Done. Next:' 'Green'
    Write-Host '    .\run.ps1 web     just the UI, on fixture data'
    Write-Host '    .\run.ps1 full    ingest service + UI together'
    Write-Host '    .\run.ps1 check   run every test'
}

# --------------------------------------------------------------------------- check

function Invoke-Check {
    Require-Venv
    $failed = @()

    Say 'web  -  typecheck'
    Push-Location $Web
    try {
        $code = Invoke-Native { npm run typecheck }
        if ($code -ne 0) { $failed += 'typecheck' } else { Ok 'typecheck' }

        Say 'web  -  build'
        $code = Invoke-Native { npm run build }
        if ($code -ne 0) { $failed += 'build' } else { Ok 'build' }

        Say 'fixtures  -  validate against contracts'
        $code = Invoke-Native { npm run validate:fixtures }
        if ($code -ne 0) { $failed += 'validate:fixtures' } else { Ok 'fixtures' }
    }
    finally { Pop-Location }

    Push-Location $Repo
    try {
        Say 'ingest  -  unit tests'
        $code = Invoke-Native { & $VenvPy -m pytest ingest\tests -q }
        if ($code -ne 0) { $failed += 'pytest' } else { Ok 'pytest' }

        Say 'ingest  -  Contract E smoke test'
        $code = Invoke-Native { & $VenvPy -m ingest.smoke_test }
        if ($code -ne 0) { $failed += 'smoke_test' } else { Ok 'smoke test' }

        # The generator is deterministic, so regenerating must change nothing. Catches anyone
        # hand-editing golden data instead of changing the generator.
        Say 'fixtures  -  reproducible?'
        $code = Invoke-Native { node fixtures\tools\generate_fixture.mjs --no-video }
        if ($code -ne 0) {
            Bad 'fixture generator failed'
            $failed += 'fixture generation'
        }
        else {
            # Only the files the generator actually writes. `-- fixtures` would also flag a
            # hand edit to README.md or tools/, reporting "regenerating changed the fixtures"
            # for a doc typo -- which sends you hunting for non-determinism that isn't there.
            $generated = @('fixtures', ':(exclude)fixtures/README.md', ':(exclude)fixtures/tools/*')
            git diff --quiet -- $generated
            if ($LASTEXITCODE -ne 0) {
                Bad 'regenerating changed the fixtures'
                git --no-pager diff --stat -- $generated
                $failed += 'fixture reproducibility'
            }
            else { Ok 'fixtures reproduce byte-for-byte' }
        }

        Say 'analysis  -  configure, build and real-video smoke test'
        Initialize-NativeToolchain
        if (-not (Have 'cmake')) {
            # Skipped, not failed. CI always has cmake, so coverage is not lost -- and the
            # whole point of CI building the C++ is that nobody NEEDS a local toolchain.
            # Hard-failing here makes `check` impossible to pass for anyone working only on
            # the web app or the ingest service, which is the opposite of the intent.
            Warn 'cmake not installed  -  skipping the C++ build. CI still checks it.'
            Warn 'To build locally: install CMake + a compiler, see docs\RUNNING.md'
        }
        else {
            $analysisDir = Join-Path $Repo 'analysis'
            $analysisBuild = Join-Path $analysisDir 'build'
            $configureArgs = @('-S', $analysisDir, '-B', $analysisBuild)
            if ($env:VCPKG_ROOT) {
                $vcpkgToolchain = Join-Path $env:VCPKG_ROOT 'scripts\buildsystems\vcpkg.cmake'
                if (Test-Path $vcpkgToolchain) {
                    $configureArgs += "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
                    $configureArgs += "-DVCPKG_INSTALLED_DIR=$(Get-NativeVcpkgInstallDir)"
                    Ok 'using vcpkg from VCPKG_ROOT'
                }
                else { Warn 'VCPKG_ROOT is set but its CMake toolchain was not found' }
            }
            else { Warn 'VCPKG_ROOT is not set; CMake must find OpenCV another way' }
            $code = Invoke-Native { & cmake @configureArgs }
            if ($code -ne 0) {
                $failed += 'analysis configure'
            }
            else {
                $code = Invoke-Native { cmake --build $analysisBuild --config Release }
                if ($code -ne 0) {
                    $failed += 'analysis build'
                }
                else {
                    $binary = Get-ChildItem -LiteralPath (Join-Path $analysisBuild 'bin') -Recurse -File |
                        Where-Object { $_.Name -in @('analysis.exe', 'analysis') } |
                        Select-Object -First 1
                    if (-not $binary) {
                        Bad 'analysis binary was not produced'
                        $failed += 'analysis smoke test'
                    }
                    else {
                        $fixtureJob = Get-Content (Join-Path $Repo 'fixtures\2026casf_qm42\job.json') -Raw |
                            ConvertFrom-Json
                        $fixtureJob.local_path = (Resolve-Path (Join-Path $Repo 'fixtures\2026casf_qm42\segment.mp4')).Path
                        # Prove the binary reads the video itself rather than trusting job metadata.
                        $fixtureJob.duration = 1
                        $fixtureJob.fps = 1
                        $fixtureJob.width = 1
                        $fixtureJob.height = 1
                        $tempJob = Join-Path ([System.IO.Path]::GetTempPath()) 'frc-scouting-analysis-job.json'
                        $tempOut = Join-Path ([System.IO.Path]::GetTempPath()) 'frc-scouting-analysis-out'
                        $fixtureJob | ConvertTo-Json -Depth 10 | Set-Content -Path $tempJob -Encoding utf8
                        New-Item -ItemType Directory -Force -Path $tempOut | Out-Null
                        # This must remain a stable plumbing test even when the developer has a
                        # real detector configured in their shell.
                        $hadDetectorConfig = Test-Path Env:FRC_DETECTOR_CONFIG
                        $previousDetectorConfig = $env:FRC_DETECTOR_CONFIG
                        Remove-Item Env:FRC_DETECTOR_CONFIG -ErrorAction SilentlyContinue
                        try {
                            $code = Invoke-Native {
                                & $binary.FullName --job $tempJob --season (Join-Path $Repo 'contracts\seasons\2026.json') --out $tempOut
                            }
                        }
                        finally {
                            if ($hadDetectorConfig) { $env:FRC_DETECTOR_CONFIG = $previousDetectorConfig }
                        }
                        $expectedVersion = [int]((Get-Content (Join-Path $Repo 'contracts\SCHEMA_VERSION') -Raw).Trim())
                        $resultPath = Join-Path $tempOut 'result.json'
                        $tracksPath = Join-Path $tempOut 'tracks.jsonl'
                        $eventsPath = Join-Path $tempOut 'events.jsonl'
                        if ($code -ne 0 -or -not (Test-Path $resultPath) -or -not (Test-Path $tracksPath) -or
                            -not (Test-Path $eventsPath)) {
                            Bad 'analysis smoke test did not produce Contract D output'
                            $failed += 'analysis smoke test'
                        }
                        else {
                            $result = Get-Content $resultPath -Raw | ConvertFrom-Json
                            $tracks = @(Get-Content $tracksPath | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
                            $events = @(Get-Content $eventsPath | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
                            if ($result.schema_version -ne $expectedVersion -or $result.frames_total -le 1 -or
                                $result.frames_analyzed -ne 0 -or $result.tracks_emitted -ne 0 -or
                                $result.model_version -ne 'detector-unconfigured' -or $tracks.Count -ne 0 -or
                                $events.Count -ne 2 -or $events[0].event_type -ne 'match_start' -or
                                $events[1].event_type -ne 'match_end') {
                                Bad 'analysis smoke output is not the expected unconfigured-detector baseline'
                                $failed += 'analysis smoke test'
                            }
                            else { Ok 'analysis real-video smoke test' }
                        }
                    }

                    # Tracker and detector behaviour. Both are pure -- no model or video needed
                    # -- so the awkward cases (crossing robots, occlusion, camera cuts, a 16:9
                    # frame in a square input, anchor clusters) are cheap to check directly and
                    # catch regressions the smoke test cannot see.
                    foreach ($nativeTest in @('tracker_test.exe', 'detector_test.exe')) {
                        $found = Get-ChildItem -Path (Join-Path $repo 'analysis/build') -Recurse -Filter $nativeTest -ErrorAction SilentlyContinue | Select-Object -First 1
                        if (-not $found) { continue }
                        $label = $nativeTest -replace '_test\.exe$', ' tests'
                        $testOut = & $found.FullName 2>&1 | Out-String
                        Write-Host $testOut
                        if ($LASTEXITCODE -ne 0) {
                            Bad "$label failed"
                            $failed += $label
                        }
                        else { Ok $label }
                    }
                }
            }
        }
    }
    finally { Pop-Location }

    if ($failed.Count -gt 0) {
        Say "FAILED: $($failed -join ', ')" 'Red'
        exit 1
    }
    Say 'All checks passed.' 'Green'
}

# --------------------------------------------------------------------------- run

function Invoke-DbCheck {
    Require-Venv
    Say 'Database connection'
    Push-Location $Repo
    try {
        $code = Invoke-Native { & $VenvPy -m ingest.db_check }
        if ($code -ne 0) {
            Bad 'database connection failed; see docs\CENTRAL-SETUP.md'
            exit $code
        }
        Ok 'database connection and tables'
    }
    finally { Pop-Location }
}

function Invoke-Web {
    Say 'Web app on fixture data  -  http://localhost:5173' 'Green'
    Write-Host '  No backend needed. Ctrl+C to stop.'
    Push-Location $Web
    try { npm run dev } finally { Pop-Location }
}

function Invoke-Api {
    Require-Venv
    Say 'Ingest service  -  http://localhost:8080' 'Green'
    Write-Host '  Health: http://localhost:8080/api/health'
    Write-Host '  Docs:   http://localhost:8080/docs'
    Push-Location $Repo
    try { & $VenvPy -m uvicorn ingest.main:app --reload --port 8080 } finally { Pop-Location }
}

function Invoke-Full {
    Require-Venv

    # Vite only reads env vars at startup, so a file beats $env: -- it survives closing the shell.
    $localEnv = Join-Path $Web '.env.local'
    if (-not (Test-Path $localEnv)) {
        Set-Content -Path $localEnv -Value 'VITE_API_MODE=http' -Encoding utf8
        Ok 'created web\.env.local  ->  VITE_API_MODE=http'
    }
    elseif (-not (Select-String -Path $localEnv -Pattern 'VITE_API_MODE\s*=\s*http' -Quiet)) {
        Warn 'web\.env.local exists but does not set VITE_API_MODE=http  -  the UI will use fixtures'
    }

    Say 'Starting the ingest service in a second window...'
    $apiCmd = "Set-Location '$Repo'; & '$VenvPy' -m uvicorn ingest.main:app --reload --port 8080"
    Start-Process powershell -ArgumentList '-NoExit', '-Command', $apiCmd | Out-Null
    Start-Sleep -Seconds 3

    try {
        $health = Invoke-RestMethod -Uri 'http://localhost:8080/api/health' -TimeoutSec 5
        Ok "ingest is up  -  schema_version $($health.schema_version)"
    }
    catch {
        Warn 'ingest did not answer yet; check the other window for errors'
    }

    Say 'Web app  -  http://localhost:5173' 'Green'
    Write-Host '  Analysis emits real RF-DETR boxes only when FRC_DETECTOR_CONFIG names a trained local ONNX model.'
    Write-Host '  Without that config, zero tracks is expected; it never invents a diagnostic robot box.'
    Push-Location $Web
    try { npm run dev } finally { Pop-Location }
}


function Invoke-Serve {
    Require-Venv

    # Build once, then let the ingest service hand out the static files. One process, one port,
    # which is what you want on a laptop at a venue -- everyone else just opens your IP.
    Say 'Building the web app...'
    Push-Location $Web
    try {
        $code = Invoke-Native { npm run build }
        if ($code -ne 0) { Bad 'build failed'; exit 1 }
    }
    finally { Pop-Location }
    Ok 'built to web\dist'

    $ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
        Select-Object -ExpandProperty IPAddress

    Say 'Serving everything on port 8080' 'Green'
    Write-Host '  This machine : http://localhost:8080'
    foreach ($ip in $ips) { Write-Host "  Others       : http://${ip}:8080" }
    Write-Host ''
    Write-Host '  Others on the same wifi use one of those addresses. If they cannot reach it,'
    Write-Host '  Windows Firewall is blocking the port -- allow python.exe on private networks.'

    Push-Location $Repo
    try { & $VenvPy -m uvicorn ingest.main:app --host 0.0.0.0 --port 8080 } finally { Pop-Location }
}

# --------------------------------------------------------------------------- dispatch

switch ($Command) {
    'setup'  { Invoke-Setup }
    'check'  { Invoke-Check }
    'web'    { Invoke-Web }
    'api'    { Invoke-Api }
    'full'   { Invoke-Full }
    'serve'  { Invoke-Serve }
    'clean'  { Require-Venv; Push-Location $Repo; try { & $VenvPy -m ingest.retention @args } finally { Pop-Location } }
    'doctor' { Invoke-Doctor }
    'db-check' { Invoke-DbCheck }
    'native-setup' { Invoke-NativeSetup }
    'native-progress' { Invoke-NativeProgress }
    default {
        Write-Host @'
Project Tengen

  .\run.ps1 setup     First-time install. Safe to re-run.
  .\run.ps1 doctor    What is installed and what is missing.

  .\run.ps1 web       Web app on fixture data. No backend needed.
  .\run.ps1 api       Ingest service only, on :8080.
  .\run.ps1 full      Ingest + web together, wired to each other.
  .\run.ps1 serve     Build the UI and serve it all on one port. For competitions.

  .\run.ps1 check     Everything CI runs. Do this before pushing.
  .\run.ps1 native-setup     Start Windows C++ dependency setup in the background.
  .\run.ps1 native-progress  Live bar for a running Windows C++ dependency install.

Run it from anywhere -- paths resolve relative to the script, not your shell.
Full guide: docs\RUNNING.md
'@
    }
}
