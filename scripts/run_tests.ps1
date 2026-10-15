# Run the full test suite. Extra arguments are passed to pytest.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PYTHON) { $env:PYTHON }
      elseif (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" }
      else { "python" }
& $py -m pytest @args
exit $LASTEXITCODE
