# Configure stable virtual memory for local Qwen loading on this Windows machine.
# Requires elevation and a restart. Keeps a small C: page file and adds D: capacity.
[CmdletBinding()]
param(
    [string]$StatusPath = "D:\1-Projects\1-TacticalReport\2-Final_Mock_Project\1-tactical-recommendation-prototype\output\portfolio\phase2\pagefile_configuration.json"
)

$ErrorActionPreference = "Stop"
$status = [ordered]@{
    status = "starting"
    changed_at = (Get-Date).ToUniversalTime().ToString("o")
    restart_required = $true
}

try {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Administrator elevation is required"
    }
    $drive = Get-PSDrive -Name D
    if ($drive.Free -lt 20GB) {
        throw "D: needs at least 20 GB free before creating the ML page file"
    }

    $computer = Get-CimInstance Win32_ComputerSystem
    Set-CimInstance -InputObject $computer -Property @{ AutomaticManagedPagefile = $false } | Out-Null

    $desired = @(
        @{ Name = "C:\pagefile.sys"; InitialSize = [uint32]1024; MaximumSize = [uint32]2048 },
        @{ Name = "D:\pagefile.sys"; InitialSize = [uint32]8192; MaximumSize = [uint32]16384 }
    )
    foreach ($item in $desired) {
        $setting = Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue |
            Where-Object Name -eq $item.Name
        if ($setting) {
            Set-CimInstance -InputObject $setting -Property @{
                InitialSize = [uint32]$item.InitialSize
                MaximumSize = [uint32]$item.MaximumSize
            } | Out-Null
        } else {
            New-CimInstance Win32_PageFileSetting -Property $item | Out-Null
        }
    }
    $status.status = "configured_restart_required"
    $status.settings = $desired
} catch {
    $status.status = "failed"
    $status.error = "{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message
}

$directory = Split-Path -Parent $StatusPath
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
if ($status.status -eq "failed") {
    throw $status.error
}
