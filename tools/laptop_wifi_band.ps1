# Laptop Wi-Fi band switch for the Scout hotspot (Killer 1435 adapter "Wi-Fi").
# Right-click -> "Run with PowerShell", click Yes on the admin prompt.
#   default      : 2.4 GHz only ("05 - 11 b/g/n") so the Windows hotspot runs on 2.4 GHz where the Radxa can see it
#   -Revert      : back to the normal setting ("12 - 11 a/b/g/n/ac")
param([switch]$Revert)

$value = if ($Revert) { "12 - 11 a/b/g/n/ac" } else { "05 - 11 b/g/n" }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($Revert) { $argList += " -Revert" }
    Start-Process powershell -Verb RunAs -ArgumentList $argList
    exit
}

Write-Host "Setting Wi-Fi 'Wireless Mode' to: $value"
Set-NetAdapterAdvancedProperty -Name "Wi-Fi" -DisplayName "Wireless Mode" -DisplayValue $value
Get-NetAdapterAdvancedProperty -Name "Wi-Fi" -DisplayName "Wireless Mode" | Format-Table DisplayName, DisplayValue -AutoSize
Write-Host "Done. Wi-Fi reconnects in a few seconds; then turn Mobile hotspot off and on."
Read-Host "Press Enter to close"
