param()
$ErrorActionPreference='Stop'
$taskName='MemeGuard-Final-Test-v1'
$root=(Get-Item 'D:/career-projects/MemeGuard').FullName
$python=(Get-Item 'D:/career-projects/envs/memeguard/Scripts/python.exe').FullName
$log=Join-Path 'D:/career-projects/runs/MemeGuard/20261027-final-test-v1' 'task-pipeline.log'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Task already registered; inspect it before starting again.' }
if (Get-CimInstance Win32_Process -Filter "name='python.exe'") { throw 'Python process already present; inspect it before launch.' }
$account=[System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action=New-ScheduledTaskAction -Execute (Join-Path $env:SystemRoot 'System32/cmd.exe') -Argument ('/d /c '+$python+' -u scripts/final_test.py --stage all > '+$log+' 2>&1') -WorkingDirectory $root
$principal=New-ScheduledTaskPrincipal -UserId $account -LogonType S4U -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Resume the already frozen MemeGuard final test; no recurring trigger.' | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
