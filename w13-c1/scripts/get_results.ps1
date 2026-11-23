# get_results.ps1: 서버의 C1 실험 결과를 내 PC로 받기 (Windows PowerShell)  김지현 2024134018
# 사용: Tailscale 연결 후, 결과를 받을 PC 폴더에서
#   powershell -ExecutionPolicy Bypass -File get_results.ps1 -Account vdb0X
# 서버에서는 아무것도 바꾸지 않고 복사만 한다.
param(
    [Parameter(Mandatory = $true)][string]$Account,     # cap-vdb 계정 (vdb01~vdb03 중 본인 것)
    [string]$Server = "100.82.53.54",
    [int]$Port = 2222,
    [string]$Remote = "/workspace/jihyeon/c1_w13",
    [string]$Local = "$HOME\Documents\capstone\c1_w13"
)
New-Item -ItemType Directory -Force -Path "$Local\results", "$Local\logs" | Out-Null

# 1) 결과 묶음(.tgz), 검증 결과, 로그를 통째로 받기
scp -P $Port "${Account}@${Server}:${Remote}/results/*.tgz" "$Local\results\"
scp -P $Port "${Account}@${Server}:${Remote}/results/check_store.json" "$Local\results\"
scp -P $Port "${Account}@${Server}:${Remote}/logs/*" "$Local\logs\"

# 2) 받은 묶음 풀기 (Windows 10 이상에는 tar가 있음)
Get-ChildItem "$Local\results\*.tgz" | ForEach-Object {
    $dir = Join-Path $_.DirectoryName ($_.BaseName)
    if (-not (Test-Path $dir)) { tar -xzf $_.FullName -C $_.DirectoryName }
}
Write-Host "받은 곳: $Local"
Get-ChildItem "$Local\results" -Directory | ForEach-Object { Write-Host "  $($_.Name)\summary.md" }
