try {
    $r = Invoke-WebRequest -Uri 'http://127.0.0.1:11434' -TimeoutSec 3
    Write-Output $r.StatusCode
} catch {
    Write-Output ("ERR: " + $_.Exception.Message)
}
