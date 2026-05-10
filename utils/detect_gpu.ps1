try {
    $g = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notlike "*Intel*" } | Select-Object -First 1
    if (-not $g) {
        $g = Get-CimInstance Win32_VideoController | Select-Object -First 1
    }
    if ($g) {
        $samples = Get-Counter "\GPU Engine(*)\Utilization Percentage" -ErrorAction SilentlyContinue
        if ($samples) {
            $dgpu = $samples.CounterSamples | Where-Object { $_.InstanceName -like "*luid_*" } | Group-Object { ($_.InstanceName -split "_engtype")[0] } | Sort-Object Count -Descending | Select-Object -First 1
            if ($dgpu) {
                Write-Output ($dgpu.Name + "|" + $g.Name)
            } else {
                Write-Output ("NONE|" + $g.Name)
            }
        } else {
            Write-Output ("NO_SAMPLES|" + $g.Name)
        }
    } else {
        Write-Output "NO_GPU|N/A"
    }
} catch {
    Write-Output ("ERROR|" + $_.Exception.Message)
}
