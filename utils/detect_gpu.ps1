try {
    $g = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notlike "*Intel*" } | Select-Object -First 1
    if (-not $g) {
        $g = Get-CimInstance Win32_VideoController | Select-Object -First 1
    }
    if ($g) {
        $samples = Get-Counter "\GPU Engine(*)\Utilization Percentage" -ErrorAction SilentlyContinue
        if ($samples) {
            # Find the most frequent LUID (which represents the active GPU)
            $dgpu = $samples.CounterSamples | Where-Object { $_.InstanceName -match "(luid_0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)" } | Group-Object { $matches[1] } | Sort-Object Count -Descending | Select-Object -First 1
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
