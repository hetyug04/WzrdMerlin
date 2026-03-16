try {
    $samples = (Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage' -ErrorAction Stop).CounterSamples
    $total = ($samples | Measure-Object -Property CookedValue -Sum).Sum
    Write-Output "$([math]::Round($total,1))"
} catch {
    Write-Output "-1"
}
