# Export Docker image for transfer to Portainer host
# Run this from PowerShell on your Windows machine

Write-Host "Exporting parking-monitor Docker image..." -ForegroundColor Cyan

# Export the image
docker save parking_spot_monitoring-parking-monitor:latest -o parking-monitor.tar

if ($LASTEXITCODE -eq 0) {
    $size = (Get-Item parking-monitor.tar).Length / 1MB
    Write-Host "`nImage exported successfully!" -ForegroundColor Green
    Write-Host "File: parking-monitor.tar ($([math]::Round($size, 2)) MB)" -ForegroundColor Yellow
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Transfer to your Portainer host:"
    Write-Host "   scp parking-monitor.tar user@your-portainer-host:/tmp/"
    Write-Host "`n2. On the Portainer host, load the image:"
    Write-Host "   docker load -i /tmp/parking-monitor.tar"
    Write-Host "   docker tag parking_spot_monitoring-parking-monitor:latest parking-monitor:latest"
} else {
    Write-Host "Failed to export image" -ForegroundColor Red
}
