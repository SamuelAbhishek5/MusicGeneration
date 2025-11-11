# PowerShell script to test the music generation endpoint
# Usage: .\test_server.ps1

$serverUrl = "http://127.0.0.1:5000"

# Test 1: Health check
Write-Host "Testing health endpoint..."
try {
    $healthResponse = Invoke-WebRequest -Uri "$serverUrl/health" -Method Get -ErrorAction Stop
    Write-Host "Health: $($healthResponse.StatusCode) - $($healthResponse.Content)"
} catch {
    Write-Host "Health endpoint failed: $($_)"
}

# Test 2: Generate music
Write-Host "`nTesting music generation..."
$payload = @{
    user_id = "default"
    mood = "happiness"
    confidence = 0.8
} | ConvertTo-Json

try {
    $generateResponse = Invoke-WebRequest -Uri "$serverUrl/generate" `
        -Method Post `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $payload `
        -ErrorAction Stop
    Write-Host "Generation: $($generateResponse.StatusCode)"
    Write-Host "Response: $($generateResponse.Content)"
} catch {
    Write-Host "Generation failed: $($_)"
}
