# GitHub Repository Setup Script
# Run this AFTER creating the repository on GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Repository Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is initialized
if (-not (Test-Path .git)) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    git add .
    git commit -m "Initial commit: Federated learning with privacy-preserving techniques"
}

# Set remote repository
$username = "atishayjain8046"
$repoName = "FEDERATED-LEARNING"
$remoteUrl = "https://github.com/$username/$repoName.git"

Write-Host "Setting up remote repository..." -ForegroundColor Yellow
Write-Host "Repository URL: $remoteUrl" -ForegroundColor Green
Write-Host ""

# Remove existing remote if any
git remote remove origin 2>$null

# Add remote
git remote add origin $remoteUrl

# Set branch to main
git branch -M main

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Go to: https://github.com/new" -ForegroundColor Yellow
Write-Host "2. Repository name: FEDERATED-LEARNING" -ForegroundColor Yellow
Write-Host "3. Description: Federated Learning with Differential Privacy, SMPC, and Homomorphic Encryption" -ForegroundColor Yellow
Write-Host "4. Set to Public (or Private)" -ForegroundColor Yellow
Write-Host "5. DO NOT initialize with README, .gitignore, or license" -ForegroundColor Yellow
Write-Host "6. Click 'Create repository'" -ForegroundColor Yellow
Write-Host ""
Write-Host "After creating the repository, run:" -ForegroundColor Cyan
Write-Host "  git push -u origin main" -ForegroundColor Green
Write-Host ""
Write-Host "Or run this script again and it will push automatically!" -ForegroundColor Cyan
Write-Host ""

# Ask if repository is already created
$response = Read-Host "Have you already created the repository on GitHub? (y/n)"

if ($response -eq "y" -or $response -eq "Y") {
    Write-Host ""
    Write-Host "Pushing code to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Success! Repository is now on GitHub!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Repository URL: https://github.com/$username/$repoName" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "Error pushing to GitHub. Make sure:" -ForegroundColor Red
        Write-Host "1. Repository exists on GitHub" -ForegroundColor Red
        Write-Host "2. You're authenticated (use GitHub Desktop or configure git credentials)" -ForegroundColor Red
        Write-Host "3. You have write access to the repository" -ForegroundColor Red
    }
} else {
    Write-Host ""
    Write-Host "Please create the repository first, then run this script again!" -ForegroundColor Yellow
}

