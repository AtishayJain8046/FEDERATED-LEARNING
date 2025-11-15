# 🚀 Quick Guide: Create GitHub Repository

## Step-by-Step Instructions

### Step 1: Create Repository on GitHub (2 minutes)

1. **Go to GitHub**: Open https://github.com/new in your browser
2. **Repository name**: `FEDERATED-LEARNING`
3. **Description**: `Federated Learning with Differential Privacy, SMPC, and Homomorphic Encryption`
4. **Visibility**: Choose **Public** (or Private if you prefer)
5. **Important**: 
   - ❌ **DO NOT** check "Add a README file"
   - ❌ **DO NOT** check "Add .gitignore"
   - ❌ **DO NOT** check "Choose a license"
   (We already have these files!)
6. Click **"Create repository"**

### Step 2: Connect Local Repository to GitHub

After creating the repository, run this in PowerShell (from the project directory):

```powershell
# Navigate to project
cd C:\Users\atish\federated-privacy-demo

# Run the setup script
.\setup_github.ps1
```

**OR** run these commands manually:

```powershell
# Set remote (replace with your actual URL if different)
git remote add origin https://github.com/atishayjain8046/FEDERATED-LEARNING.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Authentication

If you get authentication errors:

**Option A: Use GitHub Desktop**
1. Download [GitHub Desktop](https://desktop.github.com/)
2. Sign in with your GitHub account
3. Use GitHub Desktop to push

**Option B: Use Personal Access Token**
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` permissions
3. Use token as password when pushing

**Option C: Use GitHub CLI** (if installed)
```powershell
gh auth login
gh repo create FEDERATED-LEARNING --public --source=. --remote=origin --push
```

## Quick Commands Summary

```powershell
# Make sure you're in the project directory
cd C:\Users\atish\federated-privacy-demo

# Check git status
git status

# If not committed yet:
git add .
git commit -m "Initial commit: Federated learning with privacy-preserving techniques"

# Connect to GitHub (after creating repo on GitHub website)
git remote add origin https://github.com/atishayjain8046/FEDERATED-LEARNING.git
git branch -M main
git push -u origin main
```

## Verify It Worked

After pushing, visit:
**https://github.com/atishayjain8046/FEDERATED-LEARNING**

You should see all your files there!

## Troubleshooting

### "Repository not found"
- Make sure you created the repository on GitHub first
- Check the repository name matches exactly: `FEDERATED-LEARNING`

### "Authentication failed"
- Use GitHub Desktop, or
- Set up a Personal Access Token, or
- Use GitHub CLI: `gh auth login`

### "Remote origin already exists"
```powershell
git remote remove origin
git remote add origin https://github.com/atishayjain8046/FEDERATED-LEARNING.git
```

## Next Steps After Repository is Created

1. ✅ Add repository topics (federated-learning, privacy, etc.)
2. ✅ Add team members as collaborators
3. ✅ Star your own repo! ⭐
4. ✅ Share the link with your teammates

---

**Need help?** Check `GITHUB_SETUP.md` for more detailed instructions.

