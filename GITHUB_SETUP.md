# 📦 GitHub Repository Setup Guide

## Initial Setup

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon → **"New repository"**
3. Repository name: `federated-privacy-demo` (or your preferred name)
4. Description: `Federated Learning with Differential Privacy, SMPC, and Homomorphic Encryption`
5. Set to **Public** (or Private if preferred)
6. **Don't** initialize with README, .gitignore, or license (we already have these)
7. Click **"Create repository"**

### 2. Initialize Git in Your Project

```bash
# Navigate to project directory
cd federated-privacy-demo

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Federated learning with privacy-preserving techniques"

# Add remote repository (replace with your URL)
git remote add origin https://github.com/YOUR_USERNAME/federated-privacy-demo.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 3. Add Team Members as Collaborators

1. Go to your repository on GitHub
2. Click **Settings** → **Collaborators** → **Add people**
3. Add your teammates by their GitHub usernames
4. They'll receive an invitation email

### 4. Set Up Branch Protection (Optional but Recommended)

1. Go to **Settings** → **Branches**
2. Add rule for `main` branch
3. Require pull request reviews (optional)
4. This prevents accidental pushes to main

## Workflow for Team Collaboration

### Option 1: Feature Branches (Recommended)

```bash
# Create a feature branch
git checkout -b feature/differential-privacy

# Make changes and commit
git add .
git commit -m "Implement differential privacy mechanism"

# Push branch
git push origin feature/differential-privacy

# Create Pull Request on GitHub
# Teammates can review and merge
```

### Option 2: Direct Collaboration (For Hackathon Speed)

```bash
# Always pull before starting work
git pull origin main

# Make changes
# ...

# Commit and push
git add .
git commit -m "Your changes"
git push origin main
```

## Adding Files to Repository

### What to Include ✅
- All Python source files (`.py`)
- `requirements.txt`
- `README.md`
- `CONTRIBUTING.md`
- `.gitignore`
- Documentation files

### What NOT to Include ❌
- `__pycache__/` directories (handled by .gitignore)
- `venv/` or virtual environment (handled by .gitignore)
- `.pyc` files (handled by .gitignore)
- Large data files
- Model checkpoints (unless small)

## Making Your Repository Stand Out

### 1. Add Repository Topics
On GitHub repository page:
- Click the gear icon next to "About"
- Add topics: `federated-learning`, `privacy`, `differential-privacy`, `homomorphic-encryption`, `machine-learning`, `pytorch`

### 2. Add a Repository Description
"Federated Learning with Privacy-Preserving Techniques: DP, SMPC, and HE"

### 3. Pin Important Files
- README.md (automatically shown)
- Add badges if desired (build status, Python version, etc.)

### 4. Create a Good README
- Clear project description
- Setup instructions
- Usage examples
- Screenshots/demos (if possible)

## Quick Git Commands Reference

```bash
# Check status
git status

# See changes
git diff

# Add specific file
git add path/to/file.py

# Commit with message
git commit -m "Descriptive message"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main

# Create new branch
git checkout -b branch-name

# Switch branches
git checkout branch-name

# See commit history
git log --oneline
```

## Resolving Conflicts

If you get merge conflicts:

```bash
# Pull latest changes
git pull origin main

# If conflicts occur, Git will mark them
# Edit files to resolve conflicts
# Look for <<<<<<< HEAD markers

# After resolving:
git add .
git commit -m "Resolve merge conflicts"
git push origin main
```

## Hackathon-Specific Tips

1. **Commit Often**: Don't wait until the end
2. **Clear Commit Messages**: "Add DP noise mechanism" not "fix stuff"
3. **Push Regularly**: Don't lose work
4. **Use Issues**: Track bugs and tasks
5. **Tag Releases**: Tag your final submission

```bash
# Tag final submission
git tag -a v1.0 -m "Hackathon submission"
git push origin v1.0
```

## Repository Checklist

Before submission, ensure:
- [ ] All code is pushed to GitHub
- [ ] README is complete and clear
- [ ] Requirements.txt is accurate
- [ ] .gitignore is working (no unnecessary files)
- [ ] Repository is public (or judges have access)
- [ ] Team members are added as collaborators
- [ ] Repository has a good description and topics

## Need Help?

- [GitHub Docs](https://docs.github.com)
- [Git Basics](https://git-scm.com/book)
- Ask your teammates!

---

**Your repository is your portfolio piece - make it shine! ✨**

