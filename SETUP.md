# 🚀 Setup Guide for Hackathon

## Quick Setup (5 minutes)

### Step 1: Clone and Navigate
```bash
git clone <your-repo-url>
cd federated-privacy-demo
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**If TenSEAL fails to install:**
```bash
# Option 1: Try pre-built wheels
pip install tenseal --prefer-binary

# Option 2: Skip HE for now (other demos will work)
pip install torch numpy opacus scikit-learn matplotlib tqdm
```

### Step 4: Test Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "from federated import FederatedClient; print('✓ Core imports work')"
```

### Step 5: Run Demo
```bash
python experiments/run_demo.py
```

## Verification Checklist

- [ ] Virtual environment activated
- [ ] All packages installed (check with `pip list`)
- [ ] Basic imports work
- [ ] Demo script runs without errors

## Common Issues

### "ModuleNotFoundError"
- Make sure virtual environment is activated
- Reinstall: `pip install -r requirements.txt`

### "TenSEAL installation failed"
- This is OK! Other demos (FL, DP, SMPC) will still work
- HE demo will show a warning but won't crash

### "CUDA errors"
- Code defaults to CPU, so this shouldn't happen
- If it does, set `device="cpu"` explicitly

## Next Steps

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for work division
2. Start with your assigned component
3. Test incrementally
4. Ask for help early!

Good luck! 🎉

