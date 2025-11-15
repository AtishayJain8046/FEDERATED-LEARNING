# ⚡ Quick Start Guide (5 Minutes)

## For Beginners - Get Running Fast!

### Step 1: Install Python
- Download from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation
- Verify: Open terminal and type `python --version` (should show 3.8+)

### Step 2: Open Terminal/Command Prompt
- **Windows**: Press `Win + R`, type `cmd`, press Enter
- **Mac/Linux**: Open Terminal app

### Step 3: Navigate to Project
```bash
cd path/to/federated-privacy-demo
```

### Step 4: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux  
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your command line.

### Step 5: Install Packages
```bash
pip install torch numpy scikit-learn matplotlib tqdm
```

**Note**: TenSEAL (for HE) is optional. Install later if needed:
```bash
pip install tenseal
```

### Step 6: Test It Works
```bash
python experiments/run_demo.py
```

You should see output showing:
- ✅ Basic Federated Learning demo
- ✅ Differential Privacy demo  
- ✅ SMPC demo
- ⚠️ HE demo (warning if TenSEAL not installed - that's OK!)

## That's It! 🎉

You're ready to start coding. Check out:
- `CONTRIBUTING.md` - Your work assignment
- `README.md` - Full documentation
- `HACKATHON_TIPS.md` - Winning strategy

## Need Help?

**"Module not found" error?**
- Make sure virtual environment is activated (see `(venv)` in prompt)
- Run `pip install -r requirements.txt` again

**"Python not found" error?**
- Python might not be in PATH
- Try `python3` instead of `python`
- Reinstall Python with "Add to PATH" checked

**Demo doesn't run?**
- Check you're in the right directory
- Make sure all files are present
- Try running: `python -c "import torch; print('OK')"`

## Next Steps

1. ✅ Get demo running
2. ✅ Read your assignment in `CONTRIBUTING.md`
3. ✅ Start coding your component
4. ✅ Test as you go
5. ✅ Have fun!

Good luck! 🚀

