# 🐛 Debugging Guide

## Quick Troubleshooting

### 1. Charts Not Showing

**Check:**
- Open browser console (F12)
- Look for errors like "Chart is not defined"
- Check Network tab for Chart.js CDN loading

**Fix:**
```javascript
// In browser console:
console.log(typeof Chart);  // Should be "function"
// If undefined, Chart.js didn't load - check internet connection
```

### 2. Experiments Failing

**Check Terminal Output:**
```bash
# Look for:
- Import errors
- Dataset loading errors  
- Memory errors
- Invalid parameter errors
```

**Common Fixes:**
- Install missing dependencies: `pip install -r requirements.txt`
- Check dataset format (must have X and y)
- Reduce num_samples if out of memory
- Verify all parameters are valid numbers

### 3. File Upload Not Working

**Check:**
- File size (max 100MB)
- File format (.npz, .pkl, .pth, .pt)
- File structure (must contain 'X' and 'y')

**Debug:**
```javascript
// Browser console:
const file = document.getElementById('dataset_file').files[0];
console.log(file.name, file.size);
```

### 4. WebSocket Issues

**Check:**
- Server is running
- Port 5000 is accessible
- Browser console for connection errors

**Test:**
```javascript
// Browser console:
const socket = io();
socket.on('connect', () => console.log('Connected!'));
```

---

## Error Messages Reference

### Backend Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | Missing dependency | `pip install -r requirements.txt` |
| `FileNotFoundError` | Uploaded file missing | Re-upload dataset |
| `ValueError: NPZ file must contain 'X' and 'y'` | Invalid dataset format | Check file structure |
| `RuntimeError: TenSEAL not available` | HE library missing | Install TenSEAL or disable HE |

### Frontend Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Chart is not defined` | Chart.js not loaded | Check internet/CDN |
| `Canvas element not found` | HTML structure issue | Check template |
| `Invalid result: missing round_metrics` | API returned bad data | Check backend logs |

---

## Debugging Steps

### Step 1: Check Server Logs
```bash
# Terminal where app.py is running
# Look for:
- "Starting experiment..."
- "Round X completed"
- Error tracebacks
```

### Step 2: Check Browser Console
```javascript
// Press F12 → Console tab
// Look for:
- JavaScript errors
- API call failures
- Chart.js errors
```

### Step 3: Check Network Tab
```
F12 → Network tab
- Check API requests (should be 200 OK)
- Check WebSocket connection
- Check Chart.js CDN loading
```

### Step 4: Validate Data
```python
# Add to app.py temporarily:
print(f"Dataset shape: {X.shape}, {y.shape}")
print(f"Model parameters: {len(list(model.parameters()))}")
```

---

## Common Issues & Solutions

### Issue: "Experiments run but no charts appear"

**Solution:**
1. Check browser console for Chart.js errors
2. Verify canvas elements exist in HTML
3. Check if `displayResults()` is being called
4. Verify result structure has `round_metrics`

### Issue: "Upload button does nothing"

**Solution:**
1. Check file input has a file selected
2. Check browser console for JavaScript errors
3. Verify upload endpoint is accessible
4. Check file size (max 100MB)

### Issue: "WebSocket connection fails"

**Solution:**
1. Verify Flask-SocketIO is installed
2. Check server is running
3. Try different browser
4. Check firewall/antivirus blocking port 5000

### Issue: "Memory errors with large datasets"

**Solution:**
1. Reduce `num_samples` parameter
2. Use fewer clients
3. Reduce `num_rounds`
4. Use synthetic data instead of real dataset

---

## Testing Checklist

Before reporting issues, verify:

- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Server starts without errors: `python app.py`
- [ ] Browser console has no errors (F12)
- [ ] Network tab shows successful API calls
- [ ] Chart.js CDN loads (check Network tab)
- [ ] File uploads work (if using custom dataset)
- [ ] Parameters are valid (positive numbers, etc.)

---

## Getting Help

If issues persist:

1. **Check logs:** Terminal output + Browser console
2. **Verify setup:** All dependencies installed
3. **Test minimal case:** Try with default settings
4. **Check documentation:** See CODE_DOCUMENTATION.md

---

## Performance Debugging

### Slow Experiments

**Check:**
- Number of clients (fewer = faster)
- Number of rounds (fewer = faster)
- Privacy techniques (HE is slowest)
- Dataset size (smaller = faster)

**Profile:**
```python
import time
start = time.time()
# ... code ...
print(f"Took {time.time() - start:.2f}s")
```

---

## Memory Debugging

### Out of Memory

**Symptoms:**
- Process killed
- "MemoryError" exceptions
- System becomes slow

**Fixes:**
- Reduce `num_samples`
- Use fewer clients
- Disable HE (uses most memory)
- Close other applications

---

**Last Updated:** 2024

