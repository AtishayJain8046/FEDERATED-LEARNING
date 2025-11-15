# 🌐 Federated Learning Privacy Demo - Frontend

A beautiful web interface to visualize how noise affects model accuracy in federated learning with differential privacy.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the web server:**
   ```bash
   python app.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5000`

## ✨ Features

### Interactive Experiment Configuration
- Adjust number of clients, rounds, and local epochs
- Toggle differential privacy on/off
- Control privacy parameters:
  - **Privacy Budget (ε)**: Lower = More Privacy = More Noise
  - **Failure Probability (δ)**: Typically 1e-5
  - **Gradient Clip Norm**: Bounds gradient sensitivity

### Real-time Visualizations
- **Training Progress Chart**: See accuracy and loss over rounds
- **Privacy-Accuracy Trade-off**: Compare different noise levels
- **Training Curves Comparison**: Visualize how different ε values affect learning

### Key Insights
- Understand the fundamental trade-off: **Privacy vs Accuracy**
- See how lower ε values (stronger privacy) reduce model accuracy
- Compare DP-protected models against baseline (no privacy)

## 🎯 How to Use

### Running a Single Experiment

1. Configure your experiment parameters
2. Enable/disable differential privacy
3. Adjust privacy budget (ε) using the slider
4. Click **"🚀 Run Experiment"**
5. View results in the charts below

### Comparing Noise Levels

1. Click **"📊 Compare Noise Levels"**
2. The system will run experiments with ε = [0.1, 0.5, 1.0, 2.0, 5.0]
3. Compare the privacy-accuracy trade-off chart
4. See training curves for each privacy level

## 📊 Understanding the Results

### Privacy Budget (ε) Guidelines
- **ε = 0.1**: Very strong privacy, expect 10-20% accuracy drop
- **ε = 1.0**: Good balance (recommended), ~5-10% accuracy drop
- **ε = 10.0**: Weak privacy, minimal accuracy impact

### What You'll See
- **Final Accuracy**: Model performance after all rounds
- **Training Curves**: How accuracy improves over rounds
- **Baseline Comparison**: Performance without differential privacy

## 🔧 Technical Details

### Backend (Flask)
- RESTful API for running experiments
- Handles federated learning with optional DP
- Returns metrics and training history

### Frontend (HTML/CSS/JavaScript)
- Modern, responsive design
- Chart.js for visualizations
- Real-time updates and error handling

## 💡 Tips

1. **Start with a small number of rounds** (5-10) for quick testing
2. **Use "Compare Noise Levels"** to see the full trade-off curve
3. **Lower ε values** show more dramatic accuracy drops
4. **More clients** can help compensate for noise (but takes longer)

## 🐛 Troubleshooting

### Server won't start
- Make sure Flask is installed: `pip install flask flask-cors`
- Check if port 5000 is already in use

### Experiments fail
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check browser console for JavaScript errors

### Charts not showing
- Make sure Chart.js is loading (check browser console)
- Try refreshing the page

## 📝 API Endpoints

- `GET /` - Main page
- `POST /api/run_experiment` - Run single experiment
- `POST /api/compare_noise` - Compare multiple noise levels
- `GET /api/history` - Get experiment history
- `POST /api/clear_history` - Clear history

---

**Enjoy exploring the privacy-accuracy trade-off!** 🔒📊

