// Global chart instances
let trainingChart = null;
let comparisonChart = null;
let comparisonTrainingChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateSliderValues();
});

function initializeEventListeners() {
    // Run experiment button
    document.getElementById('run-btn').addEventListener('click', runExperiment);
    
    // Compare noise levels button
    document.getElementById('compare-btn').addEventListener('click', compareNoiseLevels);
    
    // Clear history button
    document.getElementById('clear-btn').addEventListener('click', clearHistory);
    
    // Update slider values
    document.getElementById('epsilon').addEventListener('input', function() {
        document.getElementById('epsilon-value').textContent = this.value;
    });
    
    document.getElementById('clip_norm').addEventListener('input', function() {
        document.getElementById('clip_norm-value').textContent = this.value;
    });
    
    // Handle privacy technique selection (checkboxes)
    document.getElementById('use_dp').addEventListener('change', updatePrivacyControls);
    document.getElementById('use_smpc').addEventListener('change', updatePrivacyControls);
    document.getElementById('use_he').addEventListener('change', updatePrivacyControls);
    
    // Initialize privacy controls visibility
    updatePrivacyControls();
    
    // Handle dataset selection
    document.getElementById('dataset_name').addEventListener('change', function() {
        const uploadControls = document.getElementById('upload-controls');
        if (this.value === 'uploaded') {
            uploadControls.classList.remove('hidden');
        } else {
            uploadControls.classList.add('hidden');
        }
    });
    
    // Handle file upload
    document.getElementById('upload-btn').addEventListener('click', uploadDataset);
}

function updatePrivacyControls() {
    const useDp = document.getElementById('use_dp').checked;
    const useSmpc = document.getElementById('use_smpc').checked;
    const useHe = document.getElementById('use_he').checked;
    
    const dpControls = document.getElementById('dp-controls');
    const smpcControls = document.getElementById('smpc-controls');
    const heControls = document.getElementById('he-controls');
    
    // Show/hide controls based on checkboxes
    if (useDp) {
        dpControls.classList.remove('hidden');
    } else {
        dpControls.classList.add('hidden');
    }
    
    if (useSmpc) {
        smpcControls.classList.remove('hidden');
    } else {
        smpcControls.classList.add('hidden');
    }
    
    if (useHe) {
        heControls.classList.remove('hidden');
    } else {
        heControls.classList.add('hidden');
    }
}

function updateSliderValues() {
    const epsilon = document.getElementById('epsilon').value;
    const clipNorm = document.getElementById('clip_norm').value;
    document.getElementById('epsilon-value').textContent = epsilon;
    document.getElementById('clip_norm-value').textContent = clipNorm;
}

async function runExperiment() {
    const btn = document.getElementById('run-btn');
    btn.disabled = true;
    
    // Show loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('current-results').classList.add('hidden');
    document.getElementById('comparison-results').classList.add('hidden');
    document.getElementById('error-message').classList.add('hidden');
    
    try {
        // Get selected privacy techniques (checkboxes)
        const useDp = document.getElementById('use_dp').checked;
        const useSmpc = document.getElementById('use_smpc').checked;
        const useHe = document.getElementById('use_he').checked;
        const datasetName = document.getElementById('dataset_name').value;
        
        // Validate dataset selection
        if (datasetName === 'uploaded' && !uploadedDatasetInfo) {
            showError('Please upload a dataset file first');
            btn.disabled = false;
            document.getElementById('loading').classList.add('hidden');
            return;
        }
        
        const config = {
            num_clients: parseInt(document.getElementById('num_clients').value),
            num_rounds: parseInt(document.getElementById('num_rounds').value),
            local_epochs: parseInt(document.getElementById('local_epochs').value),
            use_dp: useDp,
            use_smpc: useSmpc,
            use_he: useHe,
            epsilon: parseFloat(document.getElementById('epsilon').value),
            delta: parseFloat(document.getElementById('delta').value),
            clip_norm: parseFloat(document.getElementById('clip_norm').value),
            dataset_name: datasetName === 'uploaded' && uploadedDatasetInfo 
                ? `uploaded:${uploadedDatasetInfo.filename}` 
                : datasetName,
            num_samples: datasetName === 'mnist' ? 5000 : 1000  // More samples for MNIST
        };
        
        const response = await fetch('/api/run_experiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        // Get response text (can only read once)
        const responseText = await response.text();
        
        // Check if response is OK
        if (!response.ok) {
            console.error('Server error response:', responseText);
            try {
                const errorJson = JSON.parse(responseText);
                throw new Error(errorJson.error || `Server error: ${response.status}`);
            } catch (parseError) {
                throw new Error(`Server error (${response.status}): ${responseText.substring(0, 200)}`);
            }
        }
        
        // Try to parse JSON
        let result;
        try {
            result = JSON.parse(responseText);
        } catch (parseError) {
            console.error('JSON parse error:', parseError);
            console.error('Response text (first 500 chars):', responseText.substring(0, 500));
            throw new Error(`Failed to parse server response: ${parseError.message}. Response may contain non-JSON data.`);
        }
        
        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'Unknown error occurred');
            console.error('Experiment failed:', result);
        }
    } catch (error) {
        showError('Failed to run experiment: ' + error.message);
    } finally {
        btn.disabled = false;
        document.getElementById('loading').classList.add('hidden');
    }
}

async function compareNoiseLevels() {
    const btn = document.getElementById('compare-btn');
    btn.disabled = true;
    
    // Show loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('current-results').classList.add('hidden');
    document.getElementById('comparison-results').classList.add('hidden');
    document.getElementById('error-message').classList.add('hidden');
    
    try {
        const datasetName = document.getElementById('dataset_name').value;
        
        // Validate dataset selection
        if (datasetName === 'uploaded' && uploadedDatasetInfo) {
            showError('Compare noise levels is not supported with uploaded datasets. Please use synthetic or MNIST.');
            btn.disabled = false;
            document.getElementById('loading').classList.add('hidden');
            return;
        }
        
        const config = {
            epsilon_values: [0.1, 0.5, 1.0, 2.0, 5.0],
            num_rounds: parseInt(document.getElementById('num_rounds').value),
            num_clients: parseInt(document.getElementById('num_clients').value),
            local_epochs: parseInt(document.getElementById('local_epochs').value),
            dataset_name: datasetName === 'uploaded' && uploadedDatasetInfo 
                ? `uploaded:${uploadedDatasetInfo.filename}` 
                : datasetName,
            num_samples: datasetName === 'mnist' ? 5000 : 1000
        };
        
        const response = await fetch('/api/compare_noise', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayComparisonResults(result);
        } else {
            showError(result.error || 'Unknown error occurred');
        }
    } catch (error) {
        showError('Failed to compare noise levels: ' + error.message);
    } finally {
        btn.disabled = false;
        document.getElementById('loading').classList.add('hidden');
    }
}

function displayResults(result) {
    try {
        // Validate result structure
        if (!result || !result.round_metrics || result.round_metrics.length === 0) {
            throw new Error('Invalid result: missing round_metrics');
        }
        
        // Update metrics
        const accuracyEl = document.getElementById('final-accuracy');
        const lossEl = document.getElementById('final-loss');
        const privacyEl = document.getElementById('current-epsilon');
        
        if (!accuracyEl || !lossEl || !privacyEl) {
            throw new Error('Required DOM elements not found');
        }
        
        accuracyEl.textContent = result.final_accuracy ? result.final_accuracy.toFixed(2) + '%' : 'N/A';
        lossEl.textContent = result.final_loss ? result.final_loss.toFixed(4) : 'N/A';
        
        // Update privacy technique display
        let privacyInfo = [];
        if (result.config) {
            if (result.config.use_dp) {
                privacyInfo.push(`DP (ε=${result.config.epsilon || 'N/A'})`);
            }
            if (result.config.use_smpc) {
                privacyInfo.push('SMPC');
            }
            if (result.config.use_he) {
                privacyInfo.push('HE');
            }
        }
        privacyEl.textContent = privacyInfo.length > 0 ? privacyInfo.join(' + ') : 'None';
        
        // Create training chart
        const chartCanvas = document.getElementById('training-chart');
        if (!chartCanvas) {
            throw new Error('Chart canvas element not found');
        }
        
        const ctx = chartCanvas.getContext('2d');
        if (!ctx) {
            throw new Error('Could not get 2D context from canvas');
        }
        
        // Check if Chart is available
        if (typeof Chart === 'undefined') {
            throw new Error('Chart.js library not loaded. Please check your internet connection.');
        }
        
        if (trainingChart) {
            trainingChart.destroy();
        }
        
        const rounds = result.round_metrics.map(m => m.round || m.round_num || 0);
        const accuracies = result.round_metrics.map(m => m.accuracy || 0);
        const losses = result.round_metrics.map(m => m.loss || 0);
        
        // Validate data
        if (rounds.length === 0 || accuracies.length === 0) {
            throw new Error('No valid metrics data to display');
        }
        
        trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: rounds,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracies,
                    borderColor: 'rgb(102, 126, 234)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                },
                {
                    label: 'Loss',
                    data: losses,
                    borderColor: 'rgb(244, 67, 54)',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Training Progress Over Rounds',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
    
    // Show results
    document.getElementById('current-results').classList.remove('hidden');
    
    } catch (error) {
        console.error('Error displaying results:', error);
        showError('Error displaying results: ' + error.message);
        // Still try to show results panel even if chart fails
        try {
            document.getElementById('current-results').classList.remove('hidden');
        } catch (e) {
            console.error('Could not show results panel:', e);
        }
    }
}

function displayComparisonResults(result) {
    try {
        // Validate result structure
        if (!result || !result.results || result.results.length === 0) {
            throw new Error('Invalid comparison result: missing results');
        }
        
        // Create accuracy vs epsilon chart
        const chartCanvas1 = document.getElementById('comparison-chart');
        if (!chartCanvas1) {
            throw new Error('Comparison chart canvas not found');
        }
        
        const ctx1 = chartCanvas1.getContext('2d');
        if (typeof Chart === 'undefined') {
            throw new Error('Chart.js library not loaded');
        }
        
        if (comparisonChart) {
            comparisonChart.destroy();
        }
        
        const epsilons = result.results.map(r => r.epsilon || 0);
        const accuracies = result.results.map(r => r.final_accuracy || 0);
        const baselineAccuracy = result.baseline ? result.baseline.final_accuracy : null;
    
    const datasets = [
        {
            label: 'With Differential Privacy',
            data: accuracies,
            borderColor: 'rgb(102, 126, 234)',
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            tension: 0.4,
            pointRadius: 6
        }
    ];
    
    if (baselineAccuracy !== null) {
        datasets.push({
            label: 'Baseline (No DP)',
            data: new Array(epsilons.length).fill(baselineAccuracy),
            borderColor: 'rgb(76, 175, 80)',
            backgroundColor: 'rgba(76, 175, 80, 0.1)',
            borderDash: [5, 5],
            pointRadius: 0
        });
    }
    
    comparisonChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: epsilons.map(e => `ε=${e}`),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Privacy-Accuracy Trade-off: Lower ε = More Privacy = Lower Accuracy',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Final Accuracy (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Privacy Budget (ε)'
                    }
                }
            }
        }
    });
    
    // Create training curves comparison
    const chartCanvas2 = document.getElementById('comparison-training-chart');
    if (!chartCanvas2) {
        throw new Error('Comparison training chart canvas not found');
    }
    
    const ctx2 = chartCanvas2.getContext('2d');
    
    if (comparisonTrainingChart) {
        comparisonTrainingChart.destroy();
    }
    
    const trainingDatasets = result.results.map((r, idx) => {
        const colors = [
            'rgb(102, 126, 234)',
            'rgb(244, 67, 54)',
            'rgb(255, 152, 0)',
            'rgb(76, 175, 80)',
            'rgb(156, 39, 176)'
        ];
        return {
            label: `ε=${r.epsilon}`,
            data: r.round_metrics.map(m => m.accuracy),
            borderColor: colors[idx % colors.length],
            backgroundColor: colors[idx % colors.length].replace('rgb', 'rgba').replace(')', ', 0.1)'),
            tension: 0.4
        };
    });
    
    if (result.baseline) {
        trainingDatasets.push({
            label: 'Baseline (No DP)',
            data: result.baseline.round_metrics.map(m => m.accuracy),
            borderColor: 'rgb(0, 0, 0)',
            backgroundColor: 'rgba(0, 0, 0, 0.1)',
            borderDash: [5, 5],
            tension: 0.4
        });
    }
    
    comparisonTrainingChart = new Chart(ctx2, {
        type: 'line',
        data: {
            labels: result.results[0].round_metrics.map(m => `Round ${m.round}`),
            datasets: trainingDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Training Curves: Accuracy Over Rounds for Different Privacy Levels',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Round'
                    }
                }
            }
        }
    });
    
    // Show comparison results
    document.getElementById('comparison-results').classList.remove('hidden');
    
    } catch (error) {
        console.error('Error displaying comparison results:', error);
        showError('Error displaying comparison: ' + error.message);
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.textContent = 'Error: ' + message;
        errorDiv.classList.remove('hidden');
        // Auto-hide after 10 seconds
        setTimeout(() => {
            errorDiv.classList.add('hidden');
        }, 10000);
    }
    console.error('Error:', message);
}

async function clearHistory() {
    try {
        await fetch('/api/clear_history', {
            method: 'POST'
        });
        alert('History cleared successfully!');
    } catch (error) {
        showError('Failed to clear history: ' + error.message);
    }
}

// Global variable to store uploaded dataset info
let uploadedDatasetInfo = null;

async function uploadDataset() {
    const fileInput = document.getElementById('dataset_file');
    const uploadBtn = document.getElementById('upload-btn');
    const statusDiv = document.getElementById('upload-status');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('Please select a file to upload');
        return;
    }
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    statusDiv.classList.remove('hidden');
    statusDiv.textContent = 'Uploading dataset...';
    statusDiv.className = 'upload-status';
    
    try {
        const response = await fetch('/api/upload_dataset', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            uploadedDatasetInfo = result;
            statusDiv.textContent = `✓ Uploaded: ${result.num_samples} samples, ${result.num_features} features`;
            statusDiv.className = 'upload-status success';
            
            // Update dataset selector to use uploaded file
            const datasetSelect = document.getElementById('dataset_name');
            datasetSelect.innerHTML = `
                <option value="synthetic">Synthetic Data</option>
                <option value="mnist">MNIST (Handwritten Digits)</option>
                <option value="uploaded:${result.filename}" selected>Uploaded: ${result.filename}</option>
            `;
        } else {
            statusDiv.textContent = '✗ Upload failed: ' + result.error;
            statusDiv.className = 'upload-status error';
            showError(result.error || 'Upload failed');
        }
    } catch (error) {
        statusDiv.textContent = '✗ Upload failed: ' + error.message;
        statusDiv.className = 'upload-status error';
        showError('Failed to upload dataset: ' + error.message);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '📤 Upload Dataset';
    }
}

