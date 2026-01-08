/**
 * Experiments and Bulk Training functionality for Neural Elements.
 *
 * Provides UI for:
 * - Configuring and submitting bulk training jobs
 * - Monitoring job progress
 * - Viewing and comparing experiment results
 * - Loading saved weights for visualization
 */

let activePollingIntervals = {};
let currentJobResults = null;

/**
 * Initialize the experiments tab.
 */
async function initExperiments() {
    // Load datasets for checkboxes
    try {
        const datasets = await fetch('/api/datasets').then(r => r.json());
        const container = document.getElementById('bulk-datasets');
        if (container) {
            container.innerHTML = '';
            const defaultSelected = ['spirals', 'xor', 'moons'];
            Object.entries(datasets).forEach(([key, info]) => {
                const label = document.createElement('label');
                const checked = defaultSelected.includes(key) ? 'checked' : '';
                label.innerHTML = `<input type="checkbox" value="${key}" ${checked}> ${info.name || key}`;
                container.appendChild(label);
            });
        }
    } catch (error) {
        console.error('Failed to load datasets:', error);
    }

    // Update summary on config change
    document.querySelectorAll('.experiment-config input').forEach(input => {
        input.addEventListener('change', updateConfigSummary);
    });

    updateConfigSummary();
    loadActiveJobs();
    loadCompletedJobs();
}

/**
 * Update the configuration summary showing total runs.
 */
function updateConfigSummary() {
    const depths = [...document.querySelectorAll('.depth-options input:checked')].map(i => parseInt(i.value));
    const widths = [...document.querySelectorAll('.width-options input:checked')].map(i => parseInt(i.value));
    const activations = [...document.querySelectorAll('.activation-options input:checked')].map(i => i.value);
    const datasets = [...document.querySelectorAll('#bulk-datasets input:checked')].map(i => i.value);
    const trials = parseInt(document.getElementById('bulk-trials')?.value) || 1;

    const elements = depths.length * widths.length * activations.length;
    const totalRuns = elements * datasets.length * trials;

    const summary = document.getElementById('config-summary');
    if (summary) {
        if (totalRuns === 0) {
            summary.innerHTML = '<span class="warning">Select at least one option from each category</span>';
        } else {
            summary.innerHTML =
                `This will run <strong>${totalRuns}</strong> training jobs ` +
                `(${elements} element${elements !== 1 ? 's' : ''} ` +
                `× ${datasets.length} dataset${datasets.length !== 1 ? 's' : ''} ` +
                `× ${trials} trial${trials !== 1 ? 's' : ''})`;
        }
    }
}

/**
 * Generate element configurations from selected options.
 */
function generateElementConfigs() {
    const depths = [...document.querySelectorAll('.depth-options input:checked')].map(i => parseInt(i.value));
    const widths = [...document.querySelectorAll('.width-options input:checked')].map(i => parseInt(i.value));
    const activations = [...document.querySelectorAll('.activation-options input:checked')].map(i => i.value);

    const elements = [];
    for (const depth of depths) {
        for (const width of widths) {
            for (const activation of activations) {
                elements.push({
                    hidden_layers: Array(depth).fill(width),
                    activation: activation
                });
            }
        }
    }
    return elements;
}

/**
 * Start a bulk training job.
 */
async function startBulkTraining() {
    const btn = document.getElementById('start-bulk-btn');
    if (!btn) return;

    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> Submitting...';

    try {
        const elements = generateElementConfigs();
        const datasets = [...document.querySelectorAll('#bulk-datasets input:checked')].map(i => i.value);

        if (elements.length === 0 || datasets.length === 0) {
            throw new Error('Please select at least one element configuration and one dataset');
        }

        const config = {
            elements: elements,
            datasets: datasets,
            training: {
                epochs: parseInt(document.getElementById('bulk-epochs')?.value) || 500,
                learning_rate: parseFloat(document.getElementById('bulk-lr')?.value) || 0.1,
                record_every: 50,
            },
            n_trials: parseInt(document.getElementById('bulk-trials')?.value) || 1
        };

        const response = await fetch('/api/bulk/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const result = await response.json();

        if (result.error) {
            throw new Error(result.error);
        }

        // Start polling for updates
        startJobPolling(result.job_id);
        loadActiveJobs();

    } catch (error) {
        console.error('Failed to start bulk training:', error);
        alert('Failed to start training: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Start Bulk Training';
    }
}

/**
 * Start polling for job updates.
 */
function startJobPolling(jobId) {
    // Poll every 2 seconds
    activePollingIntervals[jobId] = setInterval(async () => {
        try {
            const response = await fetch(`/api/bulk/jobs/${jobId}`);
            const job = await response.json();

            updateJobDisplay(job);

            if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
                clearInterval(activePollingIntervals[jobId]);
                delete activePollingIntervals[jobId];
                loadCompletedJobs();
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

/**
 * Update the display for a job.
 */
function updateJobDisplay(job) {
    const container = document.getElementById('active-jobs-list');
    if (!container) return;

    let jobEl = document.getElementById(`job-${job.job_id}`);

    if (!jobEl) {
        jobEl = document.createElement('div');
        jobEl.id = `job-${job.job_id}`;
        jobEl.className = 'job-card';
        container.insertBefore(jobEl, container.firstChild);
    }

    const progress = job.progress || {
        completed: job.completed_runs || 0,
        failed: job.failed_runs || 0,
        total: job.total_runs || 1,
    };
    const percent = Math.round(100 * (progress.completed + progress.failed) / Math.max(1, progress.total));

    const statusClass = {
        'pending': 'status-pending',
        'running': 'status-running',
        'completed': 'status-completed',
        'failed': 'status-failed',
        'cancelled': 'status-cancelled',
    }[job.status] || '';

    jobEl.innerHTML = `
        <div class="job-header">
            <span class="job-id">${job.job_id}</span>
            <span class="job-status ${statusClass}">${job.status}</span>
        </div>
        <div class="job-progress">
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${percent}%"></div>
            </div>
            <span class="progress-text">${progress.completed}/${progress.total} (${percent}%)</span>
        </div>
        ${progress.failed > 0 ? `<div class="job-errors">${progress.failed} failed</div>` : ''}
        ${job.status === 'running' ?
            `<button class="btn btn-secondary btn-small" onclick="cancelJob('${job.job_id}')">Cancel</button>` :
            ''}
        ${job.status === 'completed' ?
            `<button class="btn btn-primary btn-small" onclick="viewJobResults('${job.job_id}')">View Results</button>` :
            ''}
    `;
}

/**
 * Load active jobs on page load.
 */
async function loadActiveJobs() {
    try {
        const response = await fetch('/api/bulk/jobs?status=running');
        const data = await response.json();

        data.jobs.forEach(job => {
            updateJobDisplay(job);
            if (!activePollingIntervals[job.job_id]) {
                startJobPolling(job.job_id);
            }
        });

        // Also load pending jobs
        const pendingResponse = await fetch('/api/bulk/jobs?status=pending');
        const pendingData = await pendingResponse.json();
        pendingData.jobs.forEach(job => updateJobDisplay(job));

    } catch (error) {
        console.error('Failed to load active jobs:', error);
    }
}

/**
 * Load completed jobs.
 */
async function loadCompletedJobs() {
    try {
        const response = await fetch('/api/bulk/jobs?limit=10');
        const data = await response.json();

        const container = document.getElementById('completed-jobs-list');
        if (!container) return;

        const completedJobs = data.jobs.filter(j =>
            j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled'
        );

        if (completedJobs.length === 0) {
            container.innerHTML = '<p class="no-data">No completed experiments yet</p>';
            return;
        }

        container.innerHTML = completedJobs.map(job => `
            <div class="job-card completed" onclick="viewJobResults('${job.job_id}')">
                <div class="job-header">
                    <span class="job-date">${new Date(job.completed_at || job.created_at).toLocaleString()}</span>
                    <span class="job-status status-${job.status}">${job.status}</span>
                </div>
                <div class="job-summary">
                    ${job.completed_runs || 0} runs completed
                    ${job.failed_runs > 0 ? `, ${job.failed_runs} failed` : ''}
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Failed to load completed jobs:', error);
    }
}

/**
 * Cancel a running job.
 */
async function cancelJob(jobId) {
    if (!confirm('Are you sure you want to cancel this job?')) return;

    try {
        await fetch(`/api/bulk/jobs/${jobId}/cancel`, { method: 'POST' });
        loadActiveJobs();
    } catch (error) {
        console.error('Failed to cancel job:', error);
        alert('Failed to cancel job: ' + error.message);
    }
}

/**
 * View results for a completed job.
 */
async function viewJobResults(jobId) {
    try {
        const response = await fetch(`/api/bulk/jobs/${jobId}/results`);
        const data = await response.json();

        currentJobResults = data;

        const analysisSection = document.getElementById('results-analysis');
        if (analysisSection) {
            analysisSection.style.display = 'block';
            renderResultsComparison(data);
        }

    } catch (error) {
        console.error('Failed to load job results:', error);
        alert('Failed to load results: ' + error.message);
    }
}

/**
 * Render results comparison visualization.
 */
function renderResultsComparison(data) {
    const container = document.getElementById('comparison-chart');
    const summaryContainer = document.getElementById('best-performers');

    if (!container || !data.results || data.results.length === 0) {
        if (container) container.innerHTML = '<p>No results to display</p>';
        return;
    }

    // Build summary
    if (summaryContainer && data.summary) {
        let summaryHtml = '<h4>Best Performers</h4>';

        if (data.summary.best_overall) {
            const best = data.summary.best_overall;
            summaryHtml += `
                <div class="best-overall">
                    <strong>Overall Best:</strong>
                    ${best.element_name} on ${best.dataset}
                    (${(best.accuracy * 100).toFixed(1)}% accuracy)
                    <button class="btn btn-small" onclick="loadExperimentBoundary('${best.experiment_id}')">
                        View
                    </button>
                </div>
            `;
        }

        if (data.summary.best_by_dataset) {
            summaryHtml += '<div class="best-by-dataset"><strong>Best by Dataset:</strong><ul>';
            for (const [dataset, info] of Object.entries(data.summary.best_by_dataset)) {
                summaryHtml += `
                    <li>
                        <span class="dataset-name">${dataset}:</span>
                        ${info.element_name} (${(info.accuracy * 100).toFixed(1)}%)
                    </li>
                `;
            }
            summaryHtml += '</ul></div>';
        }

        summaryContainer.innerHTML = summaryHtml;
    }

    // Create heatmap of accuracies
    renderAccuracyHeatmap(container, data.results);
}

/**
 * Render accuracy heatmap using D3.
 */
function renderAccuracyHeatmap(container, results) {
    // Group results by element and dataset
    const elements = [...new Set(results.map(r => r.element_name))].sort();
    const datasets = [...new Set(results.map(r => r.dataset))].sort();

    // Build matrix
    const matrix = [];
    elements.forEach((element, i) => {
        datasets.forEach((dataset, j) => {
            const matching = results.filter(r => r.element_name === element && r.dataset === dataset);
            const avgAccuracy = matching.length > 0
                ? matching.reduce((sum, r) => sum + r.final_accuracy, 0) / matching.length
                : null;
            matrix.push({
                element,
                dataset,
                accuracy: avgAccuracy,
                row: i,
                col: j,
                experiment_id: matching[0]?.experiment_id,
            });
        });
    });

    // Clear container
    container.innerHTML = '';

    // Set up dimensions
    const margin = { top: 80, right: 20, bottom: 20, left: 100 };
    const cellSize = 50;
    const width = datasets.length * cellSize + margin.left + margin.right;
    const height = elements.length * cellSize + margin.top + margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Color scale
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
        .domain([0.5, 1.0]);

    // Draw cells
    g.selectAll('rect')
        .data(matrix.filter(d => d.accuracy !== null))
        .enter()
        .append('rect')
        .attr('x', d => d.col * cellSize)
        .attr('y', d => d.row * cellSize)
        .attr('width', cellSize - 2)
        .attr('height', cellSize - 2)
        .attr('fill', d => colorScale(d.accuracy))
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('click', (event, d) => {
            if (d.experiment_id) {
                loadExperimentBoundary(d.experiment_id);
            }
        })
        .append('title')
        .text(d => `${d.element} on ${d.dataset}: ${(d.accuracy * 100).toFixed(1)}%`);

    // Add accuracy text
    g.selectAll('text.value')
        .data(matrix.filter(d => d.accuracy !== null))
        .enter()
        .append('text')
        .attr('class', 'value')
        .attr('x', d => d.col * cellSize + cellSize / 2)
        .attr('y', d => d.row * cellSize + cellSize / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', d => d.accuracy > 0.75 ? '#fff' : '#000')
        .attr('font-size', '11px')
        .text(d => (d.accuracy * 100).toFixed(0) + '%');

    // Row labels (elements)
    g.selectAll('text.row-label')
        .data(elements)
        .enter()
        .append('text')
        .attr('class', 'row-label')
        .attr('x', -5)
        .attr('y', (d, i) => i * cellSize + cellSize / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '12px')
        .text(d => d);

    // Column labels (datasets)
    g.selectAll('text.col-label')
        .data(datasets)
        .enter()
        .append('text')
        .attr('class', 'col-label')
        .attr('x', (d, i) => i * cellSize + cellSize / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('transform', (d, i) => `rotate(-45, ${i * cellSize + cellSize / 2}, -10)`)
        .text(d => d);
}

/**
 * Load and display decision boundary for a saved experiment.
 */
async function loadExperimentBoundary(experimentId) {
    try {
        // Load experiment details
        const expResponse = await fetch(`/api/bulk/experiments/${experimentId}`);
        const expData = await expResponse.json();

        // Load boundary data
        const boundaryResponse = await fetch(`/api/bulk/experiments/${experimentId}/boundary?resolution=50`);
        const boundaryData = await boundaryResponse.json();

        // Load dataset for data points
        const datasetName = expData.metadata?.dataset_name || boundaryData.dataset;
        const datasetResponse = await fetch(`/api/dataset/${datasetName}`);
        const datasetData = await datasetResponse.json();

        // Render in a modal or dedicated area
        renderExperimentVisualization(expData, boundaryData, datasetData);

    } catch (error) {
        console.error('Failed to load experiment boundary:', error);
        alert('Failed to load experiment: ' + error.message);
    }
}

/**
 * Render experiment visualization with decision boundary.
 */
function renderExperimentVisualization(expData, boundaryData, datasetData) {
    // Store current experiment ID for loading
    const experimentId = expData.experiment_id || boundaryData.experiment_id;

    // Create or show modal
    let modal = document.getElementById('experiment-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'experiment-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="modal-close" onclick="closeExperimentModal()">&times;</span>
                <h3 id="modal-title"></h3>
                <div id="modal-boundary"></div>
                <div id="modal-history"></div>
                <div id="modal-details"></div>
                <div id="modal-actions" class="modal-actions"></div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    modal.style.display = 'block';

    // Title
    document.getElementById('modal-title').textContent =
        `${expData.metadata?.element_name || 'Element'} on ${expData.metadata?.dataset_name || 'Dataset'}`;

    // Render decision boundary
    const boundaryContainer = document.getElementById('modal-boundary');
    boundaryContainer.innerHTML = '';
    renderDecisionBoundary(boundaryContainer, boundaryData, datasetData);

    // Render training history
    if (expData.result?.history) {
        const historyContainer = document.getElementById('modal-history');
        historyContainer.innerHTML = '';
        renderTrainingHistory(historyContainer, expData.result.history);
    }

    // Details
    const detailsContainer = document.getElementById('modal-details');
    detailsContainer.innerHTML = `
        <p><strong>Final Accuracy:</strong> ${((expData.result?.final_accuracy || 0) * 100).toFixed(2)}%</p>
        <p><strong>Final Loss:</strong> ${(expData.result?.final_loss || 0).toFixed(4)}</p>
        <p><strong>Training Time:</strong> ${(expData.result?.training_time_seconds || 0).toFixed(2)}s</p>
    `;

    // Action buttons
    const actionsContainer = document.getElementById('modal-actions');
    actionsContainer.innerHTML = `
        <button class="btn btn-primary" onclick="loadExperimentInTrainingLab('${experimentId}')">
            Open in Training Lab
        </button>
    `;
}

/**
 * Render decision boundary visualization.
 */
function renderDecisionBoundary(container, boundaryData, datasetData) {
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const width = 400;
    const height = 400;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xExtent = boundaryData.x_range || [-3, 3];
    const yExtent = boundaryData.y_range || [-3, 3];

    const xScale = d3.scaleLinear().domain(xExtent).range([0, width]);
    const yScale = d3.scaleLinear().domain(yExtent).range([height, 0]);

    // Color scale for probability
    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
        .domain([1, 0]);

    // Draw heatmap
    const probs = boundaryData.probs;
    const xCoords = boundaryData.x;
    const yCoords = boundaryData.y;

    if (probs && probs.length > 0) {
        const cellWidth = width / probs[0].length;
        const cellHeight = height / probs.length;

        probs.forEach((row, i) => {
            row.forEach((prob, j) => {
                g.append('rect')
                    .attr('x', j * cellWidth)
                    .attr('y', i * cellHeight)
                    .attr('width', cellWidth)
                    .attr('height', cellHeight)
                    .attr('fill', colorScale(prob));
            });
        });
    }

    // Draw data points
    if (datasetData) {
        const points = datasetData.x.map((x, i) => ({
            x: x,
            y: datasetData.y[i],
            label: datasetData.labels[i],
        }));

        g.selectAll('circle')
            .data(points)
            .enter()
            .append('circle')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 4)
            .attr('fill', d => d.label === 1 ? '#2196F3' : '#FF5722')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1);
    }

    // Axes
    g.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale));

    g.append('g')
        .call(d3.axisLeft(yScale));
}

/**
 * Render training history chart.
 */
function renderTrainingHistory(container, history) {
    const margin = { top: 20, right: 60, bottom: 40, left: 60 };
    const width = 400;
    const height = 150;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const epochs = history.loss.map((_, i) => i);

    // Scales
    const xScale = d3.scaleLinear()
        .domain([0, epochs.length - 1])
        .range([0, width]);

    const lossScale = d3.scaleLinear()
        .domain([0, d3.max(history.loss)])
        .range([height, 0]);

    const accScale = d3.scaleLinear()
        .domain([0, 1])
        .range([height, 0]);

    // Loss line
    const lossLine = d3.line()
        .x((d, i) => xScale(i))
        .y(d => lossScale(d));

    g.append('path')
        .datum(history.loss)
        .attr('fill', 'none')
        .attr('stroke', '#FF5722')
        .attr('stroke-width', 2)
        .attr('d', lossLine);

    // Accuracy line
    const accLine = d3.line()
        .x((d, i) => xScale(i))
        .y(d => accScale(d));

    g.append('path')
        .datum(history.accuracy)
        .attr('fill', 'none')
        .attr('stroke', '#4CAF50')
        .attr('stroke-width', 2)
        .attr('d', accLine);

    // Axes
    g.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale).ticks(5));

    g.append('g')
        .call(d3.axisLeft(lossScale).ticks(5))
        .append('text')
        .attr('fill', '#FF5722')
        .attr('transform', 'rotate(-90)')
        .attr('y', -40)
        .attr('x', -height / 2)
        .attr('text-anchor', 'middle')
        .text('Loss');

    g.append('g')
        .attr('transform', `translate(${width},0)`)
        .call(d3.axisRight(accScale).ticks(5))
        .append('text')
        .attr('fill', '#4CAF50')
        .attr('transform', 'rotate(-90)')
        .attr('y', 45)
        .attr('x', -height / 2)
        .attr('text-anchor', 'middle')
        .text('Accuracy');
}

/**
 * Close the experiment modal.
 */
function closeExperimentModal() {
    const modal = document.getElementById('experiment-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

/**
 * Load an experiment into the Training Lab.
 */
async function loadExperimentInTrainingLab(experimentId) {
    try {
        // Fetch experiment data in Training Lab format
        const response = await fetch(`/api/bulk/experiments/${experimentId}/training-lab`);
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Close the experiment modal
        closeExperimentModal();

        // Switch to Training Lab tab
        document.querySelector('[data-tab="training-lab"]').click();

        // Set the controls to match the experiment
        const arch = data.element.architecture;
        document.getElementById('activation-select').value = arch.activation;
        document.getElementById('layers-input').value = arch.hidden_layers.join(',');
        document.getElementById('dataset-select').value = data.dataset;

        // Display the loaded experiment in Training Lab
        displayLoadedExperiment(data);

    } catch (error) {
        console.error('Failed to load experiment in Training Lab:', error);
        alert('Failed to load experiment: ' + error.message);
    }
}

/**
 * Display a loaded experiment in the Training Lab.
 * This shows the final trained state and training history.
 */
function displayLoadedExperiment(data) {
    // Show stats
    const stats = document.getElementById('training-stats');
    stats.classList.remove('hidden');

    // Update stats to show this is a loaded experiment
    const finalIdx = data.loss_history.length - 1;
    document.getElementById('current-epoch').textContent =
        `${data.frames[finalIdx]?.epoch || 'N/A'} (loaded)`;
    document.getElementById('current-loss').textContent =
        data.loss_history[finalIdx]?.toFixed(4) || '-';
    document.getElementById('current-accuracy').textContent =
        ((data.accuracy_history[finalIdx] || 0) * 100).toFixed(1) + '%';

    // Render the final decision boundary
    const boundaryContainer = document.getElementById('decision-boundary-viz');
    boundaryContainer.innerHTML = '';

    if (data.frames.length > 0) {
        renderLoadedBoundary(boundaryContainer, data);
    }

    // Render the training progress
    const progressContainer = document.getElementById('training-progress-viz');
    progressContainer.innerHTML = '';
    renderLoadedProgress(progressContainer, data);

    // Store as current training data for reference
    if (typeof currentTrainingData !== 'undefined') {
        currentTrainingData = data;
    }
}

/**
 * Render decision boundary for a loaded experiment.
 */
function renderLoadedBoundary(container, data) {
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 300;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
        .domain(data.x_range)
        .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
        .domain(data.y_range)
        .range([innerHeight, 0]);

    // Color scale for probabilities
    const colorScale = d3.scaleLinear()
        .domain([0, 0.5, 1])
        .range(['#3498db', '#ecf0f1', '#e74c3c']);

    // Render heatmap from the last frame
    const lastFrame = data.frames[data.frames.length - 1];
    const probs = lastFrame.probs;
    const cellWidth = innerWidth / data.x.length;
    const cellHeight = innerHeight / data.y.length;

    for (let i = 0; i < data.y.length; i++) {
        for (let j = 0; j < data.x.length; j++) {
            const prob = probs[i][j];
            g.append('rect')
                .attr('x', j * cellWidth)
                .attr('y', i * cellHeight)
                .attr('width', cellWidth + 1)
                .attr('height', cellHeight + 1)
                .attr('fill', colorScale(prob))
                .attr('opacity', 0.8);
        }
    }

    // Plot data points
    const points = data.data_points;
    for (let i = 0; i < points.x.length; i++) {
        g.append('circle')
            .attr('cx', xScale(points.x[i]))
            .attr('cy', yScale(points.y[i]))
            .attr('r', 4)
            .attr('fill', points.labels[i] === 1 ? '#e74c3c' : '#3498db')
            .attr('stroke', 'white')
            .attr('stroke-width', 1);
    }

    // Add axes
    g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .attr('color', 'var(--text-muted)');

    g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .attr('color', 'var(--text-muted)');

    // Add label indicating this is a loaded experiment
    g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--primary)')
        .attr('font-size', '12px')
        .text(`Loaded: ${data.element.name} on ${data.dataset}`);
}

/**
 * Render training progress for a loaded experiment.
 */
function renderLoadedProgress(container, data) {
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 300;
    const margin = { top: 20, right: 60, bottom: 30, left: 50 };

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const lossData = data.loss_history;
    const accData = data.accuracy_history;
    const epochs = data.frames.map(f => f.epoch);

    // Scales
    const xScale = d3.scaleLinear()
        .domain([0, epochs[epochs.length - 1] || 1])
        .range([0, innerWidth]);

    const yScaleLoss = d3.scaleLog()
        .domain([Math.max(0.01, d3.min(lossData)), d3.max(lossData)])
        .range([innerHeight, 0]);

    const yScaleAcc = d3.scaleLinear()
        .domain([0, 1])
        .range([innerHeight, 0]);

    // Loss line
    const lossLine = d3.line()
        .x((d, i) => xScale(epochs[i]))
        .y(d => yScaleLoss(Math.max(0.01, d)));

    g.append('path')
        .datum(lossData)
        .attr('fill', 'none')
        .attr('stroke', '#e74c3c')
        .attr('stroke-width', 2)
        .attr('d', lossLine);

    // Accuracy line
    const accLine = d3.line()
        .x((d, i) => xScale(epochs[i]))
        .y(d => yScaleAcc(d));

    g.append('path')
        .datum(accData)
        .attr('fill', 'none')
        .attr('stroke', '#2ecc71')
        .attr('stroke-width', 2)
        .attr('d', accLine);

    // Axes
    g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .attr('color', 'var(--text-muted)');

    g.append('g')
        .call(d3.axisLeft(yScaleLoss).ticks(5))
        .attr('color', '#e74c3c');

    g.append('g')
        .attr('transform', `translate(${innerWidth},0)`)
        .call(d3.axisRight(yScaleAcc).ticks(5))
        .attr('color', '#2ecc71');

    // Labels
    g.append('text')
        .attr('x', -innerHeight / 2)
        .attr('y', -35)
        .attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle')
        .attr('fill', '#e74c3c')
        .attr('font-size', '12px')
        .text('Loss');

    g.append('text')
        .attr('x', innerWidth + innerHeight / 2)
        .attr('y', -35)
        .attr('transform', `rotate(90, ${innerWidth}, 0)`)
        .attr('text-anchor', 'middle')
        .attr('fill', '#2ecc71')
        .attr('font-size', '12px')
        .text('Accuracy');

    g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 25)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--text-muted)')
        .attr('font-size', '12px')
        .text('Epoch');
}

// Close modal on outside click
window.addEventListener('click', (event) => {
    const modal = document.getElementById('experiment-modal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
});

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Only init if experiments tab exists
    if (document.getElementById('experiments')) {
        initExperiments();
    }
});
