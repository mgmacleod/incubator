/**
 * Training Lab functionality for Neural Elements
 */

let currentTrainingData = null;
let animationFrameIndex = 0;
let animationTimer = null;
let isTraining = false;

function initTrainingLab() {
    // Set up epoch slider
    const epochsSlider = document.getElementById('epochs-slider');
    const epochsValue = document.getElementById('epochs-value');
    epochsSlider.addEventListener('input', () => {
        epochsValue.textContent = epochsSlider.value;
    });

    // Initialize empty visualizations
    initEmptyViz();
}

function initEmptyViz() {
    const boundaryViz = document.getElementById('decision-boundary-viz');
    const progressViz = document.getElementById('training-progress-viz');

    boundaryViz.innerHTML = '<p style="color: var(--text-muted);">Configure an element and click "Start Training"</p>';
    progressViz.innerHTML = '<p style="color: var(--text-muted);">Training progress will appear here</p>';
}

async function startTraining() {
    if (isTraining) return;

    const trainBtn = document.getElementById('train-btn');
    const stats = document.getElementById('training-stats');

    // Parse hidden layers
    const layersInput = document.getElementById('layers-input').value;
    const hiddenLayers = layersInput.split(',')
        .map(s => parseInt(s.trim()))
        .filter(n => !isNaN(n) && n > 0);

    if (hiddenLayers.length === 0) {
        alert('Please enter valid hidden layer widths (e.g., "4,4" or "8")');
        return;
    }

    const config = {
        dataset: document.getElementById('dataset-select').value,
        activation: document.getElementById('activation-select').value,
        hidden_layers: hiddenLayers,
        epochs: parseInt(document.getElementById('epochs-slider').value),
        record_every: 5,
    };

    // Update UI
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<span class="loading"></span> Training...';
    stats.classList.remove('hidden');
    isTraining = true;

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });

        currentTrainingData = await response.json();

        if (currentTrainingData.error) {
            throw new Error(currentTrainingData.error);
        }

        // Start animation
        animationFrameIndex = 0;
        animateTraining();

    } catch (error) {
        console.error('Training failed:', error);
        alert('Training failed: ' + error.message);
    } finally {
        trainBtn.disabled = false;
        trainBtn.innerHTML = 'Start Training';
        isTraining = false;
    }
}

function animateTraining() {
    if (!currentTrainingData || animationFrameIndex >= currentTrainingData.frames.length) {
        // Animation complete
        updateFinalStats();
        return;
    }

    const frame = currentTrainingData.frames[animationFrameIndex];

    // Update decision boundary visualization
    renderDecisionBoundary(frame, currentTrainingData);

    // Update progress visualization
    renderProgress(animationFrameIndex, currentTrainingData);

    // Update stats
    updateStats(animationFrameIndex, currentTrainingData);

    animationFrameIndex++;

    // Schedule next frame
    animationTimer = setTimeout(animateTraining, 50);
}

function updateStats(frameIndex, data) {
    document.getElementById('current-epoch').textContent =
        data.frames[frameIndex].epoch;
    document.getElementById('current-loss').textContent =
        data.loss_history[frameIndex].toFixed(4);
    document.getElementById('current-accuracy').textContent =
        (data.accuracy_history[frameIndex] * 100).toFixed(1) + '%';
}

function updateFinalStats() {
    const data = currentTrainingData;
    const lastIdx = data.loss_history.length - 1;

    document.getElementById('current-epoch').textContent =
        data.frames[lastIdx].epoch + ' (done)';
    document.getElementById('current-loss').textContent =
        data.loss_history[lastIdx].toFixed(4);
    document.getElementById('current-accuracy').textContent =
        (data.accuracy_history[lastIdx] * 100).toFixed(1) + '%';
}

function renderDecisionBoundary(frame, data) {
    const container = document.getElementById('decision-boundary-viz');
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 300;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };

    // Clear previous content
    container.innerHTML = '';

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

    // Render heatmap
    const cellWidth = innerWidth / data.x.length;
    const cellHeight = innerHeight / data.y.length;

    for (let i = 0; i < data.y.length; i++) {
        for (let j = 0; j < data.x.length; j++) {
            const prob = frame.probs[i][j];
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
}

function renderProgress(currentFrame, data) {
    const container = document.getElementById('training-progress-viz');
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 300;
    const margin = { top: 20, right: 60, bottom: 30, left: 50 };

    container.innerHTML = '';

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Prepare data up to current frame
    const lossData = data.loss_history.slice(0, currentFrame + 1);
    const accData = data.accuracy_history.slice(0, currentFrame + 1);
    const epochs = data.frames.slice(0, currentFrame + 1).map(f => f.epoch);

    // Scales
    const xScale = d3.scaleLinear()
        .domain([0, data.frames[data.frames.length - 1].epoch])
        .range([0, innerWidth]);

    const yScaleLoss = d3.scaleLog()
        .domain([Math.max(0.01, d3.min(data.loss_history)), d3.max(data.loss_history)])
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

// Comparison functionality
async function runComparison() {
    const dataset = document.getElementById('compare-dataset').value;
    const container = document.getElementById('comparison-results');

    container.innerHTML = '<p style="text-align: center;"><span class="loading"></span> Running comparison...</p>';

    const elements = [
        { hidden_layers: [4], activation: 'relu' },
        { hidden_layers: [4, 4], activation: 'relu' },
        { hidden_layers: [4, 4, 4], activation: 'relu' },
        { hidden_layers: [4], activation: 'tanh' },
        { hidden_layers: [4, 4], activation: 'tanh' },
        { hidden_layers: [8], activation: 'relu' },
    ];

    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset, elements, epochs: 500 }),
        });

        const data = await response.json();
        renderComparison(data);
    } catch (error) {
        console.error('Comparison failed:', error);
        container.innerHTML = '<p style="color: var(--error);">Comparison failed.</p>';
    }
}

function renderComparison(data) {
    const container = document.getElementById('comparison-results');
    container.innerHTML = '';

    data.results.forEach((result, index) => {
        const card = document.createElement('div');
        card.className = 'comparison-card';

        const element = result.element;
        const arch = element.architecture;
        const layers = [arch.input_dim, ...arch.hidden_layers, arch.output_dim];

        card.innerHTML = `
            <h4>${element.name}</h4>
            <div class="stats">
                <span>Arch: ${layers.join('→')}</span>
                <span>Acc: ${(element.training?.final_accuracy * 100 || 0).toFixed(1)}%</span>
            </div>
            <div class="comparison-viz" id="comparison-viz-${index}"></div>
        `;

        container.appendChild(card);

        // Render decision boundary
        setTimeout(() => {
            renderComparisonBoundary(`comparison-viz-${index}`, result.boundary, data.dataset);
        }, 100);
    });
}

function renderComparisonBoundary(containerId, boundary, dataset) {
    const container = document.getElementById(containerId);
    const width = container.clientWidth || 250;
    const height = 200;
    const margin = { top: 10, right: 10, bottom: 10, left: 10 };

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
        .domain(boundary.x_range)
        .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
        .domain(boundary.y_range)
        .range([innerHeight, 0]);

    const colorScale = d3.scaleLinear()
        .domain([0, 0.5, 1])
        .range(['#3498db', '#ecf0f1', '#e74c3c']);

    // Render heatmap
    const cellWidth = innerWidth / boundary.x.length;
    const cellHeight = innerHeight / boundary.y.length;

    for (let i = 0; i < boundary.y.length; i++) {
        for (let j = 0; j < boundary.x.length; j++) {
            const prob = boundary.z[i][j];
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
    for (let i = 0; i < dataset.x.length; i++) {
        g.append('circle')
            .attr('cx', xScale(dataset.x[i]))
            .attr('cy', yScale(dataset.y[i]))
            .attr('r', 3)
            .attr('fill', dataset.labels[i] === 1 ? '#e74c3c' : '#3498db')
            .attr('stroke', 'white')
            .attr('stroke-width', 0.5);
    }
}
