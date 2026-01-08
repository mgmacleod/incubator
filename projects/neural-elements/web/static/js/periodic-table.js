/**
 * Periodic Table visualization for Neural Elements
 */

let periodicTableData = null;
let selectedElement = null;

async function loadPeriodicTable() {
    try {
        const response = await fetch('/api/periodic-table');
        periodicTableData = await response.json();
        renderPeriodicTable(periodicTableData);
    } catch (error) {
        console.error('Failed to load periodic table:', error);
        document.getElementById('periodic-table-grid').innerHTML =
            '<p style="color: var(--error);">Failed to load periodic table data.</p>';
    }
}

function renderPeriodicTable(data) {
    const container = document.getElementById('periodic-table-grid');
    const groups = data.groups;
    const periods = data.periods;

    // Set up grid
    const numCols = groups.length + 1;
    const numRows = periods.length + 1;
    container.style.gridTemplateColumns = `80px repeat(${groups.length}, 1fr)`;
    container.style.gridTemplateRows = `repeat(${numRows}, auto)`;

    let html = '';

    // Header row - empty corner + group headers
    html += '<div class="period-header"></div>';
    groups.forEach(group => {
        const info = data.group_info[group] || {};
        html += `<div class="group-header" title="${info.description || ''}">${group.toUpperCase()}</div>`;
    });

    // Data rows
    periods.forEach(period => {
        // Period header
        const periodInfo = data.period_info[period] || {};
        html += `<div class="period-header" title="${periodInfo.description || ''}">${period} Layer${period > 1 ? 's' : ''}</div>`;

        // Elements in this period
        groups.forEach(group => {
            const key = `${period}-${group}`;
            const element = data.elements[key];

            if (element) {
                const family = element.properties?.activation_family || 'unknown';
                html += `
                    <div class="element-cell"
                         data-key="${key}"
                         data-family="${family}"
                         onclick="showElementDetail('${element.name}')">
                        <div class="symbol">${getElementSymbol(element.name)}</div>
                        <div class="name">${element.name}</div>
                        <div class="params">${element.params} params</div>
                    </div>
                `;
            } else {
                html += '<div class="element-cell empty" style="opacity: 0.3; cursor: default;"></div>';
            }
        });
    });

    container.innerHTML = html;
}

function getElementSymbol(name) {
    // Extract a short symbol from the element name
    // e.g., "REL-2x4" -> "Re" or "TAH-1x4" -> "Ta"
    const parts = name.split('-');
    if (parts.length > 0) {
        const prefix = parts[0];
        return prefix.charAt(0).toUpperCase() + prefix.charAt(1).toLowerCase();
    }
    return name.substring(0, 2);
}

async function showElementDetail(elementName) {
    try {
        const response = await fetch(`/api/element/${elementName}`);
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        selectedElement = data;
        displayElementDetail(data);

        // Load experiment history for this element
        loadElementExperiments(elementName);

        // Show overlay
        showOverlay();
    } catch (error) {
        console.error('Failed to load element details:', error);
    }
}

function displayElementDetail(data) {
    const detail = document.getElementById('element-detail');

    // Update name
    document.getElementById('element-name').textContent = data.name;

    // Update architecture
    const arch = data.architecture;
    const layers = [arch.input_dim, ...arch.hidden_layers, arch.output_dim];
    document.getElementById('element-architecture').textContent =
        layers.join(' → ') + ` (${arch.activation})`;

    // Update properties
    const props = data.properties;
    const propsList = document.getElementById('element-properties');
    propsList.innerHTML = `
        <li>Total Parameters: ${props.total_params}</li>
        <li>Depth: ${props.depth} hidden layer${props.depth > 1 ? 's' : ''}</li>
        <li>Width Pattern: ${props.width_pattern}</li>
    `;

    // Update activation info
    const actInfo = data.activation_info;
    document.getElementById('element-activation').innerHTML = `
        <strong>${actInfo.name}</strong> (${actInfo.family} family)<br>
        <small>${actInfo.description || ''}</small>
    `;

    // Show preview placeholder
    document.getElementById('element-preview').innerHTML =
        '<p style="color: var(--text-muted);">Click "Train in Lab" to see this element in action</p>';

    detail.classList.remove('hidden');
}

function closeElementDetail() {
    document.getElementById('element-detail').classList.add('hidden');
    hideOverlay();
}

function openInTrainingLab() {
    if (!selectedElement) return;

    // Set training lab controls to match selected element
    const arch = selectedElement.architecture;
    document.getElementById('activation-select').value = arch.activation;
    document.getElementById('layers-input').value = arch.hidden_layers.join(',');

    // Close detail and switch to training lab
    closeElementDetail();
    document.querySelector('[data-tab="training-lab"]').click();
}

function showOverlay() {
    let overlay = document.querySelector('.overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'overlay';
        overlay.onclick = closeElementDetail;
        document.body.appendChild(overlay);
    }
    overlay.classList.remove('hidden');
}

function hideOverlay() {
    const overlay = document.querySelector('.overlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

/**
 * Load experiment history for an element.
 */
async function loadElementExperiments(elementName) {
    const previewContainer = document.getElementById('element-preview');

    try {
        const response = await fetch(`/api/bulk/experiments/by-element/${elementName}`);
        const data = await response.json();

        if (data.experiments && data.experiments.length > 0) {
            displayElementExperiments(data.experiments, previewContainer);
        } else {
            previewContainer.innerHTML = `
                <p style="color: var(--text-muted);">No saved experiments for this element.</p>
                <p style="color: var(--text-muted); font-size: 0.9rem;">Click "Train in Lab" or run bulk experiments to create some.</p>
            `;
        }
    } catch (error) {
        console.error('Failed to load element experiments:', error);
        previewContainer.innerHTML = `
            <p style="color: var(--text-muted);">Click "Train in Lab" to see this element in action</p>
        `;
    }
}

/**
 * Display experiment history in the element detail view.
 */
function displayElementExperiments(experiments, container) {
    // Group by dataset
    const byDataset = {};
    experiments.forEach(exp => {
        const dataset = exp.dataset || 'unknown';
        if (!byDataset[dataset]) {
            byDataset[dataset] = [];
        }
        byDataset[dataset].push(exp);
    });

    let html = '<div class="element-experiments">';
    html += '<h4 style="margin-bottom: 0.75rem; color: var(--text-muted);">Saved Experiments</h4>';

    // Show best result per dataset
    html += '<div class="experiment-list">';
    for (const [dataset, exps] of Object.entries(byDataset)) {
        const best = exps[0]; // Already sorted by accuracy
        html += `
            <div class="experiment-item" onclick="loadExperimentFromTable('${best.experiment_id}')" style="cursor: pointer;">
                <span class="dataset-name">${dataset}</span>
                <span class="accuracy">${((best.final_accuracy || 0) * 100).toFixed(1)}%</span>
            </div>
        `;
    }
    html += '</div>';

    if (experiments.length > Object.keys(byDataset).length) {
        html += `<p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">
            ${experiments.length} total experiments
        </p>`;
    }

    html += '</div>';
    container.innerHTML = html;
}

/**
 * Load an experiment from the periodic table detail view.
 */
async function loadExperimentFromTable(experimentId) {
    // Close the element detail
    closeElementDetail();

    // Use the experiments.js function if available
    if (typeof loadExperimentInTrainingLab === 'function') {
        loadExperimentInTrainingLab(experimentId);
    } else {
        // Fallback: switch to Training Lab and show message
        document.querySelector('[data-tab="training-lab"]').click();
        alert('Experiment loading not available. Please check the Experiments tab.');
    }
}
