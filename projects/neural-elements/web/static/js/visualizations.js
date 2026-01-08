/**
 * General visualization utilities for Neural Elements
 */

// Color schemes
const colors = {
    primary: '#3498db',
    secondary: '#2ecc71',
    accent: '#e74c3c',
    muted: '#95a5a6',
    background: '#1a1a2e',
    surface: '#16213e',
};

// Activation family colors
const familyColors = {
    linear: '#95a5a6',
    rectified: '#3498db',
    smooth: '#2ecc71',
    periodic: '#9b59b6',
};

/**
 * Create a simple line chart
 */
function createLineChart(container, data, options = {}) {
    const width = options.width || container.clientWidth || 400;
    const height = options.height || container.clientHeight || 200;
    const margin = options.margin || { top: 20, right: 20, bottom: 30, left: 50 };

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
        .domain([0, data.length - 1])
        .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
        .domain([d3.min(data), d3.max(data)])
        .range([innerHeight, 0]);

    // Line
    const line = d3.line()
        .x((d, i) => xScale(i))
        .y(d => yScale(d));

    g.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', options.color || colors.primary)
        .attr('stroke-width', 2)
        .attr('d', line);

    // Axes
    g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .attr('color', colors.muted);

    g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .attr('color', colors.muted);

    return svg;
}

/**
 * Create a heatmap visualization
 */
function createHeatmap(container, data, options = {}) {
    const width = options.width || container.clientWidth || 400;
    const height = options.height || container.clientHeight || 400;
    const margin = options.margin || { top: 20, right: 20, bottom: 30, left: 40 };

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const rows = data.length;
    const cols = data[0].length;

    const cellWidth = innerWidth / cols;
    const cellHeight = innerHeight / rows;

    const colorScale = options.colorScale || d3.scaleLinear()
        .domain([0, 0.5, 1])
        .range(['#3498db', '#ecf0f1', '#e74c3c']);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            g.append('rect')
                .attr('x', j * cellWidth)
                .attr('y', i * cellHeight)
                .attr('width', cellWidth)
                .attr('height', cellHeight)
                .attr('fill', colorScale(data[i][j]));
        }
    }

    return svg;
}

/**
 * Create a scatter plot
 */
function createScatterPlot(container, points, options = {}) {
    const width = options.width || container.clientWidth || 400;
    const height = options.height || container.clientHeight || 400;
    const margin = options.margin || { top: 20, right: 20, bottom: 30, left: 40 };

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xExtent = d3.extent(points, d => d.x);
    const yExtent = d3.extent(points, d => d.y);

    const xScale = d3.scaleLinear()
        .domain([xExtent[0] - 0.5, xExtent[1] + 0.5])
        .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
        .domain([yExtent[0] - 0.5, yExtent[1] + 0.5])
        .range([innerHeight, 0]);

    // Points
    g.selectAll('circle')
        .data(points)
        .enter()
        .append('circle')
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('r', options.radius || 5)
        .attr('fill', d => d.color || colors.primary)
        .attr('stroke', 'white')
        .attr('stroke-width', 1);

    // Axes
    g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .attr('color', colors.muted);

    g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .attr('color', colors.muted);

    return svg;
}

/**
 * Render an activation function visualization
 */
function renderActivationViz(container, activationName) {
    const width = 200;
    const height = 150;
    const margin = { top: 10, right: 10, bottom: 20, left: 30 };

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    container.innerHTML = '';

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Generate activation function data
    const x = d3.range(-3, 3.1, 0.1);
    const y = x.map(v => activationFunction(v, activationName));

    const xScale = d3.scaleLinear()
        .domain([-3, 3])
        .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
        .domain([Math.min(-1, d3.min(y)), Math.max(1, d3.max(y))])
        .range([innerHeight, 0]);

    const line = d3.line()
        .x((d, i) => xScale(x[i]))
        .y(d => yScale(d));

    // Zero lines
    g.append('line')
        .attr('x1', 0)
        .attr('x2', innerWidth)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .attr('stroke', colors.muted)
        .attr('stroke-dasharray', '3,3');

    g.append('line')
        .attr('x1', xScale(0))
        .attr('x2', xScale(0))
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .attr('stroke', colors.muted)
        .attr('stroke-dasharray', '3,3');

    // Function line
    g.append('path')
        .datum(y)
        .attr('fill', 'none')
        .attr('stroke', colors.primary)
        .attr('stroke-width', 2)
        .attr('d', line);

    // Axes
    g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(3))
        .attr('color', colors.muted);

    g.append('g')
        .call(d3.axisLeft(yScale).ticks(3))
        .attr('color', colors.muted);
}

/**
 * Simple activation function implementations for visualization
 */
function activationFunction(x, name) {
    switch (name) {
        case 'relu':
            return Math.max(0, x);
        case 'tanh':
            return Math.tanh(x);
        case 'sigmoid':
            return 1 / (1 + Math.exp(-x));
        case 'sine':
            return Math.sin(x);
        case 'linear':
            return x;
        case 'gelu':
            return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
        case 'swish':
            return x / (1 + Math.exp(-x));
        default:
            return x;
    }
}

/**
 * Format number for display
 */
function formatNumber(num, decimals = 2) {
    if (Math.abs(num) < 0.001) return num.toExponential(decimals);
    return num.toFixed(decimals);
}

/**
 * Debounce function for resize handlers
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
