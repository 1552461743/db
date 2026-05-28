(() => {
  'use strict';

  const socket = io();
  const IMU_CHANNELS = 8;
  const CAP_CHANNELS = 10;

  const emptyAxes = () => ({ x: null, y: null, z: null });

  const state = {
    currentChannel: 0,
    currentMainView: 'imu',
    currentImuMode: 'raw',
    maxDataPoints: 240,
    latestAcc: Array.from({ length: IMU_CHANNELS }, emptyAxes),
    latestGyro: Array.from({ length: IMU_CHANNELS }, emptyAxes),
    latestMag: Array.from({ length: IMU_CHANNELS }, emptyAxes),
  };

  const dom = {
    status: document.getElementById('status'),
    mainViewButtons: Array.from(document.querySelectorAll('[data-main-view]')),
    imuButtons: Array.from(document.querySelectorAll('[data-imu-channel]')),
    imuModeButtons: Array.from(document.querySelectorAll('[data-imu-mode]')),
    imuSection: document.getElementById('imuSection'),
    capSection: document.getElementById('capSection'),
    accelPanel: document.getElementById('accelPanel'),
    gyroPanel: document.getElementById('gyroPanel'),
    magPanel: document.getElementById('magPanel'),
    accelTitle: document.getElementById('accelTitle'),
    gyroTitle: document.getElementById('gyroTitle'),
    magTitle: document.getElementById('magTitle'),
    capLastUpdate: document.getElementById('capLastUpdate'),
    capValueCells: Array.from({ length: CAP_CHANNELS }, (_, i) => document.getElementById(`cap-val-${i}`)),
    imuRows: Array.from({ length: IMU_CHANNELS }, (_, i) => ({
      acc: {
        x: document.getElementById(`imu-${i}-acc-x`),
        y: document.getElementById(`imu-${i}-acc-y`),
        z: document.getElementById(`imu-${i}-acc-z`),
      },
      gyro: {
        x: document.getElementById(`imu-${i}-gyro-x`),
        y: document.getElementById(`imu-${i}-gyro-y`),
        z: document.getElementById(`imu-${i}-gyro-z`),
      },
      mag: {
        x: document.getElementById(`imu-${i}-mag-x`),
        y: document.getElementById(`imu-${i}-mag-y`),
        z: document.getElementById(`imu-${i}-mag-z`),
      },
    })),
  };

  const axisStatCells = {
    accel: {
      x: document.getElementById('accel-x'),
      y: document.getElementById('accel-y'),
      z: document.getElementById('accel-z'),
    },
    gyro: {
      x: document.getElementById('gyro-x'),
      y: document.getElementById('gyro-y'),
      z: document.getElementById('gyro-z'),
    },
    mag: {
      x: document.getElementById('mag-x'),
      y: document.getElementById('mag-y'),
      z: document.getElementById('mag-z'),
    },
  };

  function setConnected(isConnected) {
    if (isConnected) {
      dom.status.innerHTML = '<span class="status-dot"></span><span>已连接 ✓</span>';
      dom.status.className = 'status connected';
      return;
    }
    dom.status.innerHTML = '<span class="status-dot"></span><span>连接断开 ✗</span>';
    dom.status.className = 'status disconnected';
  }

  function setActiveButton(buttons, predicate) {
    buttons.forEach((btn) => {
      btn.classList.toggle('active', predicate(btn));
    });
  }

  function formatValue(value) {
    return Number.isFinite(value) ? value.toFixed(3) : '--';
  }

  function setText(node, value) {
    if (node) {
      node.textContent = value;
    }
  }

  function createAxesChart(canvasId, labels) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: labels[0],
            data: [],
            borderColor: '#ff6384',
            backgroundColor: 'rgba(255, 99, 132, 0.12)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35,
          },
          {
            label: labels[1],
            data: [],
            borderColor: '#36a2eb',
            backgroundColor: 'rgba(54, 162, 235, 0.12)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35,
          },
          {
            label: labels[2],
            data: [],
            borderColor: '#ffce56',
            backgroundColor: 'rgba(255, 206, 86, 0.12)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2.5,
        animation: false,
        interaction: { intersect: false, mode: 'index' },
        scales: {
          x: {
            display: false,
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            ticks: { color: '#fff' },
          },
        },
        plugins: {
          legend: {
            labels: { color: '#fff' },
          },
        },
      },
    });
  }

  function createCapacitanceChart() {
    const colors = ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff', '#ff9f40', '#c9cbcf', '#4dff4d', '#ff4da6', '#4dc3ff'];
    const ctx = document.getElementById('chartCapacitance').getContext('2d');
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: Array.from({ length: CAP_CHANNELS }, (_, i) => ({
          label: `通道 ${i}`,
          data: [],
          borderColor: colors[i],
          backgroundColor: `${colors[i]}22`,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 3.0,
        animation: false,
        interaction: { intersect: false, mode: 'index' },
        scales: {
          x: {
            display: false,
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            ticks: { color: '#fff' },
          },
        },
        plugins: {
          legend: {
            labels: { color: '#fff' },
          },
        },
      },
    });
  }

  const accelChart = createAxesChart('chartAccel', ['X', 'Y', 'Z']);
  const gyroChart = createAxesChart('chartGyro', ['X', 'Y', 'Z']);
  const magChart = createAxesChart('chartMag', ['X', 'Y', 'Z']);
  const capChart = createCapacitanceChart();

  function recomputeYAxis(chart, fallbackPadding) {
    const values = [];
    chart.data.datasets.forEach((ds) => {
      ds.data.forEach((value) => {
        if (Number.isFinite(value)) {
          values.push(value);
        }
      });
    });

    if (values.length === 0) {
      delete chart.options.scales.y.min;
      delete chart.options.scales.y.max;
      return;
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = (max - min) * 0.1 || fallbackPadding;
    chart.options.scales.y.min = min - padding;
    chart.options.scales.y.max = max + padding;
  }

  function clearAxisStats(groupName) {
    const group = axisStatCells[groupName];
    setText(group.x, '--');
    setText(group.y, '--');
    setText(group.z, '--');
  }

  function clearAxesChart(chart, groupName) {
    chart.data.labels = [];
    chart.data.datasets.forEach((ds) => {
      ds.data = [];
    });
    delete chart.options.scales.y.min;
    delete chart.options.scales.y.max;
    chart.update('none');
    clearAxisStats(groupName);
  }

  function clearCapChart() {
    capChart.data.labels = [];
    capChart.data.datasets.forEach((ds) => {
      ds.data = [];
    });
    delete capChart.options.scales.y.min;
    delete capChart.options.scales.y.max;
    capChart.update('none');
  }

  function appendAxesPoint(chart, values, groupName) {
    chart.data.labels.push('');
    chart.data.datasets[0].data.push(Number(values.x));
    chart.data.datasets[1].data.push(Number(values.y));
    chart.data.datasets[2].data.push(Number(values.z));

    if (chart.data.labels.length > state.maxDataPoints) {
      chart.data.labels.shift();
      chart.data.datasets.forEach((ds) => ds.data.shift());
    }

    recomputeYAxis(chart, 1);
    chart.update('none');

    setText(axisStatCells[groupName].x, formatValue(values.x));
    setText(axisStatCells[groupName].y, formatValue(values.y));
    setText(axisStatCells[groupName].z, formatValue(values.z));
  }

  function appendCapacitance(values, timestampSec) {
    capChart.data.labels.push('');
    for (let i = 0; i < CAP_CHANNELS; i++) {
      capChart.data.datasets[i].data.push(Number(values[i]));
    }

    if (capChart.data.labels.length > state.maxDataPoints) {
      capChart.data.labels.shift();
      capChart.data.datasets.forEach((ds) => ds.data.shift());
    }

    recomputeYAxis(capChart, 10);
    capChart.update('none');

    for (let i = 0; i < CAP_CHANNELS; i++) {
      setText(dom.capValueCells[i], formatValue(Number(values[i])));
    }

    if (typeof timestampSec === 'number') {
      setText(dom.capLastUpdate, new Date(timestampSec * 1000).toLocaleTimeString());
    }
  }

  function updateImuTable(kind, channels) {
    const target = kind === 'acc' ? state.latestAcc : kind === 'gyro' ? state.latestGyro : state.latestMag;
    for (let i = 0; i < IMU_CHANNELS; i++) {
      const row = channels[i];
      if (!Array.isArray(row) || row.length < 3) {
        continue;
      }

      target[i] = {
        x: Number(row[0]),
        y: Number(row[1]),
        z: Number(row[2]),
      };

      const cells = dom.imuRows[i][kind];
      setText(cells.x, formatValue(target[i].x));
      setText(cells.y, formatValue(target[i].y));
      setText(cells.z, formatValue(target[i].z));
    }
  }

  function updateMainViewUI() {
    dom.imuSection.classList.toggle('hidden', state.currentMainView !== 'imu');
    dom.capSection.classList.toggle('hidden', state.currentMainView !== 'cap');
    setActiveButton(dom.mainViewButtons, (btn) => btn.dataset.mainView === state.currentMainView);
  }

  function updateImuModeUI() {
    const isRaw = state.currentImuMode === 'raw';
    dom.accelPanel.classList.toggle('hidden', !isRaw);
    dom.gyroPanel.classList.toggle('hidden', !isRaw);
    dom.magPanel.classList.toggle('hidden', isRaw);
    dom.accelTitle.textContent = `IMU ${state.currentChannel} 原始加速度`;
    dom.gyroTitle.textContent = `IMU ${state.currentChannel} 原始角速度`;
    dom.magTitle.textContent = `IMU ${state.currentChannel} 原始磁场`;
    setActiveButton(dom.imuModeButtons, (btn) => btn.dataset.imuMode === state.currentImuMode);
  }

  function resetImuCharts() {
    clearAxesChart(accelChart, 'accel');
    clearAxesChart(gyroChart, 'gyro');
    clearAxesChart(magChart, 'mag');
  }

  dom.mainViewButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.mainView;
      if (!view || view === state.currentMainView) {
        return;
      }
      state.currentMainView = view;
      updateMainViewUI();
    });
  });

  dom.imuButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const channel = Number(btn.dataset.imuChannel);
      if (!Number.isFinite(channel) || channel === state.currentChannel) {
        return;
      }
      state.currentChannel = channel;
      setActiveButton(dom.imuButtons, (node) => Number(node.dataset.imuChannel) === channel);
      resetImuCharts();
      updateImuModeUI();
    });
  });

  dom.imuModeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.imuMode;
      if (!mode || mode === state.currentImuMode) {
        return;
      }
      state.currentImuMode = mode;
      resetImuCharts();
      updateImuModeUI();
    });
  });

  socket.on('connect', () => setConnected(true));
  socket.on('disconnect', () => setConnected(false));

  socket.on('group_data', (data) => {
    if (!data || !Array.isArray(data.channels) || data.channels.length < IMU_CHANNELS) {
      return;
    }

    updateImuTable(data.kind, data.channels);

    const row = data.channels[state.currentChannel];
    if (!Array.isArray(row) || row.length < 3) {
      return;
    }

    const values = {
      x: Number(row[0]),
      y: Number(row[1]),
      z: Number(row[2]),
    };

    if (state.currentImuMode === 'raw') {
      if (data.kind === 'acc') {
        appendAxesPoint(accelChart, values, 'accel');
      } else if (data.kind === 'gyro') {
        appendAxesPoint(gyroChart, values, 'gyro');
      }
      return;
    }

    if (state.currentImuMode === 'mag' && data.kind === 'mag') {
      appendAxesPoint(magChart, values, 'mag');
    }
  });

  socket.on('cap_data', (data) => {
    if (!data || !Array.isArray(data.values) || data.values.length < CAP_CHANNELS) {
      return;
    }
    appendCapacitance(data.values, data.timestamp);
  });

  socket.on('server_meta', () => {});

  setConnected(false);
  updateMainViewUI();
  updateImuModeUI();
  clearCapChart();
})();
