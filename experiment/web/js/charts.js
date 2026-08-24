const THEME = {
  colors: DATA.methodColors,
  textColor: '#94a3b8',
  axisLineColor: 'rgba(255,255,255,0.06)',
  splitLineColor: 'rgba(255,255,255,0.04)',
  tooltipBg: 'rgba(15,23,42,0.95)',
  tooltipBorder: 'rgba(255,255,255,0.1)'
};

function baseOption() {
  return {
    backgroundColor: 'transparent',
    textStyle: { color: THEME.textColor, fontFamily: 'Inter' },
    tooltip: {
      backgroundColor: THEME.tooltipBg,
      borderColor: THEME.tooltipBorder,
      textStyle: { color: '#e2e8f0', fontSize: 12 }
    },
    grid: { left: 60, right: 30, top: 50, bottom: 40 }
  };
}

function initCharts() {
  renderExp1Similarity();
  renderExp1Category();
  renderExp2Radar();
  renderExp2DropRate();
  renderExp3MRR();
  renderExp3Recall();
}

function renderExp1Similarity() {
  const el = document.getElementById('chart-exp1-similarity');
  if (!el) return;
  const chart = echarts.init(el);

  const option = {
    ...baseOption(),
    title: { text: '与真实查询的平均余弦相似度', left: 'center', top: 10, textStyle: { color: '#e2e8f0', fontSize: 14, fontWeight: 500 } },
    xAxis: {
      type: 'category',
      data: DATA.methods,
      axisLine: { lineStyle: { color: THEME.axisLineColor } },
      axisLabel: { color: THEME.textColor, fontWeight: 600 }
    },
    yAxis: {
      type: 'value',
      min: 0.5,
      max: 0.85,
      splitLine: { lineStyle: { color: THEME.splitLineColor } },
      axisLabel: { color: THEME.textColor }
    },
    series: [{
      type: 'bar',
      data: [
        { value: DATA.exp1.similarity.q2q, itemStyle: { color: THEME.colors[0] } },
        { value: DATA.exp1.similarity.q2p, itemStyle: { color: THEME.colors[1] } },
        { value: DATA.exp1.similarity.q2n, itemStyle: { color: THEME.colors[2] } },
        { value: DATA.exp1.similarity.q2r, itemStyle: { color: THEME.colors[3] } },
        { value: DATA.exp1.similarity.q2c, itemStyle: { color: THEME.colors[4] } }
      ],
      barWidth: '50%',
      label: { show: true, position: 'top', color: '#e2e8f0', fontFamily: 'JetBrains Mono', fontSize: 12,
        formatter: p => p.value.toFixed(3) }
    }]
  };
  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
}

function renderExp1Category() {
  const el = document.getElementById('chart-exp1-category');
  if (!el) return;
  const chart = echarts.init(el);

  const option = {
    ...baseOption(),
    title: { text: 'Q2Q 与 Q2C 各类别差距对比', left: 'center', top: 10, textStyle: { color: '#e2e8f0', fontSize: 14, fontWeight: 500 } },
    legend: { data: ['Q2Q', 'Q2C', '差距'], top: 35, textStyle: { color: THEME.textColor } },
    grid: { top: 70, bottom: 40 },
    xAxis: {
      type: 'category',
      data: DATA.exp1.byCategory.labels,
      axisLine: { lineStyle: { color: THEME.axisLineColor } },
      axisLabel: { color: THEME.textColor, fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 0.85,
      splitLine: { lineStyle: { color: THEME.splitLineColor } },
      axisLabel: { color: THEME.textColor }
    },
    series: [
      { name: 'Q2Q', type: 'bar', data: DATA.exp1.byCategory.q2qMean, itemStyle: { color: THEME.colors[0] }, barGap: '10%' },
      { name: 'Q2C', type: 'bar', data: DATA.exp1.byCategory.q2cMean, itemStyle: { color: THEME.colors[4] } },
      { name: '差距', type: 'line', data: DATA.exp1.byCategory.gap, yAxisIndex: 0,
        lineStyle: { color: '#10b981', width: 2 }, itemStyle: { color: '#10b981' },
        symbol: 'circle', symbolSize: 8 }
    ]
  };
  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
}

function renderExp2Radar() {
  const el = document.getElementById('chart-exp2-radar');
  if (!el) return;
  const chart = echarts.init(el);

  const option = {
    ...baseOption(),
    title: { text: '各方法鲁棒性得分', left: 'center', top: 10, textStyle: { color: '#e2e8f0', fontSize: 14, fontWeight: 500 } },
    legend: { data: DATA.methods, bottom: 10, textStyle: { color: THEME.textColor } },
    radar: {
      indicator: DATA.exp2.perStyle.styles.map(s => ({ name: s, max: 1.0 })),
      shape: 'polygon',
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
      splitArea: { show: false },
      axisName: { color: THEME.textColor, fontSize: 11 }
    },
    series: [{
      type: 'radar',
      data: [
        { value: DATA.exp2.perStyle.dropQ2Q.map(d => 1 - d), name: 'Q2Q', lineStyle: { color: THEME.colors[0] }, itemStyle: { color: THEME.colors[0] }, areaStyle: { color: 'rgba(59,130,246,0.1)' } },
        { value: DATA.exp2.perStyle.dropQ2P.map(d => 1 - d), name: 'Q2P', lineStyle: { color: THEME.colors[1] }, itemStyle: { color: THEME.colors[1] }, areaStyle: { color: 'rgba(16,185,129,0.05)' } },
        { value: DATA.exp2.perStyle.dropQ2N.map(d => 1 - d), name: 'Q2N', lineStyle: { color: THEME.colors[2] }, itemStyle: { color: THEME.colors[2] }, areaStyle: { color: 'rgba(245,158,11,0.05)' } },
        { value: DATA.exp2.perStyle.dropQ2R.map(d => 1 - d), name: 'Q2R', lineStyle: { color: THEME.colors[3] }, itemStyle: { color: THEME.colors[3] }, areaStyle: { color: 'rgba(239,68,68,0.05)' } },
        { value: DATA.exp2.perStyle.dropQ2C.map(d => 1 - d), name: 'Q2C', lineStyle: { color: THEME.colors[4] }, itemStyle: { color: THEME.colors[4] }, areaStyle: { color: 'rgba(139,92,246,0.05)' } }
      ]
    }]
  };
  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
}

function renderExp2DropRate() {
  const el = document.getElementById('chart-exp2-drop');
  if (!el) return;
  const chart = echarts.init(el);

  const option = {
    ...baseOption(),
    title: { text: '各改写风格的相似度下降率', left: 'center', top: 10, textStyle: { color: '#e2e8f0', fontSize: 14, fontWeight: 500 } },
    legend: { data: DATA.methods, top: 35, textStyle: { color: THEME.textColor } },
    grid: { top: 70, bottom: 40 },
    xAxis: {
      type: 'category',
      data: DATA.exp2.perStyle.styles,
      axisLine: { lineStyle: { color: THEME.axisLineColor } },
      axisLabel: { color: THEME.textColor, fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      name: '下降率',
      nameTextStyle: { color: THEME.textColor },
      splitLine: { lineStyle: { color: THEME.splitLineColor } },
      axisLabel: { color: THEME.textColor, formatter: v => (v * 100).toFixed(0) + '%' }
    },
    series: [
      { name: 'Q2Q', type: 'bar', data: DATA.exp2.perStyle.dropQ2Q, itemStyle: { color: THEME.colors[0] } },
      { name: 'Q2P', type: 'bar', data: DATA.exp2.perStyle.dropQ2P, itemStyle: { color: THEME.colors[1] } },
      { name: 'Q2N', type: 'bar', data: DATA.exp2.perStyle.dropQ2N, itemStyle: { color: THEME.colors[2] } },
      { name: 'Q2R', type: 'bar', data: DATA.exp2.perStyle.dropQ2R, itemStyle: { color: THEME.colors[3] } },
      { name: 'Q2C', type: 'bar', data: DATA.exp2.perStyle.dropQ2C, itemStyle: { color: THEME.colors[4] } }
    ]
  };
  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
}

function renderExp3MRR() {
  const el = document.getElementById('chart-exp3-mrr');
  if (!el) return;
  const chart = echarts.init(el);

  const makeSeries = (name, data, color) => ({
    name, type: 'line', data, smooth: true, symbol: 'circle', symbolSize: 5,
    lineStyle: { color, width: 2.5 }, itemStyle: { color },
    emphasis: { lineStyle: { width: 3.5 } }
  });

  const option = {
    ...baseOption(),
    title: { text: 'MRR 随记忆池大小变化', left: 'center', top: 10, textStyle: { color: '#e2e8f0', fontSize: 14, fontWeight: 500 } },
    legend: { data: DATA.methods, top: 35, textStyle: { color: THEME.textColor } },
    grid: { top: 75, bottom: 45 },
    xAxis: {
      type: 'category',
      data: DATA.exp3.poolSizes,
      name: '记忆池大小',
      nameLocation: 'center',
      nameGap: 30,
      nameTextStyle: { color: THEME.textColor },
      axisLine: { lineStyle: { color: THEME.axisLineColor } },
      axisLabel: { color: THEME.textColor }
    },
    yAxis: {
      type: 'value',
      min: 0.4,
      max: 0.85,
      name: 'MRR',
      nameTextStyle: { color: THEME.textColor },
      splitLine: { lineStyle: { color: THEME.splitLineColor } },
      axisLabel: { color: THEME.textColor }
    },
    series: [
      makeSeries('Q2Q', DATA.exp3.mrr.q2q, THEME.colors[0]),
      makeSeries('Q2P', DATA.exp3.mrr.q2p, THEME.colors[1]),
      makeSeries('Q2N', DATA.exp3.mrr.q2n, THEME.colors[2]),
      makeSeries('Q2R', DATA.exp3.mrr.q2r, THEME.colors[3]),
      makeSeries('Q2C', DATA.exp3.mrr.q2c, THEME.colors[4])
    ]
  };
  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
}

function renderExp3Recall() {
  const el = document.getElementById('chart-exp3-recall');
  if (!el) return;
  const chart = echarts.init(el);

  const makeSeries = (name, data, color, dash) => ({
    name, type: 'line', data, smooth: true, symbol: 'circle', symbolSize: 4,
    lineStyle: { color, width: 2, type: dash ? 'dashed' : 'solid' }, itemStyle: { color }
  });

  const option = {
    ...baseOption(),
    title: { text: 'Q2Q 的 Recall@K 随池大小变化', left: 'center', top: 10, textStyle: { color: '#e2e8f0', fontSize: 14, fontWeight: 500 } },
    legend: { data: ['Recall@1', 'Recall@5'], top: 35, textStyle: { color: THEME.textColor } },
    grid: { top: 70, bottom: 45 },
    xAxis: {
      type: 'category',
      data: DATA.exp3.poolSizes,
      name: '记忆池大小',
      nameLocation: 'center',
      nameGap: 30,
      nameTextStyle: { color: THEME.textColor },
      axisLine: { lineStyle: { color: THEME.axisLineColor } },
      axisLabel: { color: THEME.textColor }
    },
    yAxis: {
      type: 'value',
      min: 0.5,
      max: 1.0,
      splitLine: { lineStyle: { color: THEME.splitLineColor } },
      axisLabel: { color: THEME.textColor }
    },
    series: [
      makeSeries('Recall@1', DATA.exp3.recallAt1.q2q, '#3b82f6', false),
      makeSeries('Recall@5', DATA.exp3.recallAt5.q2q, '#10b981', false)
    ]
  };
  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
}

document.addEventListener('DOMContentLoaded', () => {
  setTimeout(initCharts, 100);
});
