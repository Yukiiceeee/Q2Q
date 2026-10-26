// Figure directory, relative to experiment/web/index.html. This used to be a
// `figs -> ../figs/locomo` symlink, which a Windows clone checks out as a
// plain text file and every image 404s. A relative URL needs no symlink.
const FIGS_BASE = '../figs/locomo/';

const DATA = {
  meta: {
    dataset: 'LoCoMo',
    model: 'BGE-Large-EN-v1.5',
    dimension: 1024,
    nQueries: 1982,
    nSessions: 272,
    seed: 42
  },

  methods: ['Q2Q', 'Q2P', 'Q2N', 'Q2R', 'Q2C'],
  methodColors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],

  exp1: {
    similarity: {
      q2q: 0.7511, q2p: 0.7454, q2n: 0.7195, q2r: 0.6080, q2c: 0.6414
    },
    pairwise: [
      { pair: 'Q2Q vs Q2C', t: 64.0, p: '< 1e-300', d: 1.437, winRate: 0.932 },
      { pair: 'Q2Q vs Q2R', t: 84.7, p: '< 1e-300', d: 1.901, winRate: 0.986 },
      { pair: 'Q2Q vs Q2N', t: 17.3, p: '2.9e-62', d: 0.388, winRate: 0.702 },
      { pair: 'Q2Q vs Q2P', t: 3.9, p: '9.4e-5', d: 0.088, winRate: 0.528 },
      { pair: 'Q2P vs Q2C', t: 50.2, p: '< 1e-300', d: 1.129, winRate: 0.880 },
      { pair: 'Q2N vs Q2C', t: 37.1, p: '6.6e-229', d: 0.833, winRate: 0.882 }
    ],
    byCategory: {
      labels: ['类别1 (n=282)', '类别2 (n=321)', '类别3 (n=92)', '类别4 (n=841)', '类别5 (n=446)'],
      q2qMean: [0.7457, 0.7886, 0.6963, 0.7757, 0.6923],
      q2cMean: [0.6302, 0.6232, 0.6057, 0.6580, 0.6378],
      gap: [0.1155, 0.1654, 0.0907, 0.1178, 0.0545],
      cohensD: [1.937, 2.215, 1.494, 1.697, 0.830]
    },
    overall: {
      gap_mean: 0.1097,
      gap_std: 0.0763,
      cohens_d: 1.437,
      q2q_win_rate: 0.932,
      t_statistic: 64.0
    }
  },

  exp2: {
    robustness: {
      q2q: 0.790, q2p: 0.753, q2n: 0.801, q2r: 0.821, q2c: 0.847
    },
    variance: {
      q2q: 0.00171, q2p: 0.00171, q2n: 0.00158, q2r: 0.00167, q2c: 0.00147
    },
    varianceRatio: 0.862,
    perStyle: {
      styles: ['推理式', '场景式', '因果式', '外围式', '否定式'],
      dropQ2Q: [0.0919, 0.1209, 0.1041, 0.1024, 0.1215],
      dropQ2P: [0.1182, 0.1467, 0.1310, 0.1324, 0.1554],
      dropQ2N: [0.0798, 0.1049, 0.0932, 0.0908, 0.1131],
      dropQ2R: [0.0456, 0.0710, 0.0455, 0.0525, 0.0680],
      dropQ2C: [0.0393, 0.0505, 0.0511, 0.0409, 0.0575]
    }
  },

  exp3: {
    poolSizes: [27, 54, 81, 108, 135, 162, 189, 216, 243, 270, 272],
    mrr: {
      q2q: [0.801, 0.745, 0.760, 0.761, 0.766, 0.763, 0.762, 0.756, 0.747, 0.734, 0.733],
      q2p: [0.780, 0.717, 0.707, 0.714, 0.725, 0.714, 0.705, 0.692, 0.683, 0.673, 0.672],
      q2n: [0.819, 0.769, 0.766, 0.763, 0.777, 0.762, 0.758, 0.753, 0.739, 0.731, 0.732],
      q2r: [0.607, 0.540, 0.548, 0.562, 0.569, 0.547, 0.528, 0.516, 0.502, 0.492, 0.492],
      q2c: [0.670, 0.594, 0.592, 0.600, 0.614, 0.598, 0.590, 0.581, 0.574, 0.567, 0.566]
    },
    recallAt5: {
      q2q: [0.899, 0.862, 0.870, 0.870, 0.872, 0.869, 0.868, 0.864, 0.857, 0.848, 0.848],
      q2p: [0.892, 0.849, 0.828, 0.830, 0.832, 0.831, 0.822, 0.812, 0.803, 0.792, 0.791],
      q2n: [0.905, 0.880, 0.872, 0.864, 0.880, 0.876, 0.872, 0.863, 0.854, 0.848, 0.848],
      q2r: [0.777, 0.702, 0.702, 0.720, 0.726, 0.711, 0.680, 0.664, 0.647, 0.641, 0.642],
      q2c: [0.905, 0.749, 0.741, 0.750, 0.775, 0.774, 0.764, 0.756, 0.744, 0.735, 0.733]
    },
    recallAt1: {
      q2q: [0.723, 0.645, 0.670, 0.671, 0.678, 0.675, 0.675, 0.667, 0.654, 0.638, 0.637],
      q2p: [0.689, 0.611, 0.615, 0.620, 0.636, 0.617, 0.607, 0.593, 0.583, 0.574, 0.573],
      q2n: [0.757, 0.679, 0.682, 0.678, 0.691, 0.672, 0.667, 0.663, 0.644, 0.636, 0.637],
      q2r: [0.466, 0.405, 0.418, 0.436, 0.444, 0.415, 0.398, 0.386, 0.375, 0.367, 0.366],
      q2c: [0.507, 0.462, 0.461, 0.469, 0.481, 0.457, 0.454, 0.442, 0.441, 0.435, 0.434]
    }
  },

  figures: {
    exp1: [
      { file: 'exp1_method_comparison.png', title: '方法对比' },
      { file: 'exp1_gap_distribution.png', title: '差距分布' },
      { file: 'exp1_discriminability.png', title: '区分度分析' },
      { file: 'exp1_directional_alignment.png', title: '方向对齐' },
      { file: 'exp1_precision_at_k.png', title: 'Precision@K' },
      { file: 'exp1_tsne_nearest.png', title: 't-SNE（最近邻）' },
      { file: 'exp1_tsne_all.png', title: 't-SNE（全局）' }
    ],
    exp2: [
      { file: 'exp2_drop_rate.png', title: '各风格下降率' },
      { file: 'exp2_style_boxplot.png', title: '风格鲁棒性箱线图' },
      { file: 'exp2_variance_scatter.png', title: '方差散点图' }
    ],
    exp3: [
      { file: 'exp3_mrr_vs_pool.png', title: 'MRR 随池大小变化' },
      { file: 'exp3_rank_vs_pool.png', title: '排名随池大小变化' },
      { file: 'exp3_rank_drift.png', title: '排名漂移' }
    ]
  }
};
