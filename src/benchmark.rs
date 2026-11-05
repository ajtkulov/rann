//! Dataset KNN microbenchmark: query cohorts, timings, JSON + HTML reports.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::distance::{top_k_by_distance, CosineDistance};
use crate::search::{HammingStrategy, SearchParams};
use crate::trie_index::TrieIndex;

pub const DEFAULT_K_VALUES: [usize; 5] = [3, 5, 10, 20, 50];
pub const FIRST_COHORT_SIZE: usize = 20;
pub const RANDOM_COHORT_SIZE: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// HashMap sidecar walk + per-vector cosine (legacy exact path).
    Exact,
    /// Contiguous in-memory full dataset scan (baseline).
    Fullscan,
    Ann,
}

impl SearchMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "exact" => Some(SearchMode::Exact),
            "fullscan" | "baseline" => Some(SearchMode::Fullscan),
            "ann" => Some(SearchMode::Ann),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SearchMode::Exact => "exact",
            SearchMode::Fullscan => "fullscan",
            SearchMode::Ann => "ann",
        }
    }
}

/// Contiguous row-major matrix for exact full-dataset scan baseline.
#[derive(Debug, Clone)]
pub struct FullscanIndex {
    pub ids: Vec<i64>,
    /// `n * dim` floats, row-major.
    pub data: Vec<f32>,
    /// Precomputed L2 norms per row.
    pub norms: Vec<f32>,
    pub dim: usize,
}

impl FullscanIndex {
    pub fn from_vectors(vectors: &HashMap<i64, Vec<f32>>, dim: usize) -> Self {
        let mut ids: Vec<i64> = vectors.keys().copied().collect();
        ids.sort_unstable();
        let n = ids.len();
        let mut data = Vec::with_capacity(n * dim);
        let mut norms = Vec::with_capacity(n);
        for &id in &ids {
            let v = vectors.get(&id).expect("id present");
            assert_eq!(v.len(), dim);
            let mut ss = 0.0f32;
            for &x in v {
                data.push(x);
                ss += x * x;
            }
            norms.push(ss.sqrt());
        }
        Self {
            ids,
            data,
            norms,
            dim,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Exact cosine top-K over the entire matrix (smaller distance = nearer).
    pub fn search_topk(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        assert_eq!(query.len(), self.dim);
        let k = k.min(self.len()).max(1);
        let mut q_norm_sq = 0.0f32;
        for &x in query {
            q_norm_sq += x * x;
        }
        let q_norm = q_norm_sq.sqrt();

        // Max-heap of (distance, row) so we can pop worst; keep k best (smallest dist).
        let mut heap: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();

        for row in 0..self.len() {
            let start = row * self.dim;
            let row_slice = &self.data[start..start + self.dim];
            let mut dot = 0.0f32;
            for (a, b) in query.iter().zip(row_slice.iter()) {
                dot += a * b;
            }
            let nb = self.norms[row];
            let dist = if q_norm == 0.0 || nb == 0.0 {
                f32::INFINITY
            } else {
                1.0 - (dot / (q_norm * nb)).clamp(-1.0, 1.0)
            };

            if heap.len() < k {
                heap.push((OrderedFloat(dist), row));
            } else if let Some(top) = heap.peek() {
                if dist < top.0 .0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), row));
                }
            }
        }

        let mut out: Vec<(i64, f32)> = heap
            .into_iter()
            .map(|(OrderedFloat(d), row)| (self.ids[row], d))
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

/// Total-order wrapper so `f32` distances can live in a `BinaryHeap`.
#[derive(Copy, Clone, Debug)]
struct OrderedFloat(f32);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for OrderedFloat {}
impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborHit {
    pub id: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word: Option<String>,
    pub distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KRun {
    pub k: usize,
    pub elapsed_ms: f64,
    pub neighbors: Vec<NeighborHit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bloom_probes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub cohort: String,
    pub id: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word: Option<String>,
    pub runs: Vec<KRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    pub n: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub by_k: HashMap<String, TimingStats>,
    pub by_cohort_k: HashMap<String, HashMap<String, TimingStats>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunReport {
    pub version: u32,
    pub dataset: String,
    pub mode: SearchMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    pub seed: u64,
    pub k_values: Vec<usize>,
    pub load_ms: f64,
    pub num_vectors: usize,
    pub dim: usize,
    pub queries: Vec<QueryResult>,
    pub summary: Summary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_by_k: Option<HashMap<String, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCompareReport {
    pub version: u32,
    pub dataset: String,
    pub seed: u64,
    pub k_values: Vec<usize>,
    pub load_ms: f64,
    pub num_vectors: usize,
    pub dim: usize,
    pub ground_truth: String,
    pub strategies: Vec<RunReport>,
}

#[derive(Debug, Clone)]
pub struct QuerySpec {
    pub cohort: &'static str,
    pub id: i64,
}

/// Build first-20 + 20 random (seeded) query ids.
/// When `n > 20`, random ids are drawn from `[20, n)` (no overlap with first cohort).
pub fn build_query_set(num_vectors: usize, seed: u64) -> Vec<QuerySpec> {
    let n = num_vectors;
    let first_n = FIRST_COHORT_SIZE.min(n);
    let mut queries: Vec<QuerySpec> = (0..first_n)
        .map(|id| QuerySpec {
            cohort: "first",
            id: id as i64,
        })
        .collect();

    if n <= first_n {
        return queries;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pool: Vec<i64> = (first_n as i64..n as i64).collect();
    pool.shuffle(&mut rng);
    let take = RANDOM_COHORT_SIZE.min(pool.len());
    for &id in pool.iter().take(take) {
        queries.push(QuerySpec {
            cohort: "random",
            id,
        });
    }
    queries
}

fn exact_topk(index: &TrieIndex, query: &[f32], k: usize) -> Vec<(i64, f32)> {
    let ids: Vec<i64> = index.vectors.keys().copied().collect();
    top_k_by_distance(query, &ids, &index.vectors, k, &CosineDistance)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = rank - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

pub fn compute_stats(samples: &[f64]) -> TimingStats {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mean = if n == 0 {
        0.0
    } else {
        sorted.iter().sum::<f64>() / n as f64
    };
    TimingStats {
        n,
        mean_ms: mean,
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        min_ms: sorted.first().copied().unwrap_or(0.0),
        max_ms: sorted.last().copied().unwrap_or(0.0),
    }
}

pub fn build_summary(queries: &[QueryResult]) -> Summary {
    let mut by_k_samples: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut by_cohort: HashMap<String, HashMap<usize, Vec<f64>>> = HashMap::new();

    for q in queries {
        for run in &q.runs {
            by_k_samples.entry(run.k).or_default().push(run.elapsed_ms);
            by_cohort
                .entry(q.cohort.clone())
                .or_default()
                .entry(run.k)
                .or_default()
                .push(run.elapsed_ms);
        }
    }

    let by_k = by_k_samples
        .into_iter()
        .map(|(k, samples)| (k.to_string(), compute_stats(&samples)))
        .collect();

    let by_cohort_k = by_cohort
        .into_iter()
        .map(|(cohort, map)| {
            let inner = map
                .into_iter()
                .map(|(k, samples)| (k.to_string(), compute_stats(&samples)))
                .collect();
            (cohort, inner)
        })
        .collect();

    Summary { by_k, by_cohort_k }
}

pub fn run_benchmark(
    index: &mut TrieIndex,
    dataset: &str,
    mode: SearchMode,
    seed: u64,
    k_values: &[usize],
    load_ms: f64,
    id_to_word: &HashMap<i64, String>,
) -> RunReport {
    let num_vectors = index.vectors.len();
    let dim = index.dim();
    let query_specs = build_query_set(num_vectors, seed);

    if mode == SearchMode::Ann {
        index.params.max_expansions = index.params.max_expansions.max(5_000_000);
    }

    // Pack contiguous matrix once; not charged to per-query timings.
    let fullscan = if mode == SearchMode::Fullscan {
        eprintln!(
            "Packing fullscan matrix ({} × {}) ...",
            num_vectors, dim
        );
        let t_pack = Instant::now();
        let fs = FullscanIndex::from_vectors(&index.vectors, dim);
        eprintln!("Packed in {:.1} ms", t_pack.elapsed().as_secs_f64() * 1000.0);
        Some(fs)
    } else {
        None
    };

    let mut queries = Vec::with_capacity(query_specs.len());
    for spec in &query_specs {
        let vector = index
            .vectors
            .get(&spec.id)
            .cloned()
            .unwrap_or_else(|| panic!("missing vector for id {}", spec.id));
        let word = id_to_word.get(&spec.id).cloned();
        let mut runs = Vec::with_capacity(k_values.len());
        for &k in k_values {
            let t0 = Instant::now();
            let hits = match mode {
                SearchMode::Exact => exact_topk(index, &vector, k),
                SearchMode::Fullscan => fullscan
                    .as_ref()
                    .expect("fullscan matrix")
                    .search_topk(&vector, k),
                SearchMode::Ann => index.search(&vector, k),
            };
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let neighbors = hits
                .into_iter()
                .map(|(id, distance)| NeighborHit {
                    id,
                    word: id_to_word.get(&id).cloned(),
                    distance,
                })
                .collect();
            runs.push(KRun {
                k,
                elapsed_ms,
                neighbors,
                expansions: None,
                bloom_probes: None,
                recall: None,
            });
        }
        queries.push(QueryResult {
            cohort: spec.cohort.to_string(),
            id: spec.id,
            word,
            runs,
        });
    }

    let summary = build_summary(&queries);
    RunReport {
        version: 1,
        dataset: dataset.to_string(),
        mode,
        strategy: None,
        seed,
        k_values: k_values.to_vec(),
        load_ms,
        num_vectors,
        dim,
        queries,
        summary,
        recall_by_k: None,
    }
}

fn recall_at_k(pred: &[(i64, f32)], truth: &[(i64, f32)]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_ids: HashSet<i64> = truth.iter().map(|(id, _)| *id).collect();
    let hit = pred.iter().filter(|(id, _)| truth_ids.contains(id)).count();
    hit as f64 / truth.len() as f64
}

/// Benchmark one or more ANN Hamming strategies vs fullscan ground truth.
pub fn run_strategy_compare(
    index: &mut TrieIndex,
    dataset: &str,
    strategies: &[HammingStrategy],
    seed: u64,
    k_values: &[usize],
    load_ms: f64,
    id_to_word: &HashMap<i64, String>,
    search_params_base: &SearchParams,
) -> StrategyCompareReport {
    let num_vectors = index.vectors.len();
    let dim = index.dim();
    let query_specs = build_query_set(num_vectors, seed);

    eprintln!("Packing fullscan ground truth matrix...");
    let t_pack = Instant::now();
    let gt = FullscanIndex::from_vectors(&index.vectors, dim);
    eprintln!("Packed in {:.1} ms", t_pack.elapsed().as_secs_f64() * 1000.0);

    index.params.max_expansions = index
        .params
        .max_expansions
        .max(search_params_base.max_expansions.unwrap_or(5_000_000));

    let mut strategy_reports = Vec::new();
    for &strategy in strategies {
        eprintln!("Strategy {} ...", strategy.as_str());
        let mut params = search_params_base.clone();
        params.strategy = strategy;

        let mut queries = Vec::with_capacity(query_specs.len());
        let mut recall_samples: HashMap<usize, Vec<f64>> = HashMap::new();

        for spec in &query_specs {
            let vector = index.vectors.get(&spec.id).cloned().unwrap();
            let word = id_to_word.get(&spec.id).cloned();
            let qhash = index.projector.project(&vector);

            // Time discovery once per query; rerank cheaply per K.
            let t0 = Instant::now();
            let cands = index.discover_candidates(&qhash, &params);
            let discover_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let mut runs = Vec::new();
            for &k in k_values {
                let truth = gt.search_topk(&vector, k);
                let t1 = Instant::now();
                let hits = top_k_by_distance(
                    &vector,
                    &cands.ids,
                    &index.vectors,
                    k,
                    &CosineDistance,
                );
                let rerank_ms = t1.elapsed().as_secs_f64() * 1000.0;
                let rec = recall_at_k(&hits, &truth);
                recall_samples.entry(k).or_default().push(rec);
                let neighbors = hits
                    .into_iter()
                    .map(|(id, distance)| NeighborHit {
                        id,
                        word: id_to_word.get(&id).cloned(),
                        distance,
                    })
                    .collect();
                runs.push(KRun {
                    k,
                    // Attribute discovery cost to each K row for comparable mean latency.
                    elapsed_ms: discover_ms + rerank_ms,
                    neighbors,
                    expansions: Some(cands.expansions),
                    bloom_probes: Some(cands.bloom_probes),
                    recall: Some(rec),
                });
            }
            queries.push(QueryResult {
                cohort: spec.cohort.to_string(),
                id: spec.id,
                word,
                runs,
            });
        }

        let summary = build_summary(&queries);
        let recall_by_k: HashMap<String, f64> = recall_samples
            .into_iter()
            .map(|(k, samples)| {
                let mean = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
                (k.to_string(), mean)
            })
            .collect();

        strategy_reports.push(RunReport {
            version: 1,
            dataset: dataset.to_string(),
            mode: SearchMode::Ann,
            strategy: Some(strategy.as_str().to_string()),
            seed,
            k_values: k_values.to_vec(),
            load_ms,
            num_vectors,
            dim,
            queries,
            summary,
            recall_by_k: Some(recall_by_k),
        });
    }

    StrategyCompareReport {
        version: 2,
        dataset: dataset.to_string(),
        seed,
        k_values: k_values.to_vec(),
        load_ms,
        num_vectors,
        dim,
        ground_truth: "fullscan".into(),
        strategies: strategy_reports,
    }
}

pub fn write_strategy_json(
    report: &StrategyCompareReport,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    serde_json::to_writer_pretty(File::create(path)?, report)?;
    Ok(())
}

pub fn render_strategy_html(report: &StrategyCompareReport) -> String {
    let mut rows = String::new();
    let mut ks = report.k_values.clone();
    ks.sort_unstable();
    for s in &report.strategies {
        let name = s.strategy.as_deref().unwrap_or("?");
        for &k in &ks {
            let key = k.to_string();
            let timing = s.summary.by_k.get(&key);
            let recall = s
                .recall_by_k
                .as_ref()
                .and_then(|m| m.get(&key))
                .copied()
                .unwrap_or(0.0);
            let mean = timing.map(|t| t.mean_ms).unwrap_or(0.0);
            let p50 = timing.map(|t| t.p50_ms).unwrap_or(0.0);
            rows.push_str(&format!(
                "<tr><td>{name}</td><td>{k}</td><td>{mean:.3}</td><td>{p50:.3}</td><td>{recall:.4}</td></tr>\n"
            ));
        }
    }
    format!(
        r##"<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>strategy compare</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem}}
table{{border-collapse:collapse}} th,td{{border:1px solid #ccc;padding:.4rem .7rem}}
th{{background:#f0f0f0}}
</style></head><body>
<h1>Hamming strategy compare</h1>
<p>Dataset <code>{dataset}</code> · GT fullscan · seed {seed} · {n} × {d}d</p>
<table>
<thead><tr><th>strategy</th><th>K</th><th>mean ms</th><th>p50 ms</th><th>recall@K</th></tr></thead>
<tbody>
{rows}
</tbody></table>
</body></html>
"##,
        dataset = html_escape(&report.dataset),
        seed = report.seed,
        n = report.num_vectors,
        d = report.dim,
    )
}

pub fn write_strategy_html(
    report: &StrategyCompareReport,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = File::create(path)?;
    f.write_all(render_strategy_html(report).as_bytes())?;
    Ok(())
}

pub fn write_json(report: &RunReport, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = File::create(path)?;
    serde_json::to_writer_pretty(f, report)?;
    Ok(())
}

fn svg_grouped_bars(
    title: &str,
    categories: &[(String, f64, f64)], // label, first_mean, random_mean
) -> String {
    let width = 720.0;
    let height = 280.0;
    let margin_l = 56.0;
    let margin_b = 40.0;
    let margin_t = 36.0;
    let margin_r = 24.0;
    let plot_w = width - margin_l - margin_r;
    let plot_h = height - margin_t - margin_b;
    let max_v = categories
        .iter()
        .map(|(_, a, b)| a.max(*b))
        .fold(0.0_f64, f64::max)
        .max(1e-6);
    let n = categories.len().max(1) as f64;
    let group_w = plot_w / n;
    let bar_w = group_w * 0.32;

    let mut bars = String::new();
    for (i, (label, first, random)) in categories.iter().enumerate() {
        let gx = margin_l + i as f64 * group_w;
        let h1 = (*first / max_v) * plot_h;
        let h2 = (*random / max_v) * plot_h;
        let y1 = margin_t + plot_h - h1;
        let y2 = margin_t + plot_h - h2;
        bars.push_str(&format!(
            r##"<rect x="{x1:.1}" y="{y1:.1}" width="{bar_w:.1}" height="{h1:.1}" fill="#4C6EF5"/>
<rect x="{x2:.1}" y="{y2:.1}" width="{bar_w:.1}" height="{h2:.1}" fill="#F08C00"/>
<text x="{cx:.1}" y="{ty:.1}" text-anchor="middle" font-size="12" fill="#333">{label}</text>
"##,
            x1 = gx + group_w * 0.15,
            x2 = gx + group_w * 0.15 + bar_w + 4.0,
            cx = gx + group_w * 0.5,
            ty = height - 12.0,
        ));
    }

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <text x="{tx}" y="22" font-size="16" font-weight="600" fill="#111">{title}</text>
  <text x="12" y="{ymid}" font-size="11" fill="#555" transform="rotate(-90 12 {ymid})">mean latency (ms)</text>
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{yb}" stroke="#999"/>
  <line x1="{margin_l}" y1="{yb}" x2="{xr}" y2="{yb}" stroke="#999"/>
  <text x="{margin_l}" y="{yt}" font-size="10" fill="#666">{max_label}</text>
  {bars}
  <rect x="{lx}" y="8" width="12" height="12" fill="#4C6EF5"/><text x="{ltx}" y="18" font-size="12" fill="#333">first</text>
  <rect x="{lx2}" y="8" width="12" height="12" fill="#F08C00"/><text x="{ltx2}" y="18" font-size="12" fill="#333">random</text>
</svg>"##,
        tx = margin_l,
        ymid = margin_t + plot_h * 0.5,
        yb = margin_t + plot_h,
        xr = margin_l + plot_w,
        yt = margin_t + 10.0,
        max_label = format!("{max_v:.2}"),
        lx = width - 140.0,
        ltx = width - 124.0,
        lx2 = width - 80.0,
        ltx2 = width - 64.0,
    )
}

pub fn render_html(report: &RunReport) -> String {
    let mut k_sorted: Vec<usize> = report.k_values.clone();
    k_sorted.sort_unstable();

    let mut cohort_cats = Vec::new();
    for &k in &k_sorted {
        let key = k.to_string();
        let first = report
            .summary
            .by_cohort_k
            .get("first")
            .and_then(|m| m.get(&key))
            .map(|s| s.mean_ms)
            .unwrap_or(0.0);
        let random = report
            .summary
            .by_cohort_k
            .get("random")
            .and_then(|m| m.get(&key))
            .map(|s| s.mean_ms)
            .unwrap_or(0.0);
        cohort_cats.push((format!("K={k}"), first, random));
    }

    let chart = svg_grouped_bars(
        "Mean KNN latency by K (first vs random cohort)",
        &cohort_cats,
    );

    let mut rows = String::new();
    for &k in &k_sorted {
        let key = k.to_string();
        if let Some(s) = report.summary.by_k.get(&key) {
            rows.push_str(&format!(
                "<tr><td>{k}</td><td>{n}</td><td>{mean:.3}</td><td>{p50:.3}</td><td>{p95:.3}</td><td>{min:.3}</td><td>{max:.3}</td></tr>\n",
                n = s.n,
                mean = s.mean_ms,
                p50 = s.p50_ms,
                p95 = s.p95_ms,
                min = s.min_ms,
                max = s.max_ms,
            ));
        }
    }

    let mut cohort_rows = String::new();
    for cohort in ["first", "random"] {
        for &k in &k_sorted {
            let key = k.to_string();
            if let Some(s) = report
                .summary
                .by_cohort_k
                .get(cohort)
                .and_then(|m| m.get(&key))
            {
                cohort_rows.push_str(&format!(
                    "<tr><td>{cohort}</td><td>{k}</td><td>{mean:.3}</td><td>{p50:.3}</td><td>{min:.3}</td><td>{max:.3}</td></tr>\n",
                    mean = s.mean_ms,
                    p50 = s.p50_ms,
                    min = s.min_ms,
                    max = s.max_ms,
                ));
            }
        }
    }

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>rann benchmark — {dataset}</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; color: #222; background: #fafafa; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  .meta {{ color: #555; margin-bottom: 1.5rem; font-size: 0.95rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0 2rem; background: #fff; }}
  th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.7rem; font-size: 0.9rem; }}
  th {{ background: #f0f0f0; text-align: left; }}
  .chart {{ background: #fff; border: 1px solid #e5e5e5; padding: 0.75rem; display: inline-block; }}
  code {{ background: #eee; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
  <h1>KNN benchmark</h1>
  <div class="meta">
    Dataset <code>{dataset}</code> · mode <code>{mode}</code> · seed {seed}<br/>
    {num_vectors} vectors × {dim}d · load {load_ms:.1} ms · queries {nq}<br/>
    Single-run wall times; first query may include allocator noise.
  </div>
  <div class="chart">{chart}</div>
  <h2>Overall latency by K (ms)</h2>
  <table>
    <thead><tr><th>K</th><th>n</th><th>mean</th><th>p50</th><th>p95</th><th>min</th><th>max</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <h2>Latency by cohort × K (ms)</h2>
  <table>
    <thead><tr><th>cohort</th><th>K</th><th>mean</th><th>p50</th><th>min</th><th>max</th></tr></thead>
    <tbody>
{cohort_rows}
    </tbody>
  </table>
</body>
</html>
"##,
        dataset = html_escape(&report.dataset),
        mode = report.mode.as_str(),
        seed = report.seed,
        num_vectors = report.num_vectors,
        dim = report.dim,
        load_ms = report.load_ms,
        nq = report.queries.len(),
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

pub fn write_html(report: &RunReport, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = File::create(path)?;
    f.write_all(render_html(report).as_bytes())?;
    Ok(())
}

pub fn load_vocab(path: &Path) -> Result<HashMap<i64, String>, Box<dyn std::error::Error>> {
    let f = File::open(path)?;
    let mut i2w = HashMap::new();
    for line in std::io::BufRead::lines(BufReader::new(f)) {
        let line = line?;
        let mut parts = line.splitn(2, '\t');
        let id: i64 = parts.next().ok_or("bad vocab")?.parse()?;
        let word = parts.next().ok_or("bad vocab")?.to_string();
        i2w.insert(id, word);
    }
    Ok(i2w)
}

pub fn default_output_stem(dataset_prefix: &Path) -> PathBuf {
    let stem = dataset_prefix
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset");
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from(format!("data/benchmarks/{stem}-{ts}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IdWidth, IndexParams};
    use crate::trie_index::build_from_dataset;

    fn tiny_index() -> TrieIndex {
        let params = IndexParams {
            h: 24,
            nn: 6,
            seed: 1729,
            fp_rate: 1e-6,
            id_width: IdWidth::U32,
            max_expansions: 50_000,
        };
        let data: Vec<(i64, Vec<f32>)> = (0..80)
            .map(|i| {
                let a = i as f32 * 0.07;
                (i, vec![a.cos(), a.sin(), (i % 5) as f32 * 0.1])
            })
            .collect();
        build_from_dataset(&data, params)
    }

    #[test]
    fn query_set_sizes_no_overlap() {
        let qs = build_query_set(80, 1729);
        let first: Vec<_> = qs.iter().filter(|q| q.cohort == "first").collect();
        let random: Vec<_> = qs.iter().filter(|q| q.cohort == "random").collect();
        assert_eq!(first.len(), 20);
        assert_eq!(random.len(), 20);
        let first_ids: std::collections::HashSet<_> = first.iter().map(|q| q.id).collect();
        for q in &random {
            assert!(!first_ids.contains(&q.id));
        }
    }

    #[test]
    fn seeded_random_reproducible() {
        let a: Vec<_> = build_query_set(100, 42)
            .into_iter()
            .filter(|q| q.cohort == "random")
            .map(|q| q.id)
            .collect();
        let b: Vec<_> = build_query_set(100, 42)
            .into_iter()
            .filter(|q| q.cohort == "random")
            .map(|q| q.id)
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn json_report_roundtrip_shape() {
        let mut index = tiny_index();
        let report = run_benchmark(
            &mut index,
            "test://tiny",
            SearchMode::Exact,
            1729,
            &DEFAULT_K_VALUES,
            1.5,
            &HashMap::new(),
        );
        assert_eq!(report.version, 1);
        assert_eq!(report.k_values, DEFAULT_K_VALUES);
        assert_eq!(report.queries.len(), 40);
        let json = serde_json::to_string(&report).unwrap();
        let parsed: RunReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.queries.len(), 40);
        assert!(parsed.summary.by_k.contains_key("10"));
        assert!(parsed.summary.by_cohort_k.contains_key("first"));
        assert!(parsed.summary.by_cohort_k.contains_key("random"));
        let html = render_html(&parsed);
        assert!(html.contains("Mean KNN latency"));
        assert!(html.contains("<svg"));
    }

    #[test]
    fn fullscan_matches_exact_neighbors() {
        let index = tiny_index();
        let fs = FullscanIndex::from_vectors(&index.vectors, index.dim());
        let q = index.vectors.get(&0).unwrap();
        let a = exact_topk(&index, q, 5);
        let b = fs.search_topk(q, 5);
        assert_eq!(a.len(), b.len());
        for ((id_a, d_a), (id_b, d_b)) in a.iter().zip(b.iter()) {
            assert_eq!(id_a, id_b);
            assert!((d_a - d_b).abs() < 1e-5, "{d_a} vs {d_b}");
        }
    }

    #[test]
    fn fullscan_benchmark_runs() {
        let mut index = tiny_index();
        let report = run_benchmark(
            &mut index,
            "test://tiny-fullscan",
            SearchMode::Fullscan,
            1729,
            &[3, 10],
            0.0,
            &HashMap::new(),
        );
        assert_eq!(report.mode, SearchMode::Fullscan);
        assert_eq!(report.queries.len(), 40);
        assert!(report.summary.by_k.contains_key("10"));
    }
}
