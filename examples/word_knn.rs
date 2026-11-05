//! Example: word(s) from the GloVe benchmark dataset → top-K neighbors.
//!
//! Includes cosine similarity and LSH Hamming distance (query hash vs neighbor hash).
//!
//! Single word:
//! ```bash
//! cargo run --release --example word_knn -- singapore -k 10
//! ```
//!
//! Batch (first 20 + 20 random ids):
//! ```bash
//! cargo run --release --example word_knn -- --batch -k 10 \
//!   -o data/benchmarks/word_knn_batch
//! ```

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use rann::benchmark::{build_query_set, FullscanIndex};
use rann::{BitCode, CosineDistance, HammingStrategy, SearchParams, TrieIndex};
use serde::Serialize;

fn usage() -> ! {
    eprintln!(
        "Usage:\n\
           word_knn <word> [-k N] [--data prefix] [--mode MODE]\n\
           word_knn --batch [-k N] [--data prefix] [--mode MODE] [--seed N] [-o stem]\n\
         \n\
         --batch uses first 20 + 20 random vocab lines (benchmark cohorts).\n\
         MODE: fullscan (default) | astar | multiprobe | beam\n\
         Defaults: --data data/glove.6B.300d  -k 10  --seed 1729"
    );
    std::process::exit(2);
}

#[derive(Serialize)]
struct NeighborOut {
    rank: usize,
    id: i64,
    word: String,
    cosine_dist: f32,
    cosine_sim: f32,
    /// Hamming distance between LSH binary codes (H bits).
    hamming_dist: u32,
}

#[derive(Serialize)]
struct QueryOut {
    cohort: String,
    id: i64,
    word: String,
    elapsed_ms: f64,
    h_bits: usize,
    neighbors: Vec<NeighborOut>,
}

#[derive(Serialize)]
struct BatchReport {
    dataset: String,
    mode: String,
    k: usize,
    seed: u64,
    h_bits: usize,
    num_queries: usize,
    queries: Vec<QueryOut>,
}

fn load_vocab(path: &PathBuf) -> Result<(HashMap<String, i64>, HashMap<i64, String>), Box<dyn std::error::Error>> {
    let f = File::open(path)?;
    let mut w2i = HashMap::new();
    let mut i2w = HashMap::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        let mut parts = line.splitn(2, '\t');
        let id: i64 = parts.next().ok_or("bad vocab")?.parse()?;
        let word = parts.next().ok_or("bad vocab")?.to_string();
        w2i.insert(word.clone(), id);
        i2w.insert(id, word);
    }
    Ok((w2i, i2w))
}

fn search_hits(
    index: &TrieIndex,
    fullscan: Option<&FullscanIndex>,
    query: &[f32],
    k: usize,
    mode: &str,
) -> Vec<(i64, f32)> {
    match mode {
        "fullscan" => fullscan
            .expect("fullscan matrix")
            .search_topk(query, k),
        "astar" | "multiprobe" | "beam" => {
            let strategy = HammingStrategy::parse(mode).expect("mode");
            let mut params = SearchParams::default();
            params.strategy = strategy;
            params.r_max = 32;
            params.min_candidates = k.max(50);
            params.beam_width = 128;
            params.max_expansions = Some(500_000);
            index.search_with_params(query, k, &params, &CosineDistance)
        }
        _ => unreachable!(),
    }
}

fn hamming_to_neighbor(index: &TrieIndex, qhash: &BitCode, neighbor_id: i64) -> u32 {
    let Some(v) = index.vectors.get(&neighbor_id) else {
        return u32::MAX;
    };
    let nhash = index.projector.project(v);
    qhash.hamming(&nhash)
}

fn enrich_neighbors(
    index: &TrieIndex,
    i2w: &HashMap<i64, String>,
    qhash: &BitCode,
    hits: &[(i64, f32)],
) -> Vec<NeighborOut> {
    hits.iter()
        .enumerate()
        .map(|(rank, (id, dist))| NeighborOut {
            rank: rank + 1,
            id: *id,
            word: i2w.get(id).cloned().unwrap_or_default(),
            cosine_dist: *dist,
            cosine_sim: 1.0 - dist,
            hamming_dist: hamming_to_neighbor(index, qhash, *id),
        })
        .collect()
}

fn print_table_row(cohort: &str, qword: &str, neighbors: &[NeighborOut]) {
    let top: Vec<String> = neighbors
        .iter()
        .take(5)
        .map(|n| format!("{}({:.2}/H{})", n.word, n.cosine_sim, n.hamming_dist))
        .collect();
    let more = if neighbors.len() > 5 {
        format!(" +{}", neighbors.len() - 5)
    } else {
        String::new()
    };
    println!(
        "{:<8} {:<16} {}",
        cohort,
        qword,
        top.join(", ") + &more
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
    }

    let mut batch = false;
    let mut word: Option<String> = None;
    let mut k = 10usize;
    let mut prefix = PathBuf::from("data/glove.6B.300d");
    let mut mode = "fullscan".to_string();
    let mut seed = 1729u64;
    let mut out_stem = PathBuf::from("data/benchmarks/word_knn_batch");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--batch" => batch = true,
            "-k" => {
                i += 1;
                k = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "--data" => {
                i += 1;
                prefix = PathBuf::from(args.get(i).unwrap_or_else(|| usage()));
            }
            "--mode" => {
                i += 1;
                mode = args.get(i).unwrap_or_else(|| usage()).clone();
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "-o" | "--out" => {
                i += 1;
                out_stem = PathBuf::from(args.get(i).unwrap_or_else(|| usage()));
            }
            s if !s.starts_with('-') && word.is_none() && !batch => {
                word = Some(s.to_string());
            }
            _ => usage(),
        }
        i += 1;
    }

    if !batch && word.is_none() {
        usage();
    }
    if !matches!(
        mode.as_str(),
        "fullscan" | "astar" | "multiprobe" | "beam"
    ) {
        eprintln!("unknown --mode {mode}");
        usage();
    }

    let rann_path = PathBuf::from(format!("{}.rann", prefix.display()));
    let vocab_path = PathBuf::from(format!("{}.vocab", prefix.display()));

    eprintln!("Loading vocab {} ...", vocab_path.display());
    let (w2i, i2w) = load_vocab(&vocab_path)?;

    eprintln!("Loading index {} ...", rann_path.display());
    let t_load = Instant::now();
    let index = TrieIndex::load(&mut BufReader::with_capacity(
        1 << 20,
        File::open(&rann_path)?,
    ))?;
    eprintln!(
        "Loaded {} vectors × {}d (H={}) in {:.1} ms",
        index.vectors.len(),
        index.dim(),
        index.params.h,
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    let fullscan = if mode == "fullscan" {
        eprintln!("Packing fullscan matrix ...");
        let t = Instant::now();
        let fs = FullscanIndex::from_vectors(&index.vectors, index.dim());
        eprintln!("Packed in {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        Some(fs)
    } else {
        None
    };

    if !batch {
        let word = word.unwrap();
        let qid = *w2i
            .get(&word)
            .ok_or_else(|| format!("word not in vocab: {word:?}"))?;
        let query = index.vectors.get(&qid).ok_or("missing vector")?.clone();
        let qhash = index.projector.project(&query);
        let t0 = Instant::now();
        let hits = search_hits(&index, fullscan.as_ref(), &query, k, &mode);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let neighbors = enrich_neighbors(&index, &i2w, &qhash, &hits);

        println!();
        println!(
            "word → top-{k}  ({mode}, {ms:.1} ms, H={} LSH bits)",
            index.params.h
        );
        println!("{:-<72}", "");
        println!(
            "{:<4} {:<12} {:<18} {:>10} {:>10} {:>8}",
            "rank", "id", "word", "cos_dist", "cos_sim", "hamming"
        );
        println!("{:-<72}", "");
        for n in &neighbors {
            println!(
                "{:<4} {:<12} {:<18} {:>10.6} {:>10.6} {:>8}",
                n.rank, n.id, n.word, n.cosine_dist, n.cosine_sim, n.hamming_dist
            );
        }
        return Ok(());
    }

    let specs = build_query_set(index.vectors.len(), seed);
    eprintln!(
        "Batch: {} queries (first+random), K={k}, mode={mode}, seed={seed}",
        specs.len()
    );

    let mut queries_out = Vec::with_capacity(specs.len());
    println!();
    println!(
        "{:<8} {:<16} {}",
        "cohort", "query", "top (word(cos_sim/Hhamming), …)"
    );
    println!("{:-<88}", "");

    for spec in &specs {
        let qword = i2w
            .get(&spec.id)
            .cloned()
            .unwrap_or_else(|| format!("id:{}", spec.id));
        let query = index
            .vectors
            .get(&spec.id)
            .ok_or("missing vector")?
            .clone();
        let qhash = index.projector.project(&query);
        let t0 = Instant::now();
        let hits = search_hits(&index, fullscan.as_ref(), &query, k, &mode);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let neighbors = enrich_neighbors(&index, &i2w, &qhash, &hits);

        print_table_row(spec.cohort, &qword, &neighbors);

        queries_out.push(QueryOut {
            cohort: spec.cohort.to_string(),
            id: spec.id,
            word: qword,
            elapsed_ms,
            h_bits: index.params.h,
            neighbors,
        });
    }

    let report = BatchReport {
        dataset: prefix.display().to_string(),
        mode: mode.clone(),
        k,
        seed,
        h_bits: index.params.h,
        num_queries: queries_out.len(),
        queries: queries_out,
    };

    if let Some(parent) = out_stem.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json_path = PathBuf::from(format!("{}.json", out_stem.display()));
    let mut f = File::create(&json_path)?;
    serde_json::to_writer_pretty(&mut f, &report)?;
    f.write_all(b"\n")?;
    eprintln!("\nWrote {}", json_path.display());

    let tsv_path = PathBuf::from(format!("{}.tsv", out_stem.display()));
    let mut tsv = File::create(&tsv_path)?;
    writeln!(
        tsv,
        "cohort\tquery_id\tquery_word\telapsed_ms\trank\tneighbor_id\tneighbor_word\tcosine_dist\tcosine_sim\thamming_dist\th_bits"
    )?;
    for q in &report.queries {
        for n in &q.neighbors {
            writeln!(
                tsv,
                "{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}",
                q.cohort,
                q.id,
                q.word,
                q.elapsed_ms,
                n.rank,
                n.id,
                n.word,
                n.cosine_dist,
                n.cosine_sim,
                n.hamming_dist,
                q.h_bits
            )?;
        }
    }
    eprintln!("Wrote {}", tsv_path.display());

    Ok(())
}
