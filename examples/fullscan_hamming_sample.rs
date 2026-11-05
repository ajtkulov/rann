//! Fullscan top-10 (cosine) on 250 random queries; report LSH/trie Hamming.
//!
//! Uses the Bloom-trie index's hyperplanes to hash each vector, then
//! `BitCode::hamming` between query and neighbor codes (same codes the trie uses).
//!
//! ```bash
//! cargo run --release --example fullscan_hamming_sample -- \
//!   --data data/glove.6B.300d --n 250 -k 10 --seed 1729 \
//!   -o data/benchmarks/fullscan_hamming_250
//! ```

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rann::benchmark::FullscanIndex;
use rann::{BitCode, TrieIndex};
use serde::Serialize;

fn usage() -> ! {
    eprintln!(
        "Usage: fullscan_hamming_sample [--data prefix] [--n 250] [-k 10] [--seed 1729] [-o stem]\n\
         Defaults: --data data/glove.6B.300d --n 250 -k 10 --seed 1729\n\
                   -o data/benchmarks/fullscan_hamming_250"
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
    /// Hamming distance of LSH codes (trie hash bits only).
    hamming_dist: u32,
}

#[derive(Serialize)]
struct QueryOut {
    id: i64,
    word: String,
    elapsed_ms: f64,
    neighbors: Vec<NeighborOut>,
}

#[derive(Serialize)]
struct Report {
    dataset: String,
    mode: &'static str,
    n_sample: usize,
    k: usize,
    seed: u64,
    h_bits: usize,
    num_vectors: usize,
    dim: usize,
    pack_ms: f64,
    total_query_ms: f64,
    /// Mean Hamming over neighbors with rank > 1 (exclude self at Hamming 0).
    mean_hamming_nonself: f64,
    /// Mean Hamming over all returned neighbors including self.
    mean_hamming_all: f64,
    queries: Vec<QueryOut>,
}

fn load_vocab(path: &PathBuf) -> Result<HashMap<i64, String>, Box<dyn std::error::Error>> {
    let f = File::open(path)?;
    let mut i2w = HashMap::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        let mut parts = line.splitn(2, '\t');
        let id: i64 = parts.next().ok_or("bad vocab")?.parse()?;
        let word = parts.next().ok_or("bad vocab")?.to_string();
        i2w.insert(id, word);
    }
    Ok(i2w)
}

fn sample_ids(n_total: usize, n_sample: usize, seed: u64) -> Vec<i64> {
    let mut ids: Vec<i64> = (0..n_total as i64).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    ids.shuffle(&mut rng);
    let take = n_sample.min(n_total);
    ids.truncate(take);
    ids
}

fn trie_hamming(index: &TrieIndex, qhash: &BitCode, neighbor_id: i64) -> u32 {
    let Some(v) = index.vectors.get(&neighbor_id) else {
        return u32::MAX;
    };
    qhash.hamming(&index.projector.project(v))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
    }

    let mut prefix = PathBuf::from("data/glove.6B.300d");
    let mut n_sample = 250usize;
    let mut k = 10usize;
    let mut seed = 1729u64;
    let mut out_stem = PathBuf::from("data/benchmarks/fullscan_hamming_250");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                prefix = PathBuf::from(args.get(i).unwrap_or_else(|| usage()));
            }
            "--n" => {
                i += 1;
                n_sample = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "-k" => {
                i += 1;
                k = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "-o" | "--out" => {
                i += 1;
                out_stem = PathBuf::from(args.get(i).unwrap_or_else(|| usage()));
            }
            _ => usage(),
        }
        i += 1;
    }

    let rann_path = PathBuf::from(format!("{}.rann", prefix.display()));
    let vocab_path = PathBuf::from(format!("{}.vocab", prefix.display()));

    eprintln!("Loading {} ...", rann_path.display());
    let t_load = Instant::now();
    let index = TrieIndex::load(&mut BufReader::with_capacity(
        1 << 20,
        File::open(&rann_path)?,
    ))?;
    eprintln!(
        "Loaded {} × {}d H={} in {:.1} ms",
        index.vectors.len(),
        index.dim(),
        index.params.h,
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    let i2w = if vocab_path.exists() {
        load_vocab(&vocab_path)?
    } else {
        HashMap::new()
    };

    eprintln!("Packing fullscan matrix ...");
    let t_pack = Instant::now();
    let fs = FullscanIndex::from_vectors(&index.vectors, index.dim());
    let pack_ms = t_pack.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Packed in {pack_ms:.1} ms");

    let query_ids = sample_ids(index.vectors.len(), n_sample, seed);
    eprintln!(
        "Fullscan top-{k} for {} random queries (seed={seed}) ...",
        query_ids.len()
    );

    let mut queries = Vec::with_capacity(query_ids.len());
    let mut ham_all: Vec<u32> = Vec::new();
    let mut ham_nonself: Vec<u32> = Vec::new();
    let t_all = Instant::now();

    println!();
    println!(
        "{:<6} {:<16} {}",
        "id", "word", "top-10: word(cos/H)"
    );
    println!("{:-<90}", "");

    for (qi, &qid) in query_ids.iter().enumerate() {
        let word = i2w
            .get(&qid)
            .cloned()
            .unwrap_or_else(|| format!("id:{qid}"));
        let query = index.vectors.get(&qid).ok_or("missing vector")?.clone();
        let qhash = index.projector.project(&query);

        let t0 = Instant::now();
        let hits = fs.search_topk(&query, k);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let neighbors: Vec<NeighborOut> = hits
            .iter()
            .enumerate()
            .map(|(rank, (id, dist))| {
                let hd = trie_hamming(&index, &qhash, *id);
                ham_all.push(hd);
                if *id != qid {
                    ham_nonself.push(hd);
                }
                NeighborOut {
                    rank: rank + 1,
                    id: *id,
                    word: i2w.get(id).cloned().unwrap_or_default(),
                    cosine_dist: *dist,
                    cosine_sim: 1.0 - dist,
                    hamming_dist: hd,
                }
            })
            .collect();

        if qi < 15 || qi + 1 == query_ids.len() {
            let preview: Vec<String> = neighbors
                .iter()
                .take(5)
                .map(|n| format!("{}({:.2}/H{})", n.word, n.cosine_sim, n.hamming_dist))
                .collect();
            println!(
                "{:<6} {:<16} {}{}",
                qid,
                word,
                preview.join(", "),
                if neighbors.len() > 5 {
                    format!(" +{}", neighbors.len() - 5)
                } else {
                    String::new()
                }
            );
        } else if qi == 15 {
            println!("... ({} more queries)", query_ids.len() - 16);
        }

        if (qi + 1) % 50 == 0 {
            eprintln!("  done {}/{}", qi + 1, query_ids.len());
        }

        queries.push(QueryOut {
            id: qid,
            word,
            elapsed_ms,
            neighbors,
        });
    }

    let total_query_ms = t_all.elapsed().as_secs_f64() * 1000.0;
    let mean_hamming_all = if ham_all.is_empty() {
        0.0
    } else {
        ham_all.iter().map(|&x| x as f64).sum::<f64>() / ham_all.len() as f64
    };
    let mean_hamming_nonself = if ham_nonself.is_empty() {
        0.0
    } else {
        ham_nonself.iter().map(|&x| x as f64).sum::<f64>() / ham_nonself.len() as f64
    };

    let report = Report {
        dataset: prefix.display().to_string(),
        mode: "fullscan",
        n_sample: query_ids.len(),
        k,
        seed,
        h_bits: index.params.h,
        num_vectors: index.vectors.len(),
        dim: index.dim(),
        pack_ms,
        total_query_ms,
        mean_hamming_nonself,
        mean_hamming_all,
        queries,
    };

    eprintln!();
    eprintln!(
        "Queries: {}  total {:.1}s  mean {:.2} ms/query",
        report.n_sample,
        total_query_ms / 1000.0,
        total_query_ms / report.n_sample as f64
    );
    eprintln!(
        "Mean trie Hamming: all={:.1}  non-self={:.1}  (H={})",
        mean_hamming_all, mean_hamming_nonself, index.params.h
    );

    if let Some(parent) = out_stem.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json_path = PathBuf::from(format!("{}.json", out_stem.display()));
    serde_json::to_writer_pretty(File::create(&json_path)?, &report)?;
    eprintln!("Wrote {}", json_path.display());

    let tsv_path = PathBuf::from(format!("{}.tsv", out_stem.display()));
    let mut tsv = File::create(&tsv_path)?;
    writeln!(
        tsv,
        "query_id\tquery_word\telapsed_ms\trank\tneighbor_id\tneighbor_word\tcosine_dist\tcosine_sim\thamming_dist\th_bits"
    )?;
    for q in &report.queries {
        for n in &q.neighbors {
            writeln!(
                tsv,
                "{}\t{}\t{:.3}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}",
                q.id,
                q.word,
                q.elapsed_ms,
                n.rank,
                n.id,
                n.word,
                n.cosine_dist,
                n.cosine_sim,
                n.hamming_dist,
                report.h_bits
            )?;
        }
    }
    eprintln!("Wrote {}", tsv_path.display());

    Ok(())
}
