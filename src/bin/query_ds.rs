//! Query a serialized rann dataset for top-K neighbors.
//!
//! Usage:
//!   cargo run --release --bin query_ds -- data/glove.6B.300d 1353
//!   cargo run --release --bin query_ds -- data/glove.6B.300d singapore -k 10
//!   cargo run --release --bin query_ds -- data/glove.6B.300d singapore --ann

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use rann::{CosineDistance, Distance, TrieIndex};

fn usage() -> ! {
    eprintln!("Usage: query_ds <out_prefix> <id|word> [-k N] [--ann]");
    std::process::exit(2);
}

fn exact_topk(index: &TrieIndex, query: &[f32], k: usize) -> Vec<(i64, f32)> {
    let dist = CosineDistance;
    let mut scored: Vec<(i64, f32)> = index
        .vectors
        .iter()
        .map(|(&id, v)| (id, dist.distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        usage();
    }
    let prefix = PathBuf::from(args.remove(0));
    let query_token = args.remove(0);
    let mut k = 10usize;
    let mut use_ann = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-k" => {
                i += 1;
                k = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "--ann" => use_ann = true,
            _ => usage(),
        }
        i += 1;
    }

    let rann_path = PathBuf::from(format!("{}.rann", prefix.display()));
    let vocab_path = PathBuf::from(format!("{}.vocab", prefix.display()));

    eprintln!("Loading vocab {} ...", vocab_path.display());
    let (w2i, i2w) = load_vocab(&vocab_path)?;

    let qid: i64 = if let Ok(id) = query_token.parse::<i64>() {
        id
    } else {
        *w2i
            .get(&query_token)
            .ok_or_else(|| format!("word not in vocab: {query_token}"))?
    };
    let qword = i2w.get(&qid).map(|s| s.as_str()).unwrap_or("?");
    eprintln!("Query id={qid} word={qword} k={k}");

    eprintln!("Loading index {} ...", rann_path.display());
    let t0 = Instant::now();
    let mut index = TrieIndex::load(&mut BufReader::with_capacity(
        1 << 20,
        File::open(&rann_path)?,
    ))?;
    eprintln!(
        "Loaded in {:.1}s ({} vectors)",
        t0.elapsed().as_secs_f64(),
        index.vectors.len()
    );

    let query = index
        .vectors
        .get(&qid)
        .ok_or_else(|| format!("id {qid} missing from index vectors"))?
        .clone();

    let t1 = Instant::now();
    let hits = if use_ann {
        index.params.max_expansions = 5_000_000;
        eprintln!("ANN search (max_expansions={}) ...", index.params.max_expansions);
        index.search(&query, k)
    } else {
        eprintln!("Exact cosine scan over {} vectors ...", index.vectors.len());
        exact_topk(&index, &query, k)
    };
    eprintln!("Search done in {:.3}s\n", t1.elapsed().as_secs_f64());

    println!("rank\tid\tword\tcosine_dist\tcosine_sim");
    for (rank, (id, dist)) in hits.iter().enumerate() {
        let word = i2w.get(id).map(|s| s.as_str()).unwrap_or("?");
        let sim = 1.0 - dist;
        println!("{}\t{}\t{}\t{:.6}\t{:.6}", rank + 1, id, word, dist, sim);
    }
    Ok(())
}
