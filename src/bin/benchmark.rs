//! Microbenchmark: load a dataset, time top-K KNN on first+random cohorts.
//!
//! Usage:
//!   cargo run --release --bin benchmark -- data/glove.6B.300d
//!   cargo run --release --bin benchmark -- data/glove.6B.300d --mode exact --seed 1729 -o data/benchmarks/glove
//!   cargo run --release --bin benchmark -- data/glove.6B.300d --mode ann --k 3,5,10

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use rann::benchmark::{
    default_output_stem, load_vocab, run_benchmark, write_html, write_json, SearchMode,
    DEFAULT_K_VALUES,
};
use rann::TrieIndex;

fn usage() -> ! {
    eprintln!(
        "Usage: benchmark <dataset_prefix> [--seed N] [--mode exact|ann] [-o stem] [--k 3,5,10,...]\n\
         \n\
         Writes <stem>.json and <stem>.html"
    );
    std::process::exit(2);
}

fn parse_k_list(s: &str) -> Result<Vec<usize>, String> {
    let mut ks = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        ks.push(
            part.parse::<usize>()
                .map_err(|_| format!("bad K value: {part}"))?,
        );
    }
    if ks.is_empty() {
        return Err("empty --k list".into());
    }
    Ok(ks)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
    }

    let prefix = PathBuf::from(args.remove(0));
    let mut seed = 1729u64;
    let mut mode = SearchMode::Exact;
    let mut out: Option<PathBuf> = None;
    let mut k_values: Vec<usize> = DEFAULT_K_VALUES.to_vec();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "--mode" => {
                i += 1;
                let m = args.get(i).unwrap_or_else(|| usage());
                mode = SearchMode::parse(m).unwrap_or_else(|| usage());
            }
            "-o" | "--out" => {
                i += 1;
                out = Some(PathBuf::from(args.get(i).unwrap_or_else(|| usage())));
            }
            "--k" => {
                i += 1;
                k_values = parse_k_list(args.get(i).unwrap_or_else(|| usage()))?;
            }
            _ => usage(),
        }
        i += 1;
    }

    let stem = out.unwrap_or_else(|| default_output_stem(&prefix));
    let rann_path = PathBuf::from(format!("{}.rann", prefix.display()));
    let vocab_path = PathBuf::from(format!("{}.vocab", prefix.display()));

    eprintln!("Loading {} ...", rann_path.display());
    let t_load = Instant::now();
    let mut index = TrieIndex::load(&mut BufReader::with_capacity(
        1 << 20,
        File::open(&rann_path)?,
    ))?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Loaded {} vectors × {}d in {:.1} ms",
        index.vectors.len(),
        index.dim(),
        load_ms
    );

    let id_to_word = if vocab_path.exists() {
        eprintln!("Loading vocab {} ...", vocab_path.display());
        load_vocab(&vocab_path)?
    } else {
        Default::default()
    };

    eprintln!(
        "Running benchmark mode={} seed={} K={:?} ...",
        mode.as_str(),
        seed,
        k_values
    );
    let t_run = Instant::now();
    let report = run_benchmark(
        &mut index,
        &prefix.display().to_string(),
        mode,
        seed,
        &k_values,
        load_ms,
        &id_to_word,
    );
    eprintln!(
        "Finished {} query×K runs in {:.1}s",
        report.queries.iter().map(|q| q.runs.len()).sum::<usize>(),
        t_run.elapsed().as_secs_f64()
    );

    let json_path = PathBuf::from(format!("{}.json", stem.display()));
    let html_path = PathBuf::from(format!("{}.html", stem.display()));
    write_json(&report, &json_path)?;
    write_html(&report, &html_path)?;
    eprintln!("Wrote {}", json_path.display());
    eprintln!("Wrote {}", html_path.display());

    // Print compact summary
    let mut ks = k_values.clone();
    ks.sort_unstable();
    eprintln!("\nK\tmean_ms\tp50_ms\tp95_ms");
    for k in ks {
        if let Some(s) = report.summary.by_k.get(&k.to_string()) {
            eprintln!(
                "{}\t{:.3}\t{:.3}\t{:.3}",
                k, s.mean_ms, s.p50_ms, s.p95_ms
            );
        }
    }

    Ok(())
}
