//! Microbenchmark: load a dataset, time top-K KNN on first+random cohorts.
//!
//! Usage:
//!   cargo run --release --bin benchmark -- data/glove.6B.300d --mode fullscan -o data/benchmarks/glove-fullscan
//!   cargo run --release --bin benchmark -- data/glove.6B.300d --strategies astar,multiprobe,beam -o data/benchmarks/glove-strategies

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use rann::benchmark::{
    default_output_stem, load_vocab, run_benchmark, run_strategy_compare, write_html, write_json,
    write_strategy_html, write_strategy_json, SearchMode, DEFAULT_K_VALUES,
};
use rann::{HammingStrategy, SearchParams, TrieIndex};

fn usage() -> ! {
    eprintln!(
        "Usage: benchmark <dataset_prefix> [options]\n\
         \n\
         Options:\n\
           --seed N\n\
           --mode exact|fullscan|ann\n\
           --strategies astar,multiprobe,beam   (ANN strategy compare vs fullscan GT)\n\
           --r-max N --beam-width N --min-candidates N --max-expansions N\n\
           -o stem\n\
           --k 3,5,10,20,50\n\
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

fn parse_strategies(s: &str) -> Result<Vec<HammingStrategy>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out.push(
            HammingStrategy::parse(part)
                .ok_or_else(|| format!("unknown strategy: {part}"))?,
        );
    }
    if out.is_empty() {
        return Err("empty --strategies".into());
    }
    Ok(out)
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
    let mut strategies: Option<Vec<HammingStrategy>> = None;
    let mut search_params = SearchParams::default();

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
            "--strategies" => {
                i += 1;
                strategies = Some(parse_strategies(
                    args.get(i).unwrap_or_else(|| usage()),
                )?);
            }
            "--r-max" => {
                i += 1;
                search_params.r_max = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "--beam-width" => {
                i += 1;
                search_params.beam_width = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "--min-candidates" => {
                i += 1;
                search_params.min_candidates = args.get(i).unwrap_or_else(|| usage()).parse()?;
            }
            "--max-expansions" => {
                i += 1;
                search_params.max_expansions =
                    Some(args.get(i).unwrap_or_else(|| usage()).parse()?);
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

    let json_path = PathBuf::from(format!("{}.json", stem.display()));
    let html_path = PathBuf::from(format!("{}.html", stem.display()));

    if let Some(strats) = strategies {
        eprintln!(
            "Comparing strategies {:?} seed={} K={:?}",
            strats.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            seed,
            k_values
        );
        let t_run = Instant::now();
        let report = run_strategy_compare(
            &mut index,
            &prefix.display().to_string(),
            &strats,
            seed,
            &k_values,
            load_ms,
            &id_to_word,
            &search_params,
        );
        eprintln!("Finished in {:.1}s", t_run.elapsed().as_secs_f64());
        write_strategy_json(&report, &json_path)?;
        write_strategy_html(&report, &html_path)?;
        eprintln!("Wrote {}", json_path.display());
        eprintln!("Wrote {}", html_path.display());

        eprintln!("\nstrategy\tK\tmean_ms\tp50_ms\trecall");
        let mut ks = k_values.clone();
        ks.sort_unstable();
        for s in &report.strategies {
            let name = s.strategy.as_deref().unwrap_or("?");
            for &k in &ks {
                let key = k.to_string();
                let mean = s.summary.by_k.get(&key).map(|t| t.mean_ms).unwrap_or(0.0);
                let p50 = s.summary.by_k.get(&key).map(|t| t.p50_ms).unwrap_or(0.0);
                let rec = s
                    .recall_by_k
                    .as_ref()
                    .and_then(|m| m.get(&key))
                    .copied()
                    .unwrap_or(0.0);
                eprintln!("{name}\t{k}\t{mean:.3}\t{p50:.3}\t{rec:.4}");
            }
        }
        return Ok(());
    }

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

    write_json(&report, &json_path)?;
    write_html(&report, &html_path)?;
    eprintln!("Wrote {}", json_path.display());
    eprintln!("Wrote {}", html_path.display());

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
