//! Build a serialized rann dataset from a GloVe-style text file.
//!
//! Format: `word f1 f2 ... fD` per line.
//!
//! Usage:
//!   cargo run --release --bin gen_ds -- glove.6B.300d.txt
//!   cargo run --release --bin gen_ds -- glove.6B.300d.txt --limit 10000
//!   cargo run --release --bin gen_ds -- glove.6B.300d.txt -o data/glove

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rann::{IdWidth, IndexParams, TrieIndex};

fn usage() -> ! {
    eprintln!(
        "Usage: gen_ds <glove.txt> [-o out_prefix] [--limit N] [--fp-rate P]\n\
         \n\
         Writes:\n\
           <out_prefix>.rann   — serialized index (Bloom filters + vectors)\n\
           <out_prefix>.vocab  — lines `id\\tword`"
    );
    std::process::exit(2);
}

fn parse_args() -> (PathBuf, PathBuf, Option<usize>, f64) {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
    }

    let input = PathBuf::from(args.remove(0));
    let mut out_prefix: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut fp_rate = 1e-4;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                i += 1;
                out_prefix = Some(PathBuf::from(args.get(i).unwrap_or_else(|| usage())));
            }
            "--limit" => {
                i += 1;
                limit = Some(
                    args.get(i)
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                );
            }
            "--fp-rate" => {
                i += 1;
                fp_rate = args
                    .get(i)
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage());
            }
            _ => usage(),
        }
        i += 1;
    }

    let out = out_prefix.unwrap_or_else(|| {
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset");
        PathBuf::from(format!("data/{stem}"))
    });

    (input, out, limit, fp_rate)
}

fn load_glove(
    path: &Path,
    limit: Option<usize>,
) -> Result<(Vec<(i64, Vec<f32>)>, Vec<String>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(1 << 20, file);
    let mut dataset = Vec::new();
    let mut vocab = Vec::new();
    let mut dim: Option<usize> = None;

    for (line_no, line) in reader.lines().enumerate() {
        if let Some(n) = limit {
            if dataset.len() >= n {
                break;
            }
        }
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let word = parts.next().ok_or("empty line")?.to_string();
        let mut vec: Vec<f32> = Vec::new();
        for tok in parts {
            vec.push(tok.parse::<f32>()?);
        }
        if vec.is_empty() {
            return Err(format!("line {}: missing vector", line_no + 1).into());
        }
        match dim {
            None => dim = Some(vec.len()),
            Some(d) if d != vec.len() => {
                return Err(format!(
                    "line {}: dim {} != expected {}",
                    line_no + 1,
                    vec.len(),
                    d
                )
                .into());
            }
            _ => {}
        }
        let id = dataset.len() as i64;
        vocab.push(word);
        dataset.push((id, vec));

        if dataset.len() % 50_000 == 0 {
            eprintln!("  parsed {:>7} vectors...", dataset.len());
        }
    }

    if dataset.is_empty() {
        return Err("no vectors loaded".into());
    }
    Ok((dataset, vocab))
}

fn write_vocab(path: &Path, vocab: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    for (id, word) in vocab.iter().enumerate() {
        writeln!(w, "{id}\t{word}")?;
    }
    w.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (input, out_prefix, limit, fp_rate) = parse_args();
    let rann_path = PathBuf::from(format!("{}.rann", out_prefix.display()));
    let vocab_path = PathBuf::from(format!("{}.vocab", out_prefix.display()));

    eprintln!("Loading {} ...", input.display());
    let t0 = Instant::now();
    let (dataset, vocab) = load_glove(&input, limit)?;
    let dim = dataset[0].1.len();
    eprintln!(
        "Loaded {} vectors × {}d in {:.1}s",
        dataset.len(),
        dim,
        t0.elapsed().as_secs_f64()
    );

    let params = IndexParams {
        h: 256,
        nn: 6,
        seed: 1729,
        fp_rate,
        id_width: IdWidth::U32,
        max_expansions: 100_000,
    };
    eprintln!(
        "Building index (H={}, NN={}, seed={}, fp_rate={}) ...",
        params.h, params.nn, params.seed, params.fp_rate
    );
    let t1 = Instant::now();
    let mut index = TrieIndex::new(dim, params, dataset.len());
    for (i, (id, vec)) in dataset.iter().enumerate() {
        index.insert(*id, vec.clone());
        if (i + 1) % 25_000 == 0 {
            eprintln!(
                "  inserted {:>7}/{} ({:.1}s)",
                i + 1,
                dataset.len(),
                t1.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!("Index built in {:.1}s", t1.elapsed().as_secs_f64());

    if let Some(parent) = rann_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    eprintln!("Writing {} ...", rann_path.display());
    let t2 = Instant::now();
    {
        let mut w = BufWriter::new(File::create(&rann_path)?);
        index.save(&mut w)?;
        w.flush()?;
    }
    eprintln!(
        "Wrote {} ({:.1} MB) in {:.1}s",
        rann_path.display(),
        std::fs::metadata(&rann_path)?.len() as f64 / (1024.0 * 1024.0),
        t2.elapsed().as_secs_f64()
    );

    write_vocab(&vocab_path, &vocab)?;
    eprintln!("Wrote {} ({} words)", vocab_path.display(), vocab.len());

    // Smoke: search nearest to first vector
    let q = &dataset[0].1;
    let hits = index.search(q, 5);
    eprintln!("Smoke top-5 for id=0 ('{}'):", vocab[0]);
    for (id, dist) in hits {
        let word = vocab.get(id as usize).map(|s| s.as_str()).unwrap_or("?");
        eprintln!("  id={id} word={word} cosine_dist={dist:.6}");
    }

    eprintln!("Done.");
    Ok(())
}
