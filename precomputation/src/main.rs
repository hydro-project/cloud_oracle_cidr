#![feature(test)]
extern crate test;

use std::path::PathBuf;
use clap::Parser;

mod monolith;
mod monolith_enumeration;
mod pipelined;
mod util;

use hydroflow::tokio;
#[cfg(dev)]
use tracing_subscriber::fmt::format;
use util::{LatencyType, HfImplementation, compute_combinations};

#[derive(Parser)]
struct Args {
    #[clap(short, long)]
    output_dir: PathBuf,
}

#[hydroflow::main]
async fn main() {

    let output_dir_precomputation = PathBuf::from("precomputation");
    let output_file_name = "precomputation.csv";
    let args = Args::parse();
    let output_dir_full = args.output_dir.join(output_dir_precomputation);
    let output_file = output_dir_full.join(output_file_name);
    std::fs::create_dir_all(&output_dir_full).unwrap();

    // Enable tracing for dev build
    #[cfg(dev)]
    {
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    }

    let distance_constraint = 200;
    let d_scaling = [2, 10, 100, 300, 1000];
    let c_scaling = [2, 10, 100, 300, 1000];
    let iterations = 5;

    let implementations = &[HfImplementation::Monolith_Enumeration, HfImplementation::Monolith, HfImplementation::Pipelined];

    let num_filter_threads = 1;
    let num_dominated_threads = 1;

    let mut results = Vec::new();
    for num_d in d_scaling {
        for num_c in c_scaling {
            // All distances, s.t., all pairs of data centers fulfill the distance constraint, except for the diagonal
            let dist: Vec<_> = (0..num_d * num_d).into_iter().map(|i| {
                // Compute x and y indexes from linear index i
                let x = i % num_d;
                let y = i / num_d;
                test::black_box(if x != y {i + distance_constraint*2} else {0})
            }).collect();
            // All latencies, s.t., no placement is dominated, i.e., simply identity
            let latency: Vec<_> = (0..num_d*num_c).into_iter().map(|_i| test::black_box( 1.0 as f32)).collect();

            let expected_optimal_count = compute_combinations(num_d, num_d);

            for implementation in implementations {
                let implementation_str = format!("{}", implementation);
                println!("Running: {}, {}, {}, {}", num_d, num_c, distance_constraint, &implementation_str);

                let duration = create_and_run(implementation, &expected_optimal_count, &num_d, &num_c, &dist, &latency, &distance_constraint, &num_filter_threads, &num_dominated_threads).await;
                println!("Warmup duration: {:?}", duration);
    
                for _ in 0..iterations {
                    let duration = create_and_run(implementation, &expected_optimal_count, &num_d, &num_c, &dist, &latency, &distance_constraint, &num_filter_threads, &num_dominated_threads).await;

        
                    results.push((num_d, num_c, distance_constraint, implementation_str.clone(), duration));

                    print_results(&output_file, &results);
                }
            }
        }
    }

    print_results(&output_file, &results);
}

fn print_results(filename: &PathBuf, results: &Vec<(usize, usize, usize, String, std::time::Duration)>) {
    let header = "num_d, num_c, distance_constraint, implementation, duration_s";

    let mut output_lines = Vec::new();
    output_lines.push(header.to_string());

    for (num_d, num_c, distance_constraint, implementation, duration) in results {
        let content = format!("{}, {}, {}, {}, {}", num_d, num_c, distance_constraint, implementation, duration.as_secs_f64());
        output_lines.push(content);
    }

    let output_text = output_lines.join("\n");

    println!("{}", &output_text);
    std::fs::write(filename, output_text).expect("File write failed");
}

async fn create_and_run<'a>(implementation: &HfImplementation, expected_optimal_count: &'a usize, num_d: &'a usize, num_c: &'a usize, dist: &'a [usize], latency: &'a [LatencyType], dist_constraint: &'a usize, num_filter_threads: &'a usize, num_dominated_threads: &'a usize) -> std::time::Duration {

    match implementation {
        HfImplementation::Monolith => {
            monolith::create_and_run_hf(*expected_optimal_count, *num_d, *num_c, dist, latency, *dist_constraint).await
        }
        HfImplementation::Monolith_Enumeration => {
            monolith_enumeration::create_and_run_hf(*expected_optimal_count, *num_d, *num_c, dist, latency, *dist_constraint).await
        }
        HfImplementation::Pipelined => {
            pipelined::create_and_run_pipelined_hf(expected_optimal_count, num_d, num_c, dist, latency, dist_constraint, num_filter_threads, num_dominated_threads).await
        }
    }
}