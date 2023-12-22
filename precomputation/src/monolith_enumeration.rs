use std::thread;
use std::time::Instant;

use hydroflow::futures::StreamExt;
use hydroflow::itertools::Itertools;

use crate::debug_println;
use crate::util::{DataCenterIndexType, LatencyType, PlacementType};

pub(crate) async fn create_and_run_hf<'a>(
    expected_optimal_count: usize,
    num_d: usize,
    num_c: usize,
    dist: &[usize],
    latency: &[LatencyType],
    dist_constraint: usize,
) -> std::time::Duration {
    let d: Vec<_> = (0 as DataCenterIndexType..num_d as DataCenterIndexType).collect();
    let expected_optimal_count_clone = expected_optimal_count.clone();

    let (tx_optimal_count, rx_optimal_count) = hydroflow::util::unbounded_channel();

    // Atomic flag for thread to wait for executing and when to stop
    let coordination_flag_main = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let latency = latency.to_vec();
    let dist = dist.to_vec();

    let coordination_flag_thread = coordination_flag_main.clone();
    let executor_thread = thread::spawn(move || {
        let mut hf = hydroflow::hydroflow_syntax! {
            // Directly create pairs as combinations of data centers
            data_center_pairs = source_iter(d.into_iter().tuple_combinations());

            valid_placements = data_center_pairs
                -> inspect(|_| {debug_println!("Filtering DC pair")})
                // Apply distance constraint
                -> filter(|(d_0, d_1)|{
                    let dist_index = (*d_0) as usize *num_d + (*d_1) as usize;
                    dist[dist_index] > dist_constraint
                })
                // Create placement and compute its latency to all clients
                -> inspect(|_|{debug_println!("Computing placement latency")})
                -> map(|(d_0, d_1)| -> PlacementType {
                    let latency_c: Vec<_> = (0..num_c).into_iter().map(|c| {
                        let latency_index_0 = d_0 as usize * num_c + c;
                        let latency_index_1 = d_1 as usize * num_c + c;
                        let latency_0 = latency[latency_index_0];
                        let latency_1 = latency[latency_index_1];
                        let write_latency = LatencyType::max(latency_0, latency_1);
                        let read_latency = LatencyType::min(latency_0, latency_1);
                        (write_latency, read_latency)
                    }).collect();

                    PlacementType {d_0, d_1, latency: latency_c}
                });

                valid_placements
                    -> inspect(|_|{debug_println!("Found optimal placement");})
                    -> map(|_|1) -> reduce::<'static>(|accum: &mut _, elem| {*accum += elem})
                    -> for_each(|acc|{
                        debug_println!("Number of optimal placements: {}", acc);
                        tx_optimal_count.send(acc).unwrap();
                    });

        };

        // Coordinate start by waiting for true
        while coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == false {}

        debug_println!("Computing optimal placements");

        while coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == true {
            let _worked = hf.run_available();
            debug_println!("Iteration of dominated returned {}", _worked);
        }
    });

    // Receive last element from rx_optimal_count channel
    let start_time = Instant::now();

    coordination_flag_main.store(true, std::sync::atomic::Ordering::Release);

    let mut rx_optimal_count = rx_optimal_count;
    while let Some(element) = rx_optimal_count.next().await {
        if element >= expected_optimal_count_clone {
            println!("Found {}/{} optimal placements", element, expected_optimal_count_clone);
            break;
        }
    }

    coordination_flag_main.store(false, std::sync::atomic::Ordering::Release);

    let end_time = Instant::now();
    let duration = end_time - start_time;

    executor_thread.join().expect("Failed to join thread");

    return duration;
}
