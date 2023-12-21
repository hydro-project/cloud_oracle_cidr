use hydroflow::hydroflow_syntax;
use hydroflow::scheduled::graph::Hydroflow;
use hydroflow::tokio::sync::mpsc::UnboundedSender;
use hydroflow::tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};
use std::thread;
use std::time::Instant;

use crate::debug_println;
use crate::util::{lt_slice, DataCenterIndexType, LatencyType, PlacementType};

// in main: //pipelined::create_and_run_pipelined_hf(expected_optimal_count, num_d, num_c, dist, latency, dist_constraint, num_filter_threads, num_dominated_threads).await

fn create_data_centers_hf<'a>(
    num_d: &usize,
    outputs_build: Vec<UnboundedSender<DataCenterIndexType>>,
    outputs_probe: Vec<UnboundedSender<DataCenterIndexType>>,
) -> Hydroflow<'a> {
    let d: Vec<_> = (0 as DataCenterIndexType..*num_d as DataCenterIndexType).collect();

    hydroflow_syntax! {
        data_center = source_iter(d.into_iter()) -> tee();
        // Broadcast data center to all build sides
        data_center -> for_each(|d|{
            debug_println!("Broadcasting DC");
            outputs_build.iter().for_each(|output|{output.send(d).unwrap();});
        });
        // Distribute data center across all probe sides
        data_center -> enumerate() -> for_each(|(i, pair): (usize, _)|{
            debug_println!("Distributing DC");
            let output_index = i % outputs_probe.len();
            outputs_probe[output_index].send(pair).unwrap();
        });
    }
}

fn create_valid_placements_hf<'a>(
    num_d: &'a usize,
    num_c: &'a usize,
    dist_constraint: &'a usize,
    dist: &'a [usize],
    latency: &'a [LatencyType],
    rx_data_center_build: UnboundedReceiverStream<DataCenterIndexType>,
    rx_data_center_probe: UnboundedReceiverStream<DataCenterIndexType>,
    txs_all_valid_placements: Vec<UnboundedSender<PlacementType>>,
    txs_probe_valid_placements: Vec<UnboundedSender<PlacementType>>,
) -> Hydroflow<'a> {
    hydroflow_syntax! {
        source_stream(rx_data_center_build)
            -> inspect(|_| {debug_println!("Received build DC")})
            -> [0]data_center_pairs;
        source_stream(rx_data_center_probe)
            -> inspect(|_| {debug_println!("Received probe DC")})
            -> [1]data_center_pairs;
        data_center_pairs = cross_join::<'static,'tick>();

        valid_placements = data_center_pairs
            -> inspect(|_| {debug_println!("Filtering DC pair")})
            // Apply distance constraint
            -> filter(|(d_0, d_1)|{
                let dist_index = (*d_0) as usize *num_d + (*d_1) as usize;
                dist[dist_index] > *dist_constraint
            })
            // Create placement and compute its latency to all clients
            -> inspect(|_|{debug_println!("Computing placement latency")})
            -> map(|(d_0, d_1)| -> PlacementType {
                let latency_c: Vec<_> = (0..*num_c).into_iter().map(|c| {
                    let latency_index_0 = d_0 as usize * *num_c + c;
                    let latency_index_1 = d_1 as usize * *num_c + c;
                    let latency_0 = latency[latency_index_0];
                    let latency_1 = latency[latency_index_1];
                    let write_latency = LatencyType::max(latency_0, latency_1);
                    let read_latency = LatencyType::min(latency_0, latency_1);
                    (write_latency, read_latency)
                }).collect();

                PlacementType {d_0, d_1, latency: latency_c}
            }) -> tee();

        // Broadcast
        valid_placements -> for_each(|placement: PlacementType| {
            debug_println!("Broadcasting placement");
            txs_all_valid_placements.iter().for_each(|output|{output.send(placement.clone()).unwrap();})
        });
        // Round robin distribution across probe sides
        valid_placements -> enumerate() -> for_each(|(i, placement): (usize, _)|{
            debug_println!("Distributing placement");
            let output_index = i % txs_probe_valid_placements.len();
            txs_probe_valid_placements[output_index].send(placement).unwrap();
        });

    }
}

fn create_find_dominated_hf<'a>(
    rx_all_valid_placements: UnboundedReceiverStream<PlacementType>,
    rx_probe_valid_placements: UnboundedReceiverStream<PlacementType>,
    tx_dominated_placements: UnboundedSender<PlacementType>,
) -> Hydroflow<'a> {
    hydroflow_syntax! {
        source_stream(rx_all_valid_placements) -> [0]compare_all_to_all;
        source_stream(rx_probe_valid_placements) -> [1]compare_all_to_all;

        compare_all_to_all = cross_join::<'static,'tick>();
        filtered_placements = compare_all_to_all
        //-> inspect(|_|{debug_println!("Filter dominated placement");})
        -> filter_map(|(p_0, p_1)|{
            // p_0 is dominated by p_1
            if lt_slice(&p_1.latency, &p_0.latency) {Some(p_0)}
            else {None}
        });
        filtered_placements
            -> inspect(|_|{debug_println!("Forwarding dominated placement");})
            -> for_each(|p|{tx_dominated_placements.send(p).unwrap();});
    }
}

fn create_difference_hf<'a>(
    rx_all_placements: UnboundedReceiverStream<PlacementType>,
    rx_dominated_placements: UnboundedReceiverStream<PlacementType>,
    tx_optimal_count: UnboundedSender<usize>,
) -> Hydroflow<'a> {
    hydroflow_syntax! {
        // Insert all placements into positive/build side
        source_stream(rx_all_placements)
            -> inspect(|_|{debug_println!("Received placement");})
            -> [pos]optimal_placements;
        // Remove dominated placements from positive side
        source_stream(rx_dominated_placements)
            -> inspect(|_|{debug_println!("Received dominated placement")})
            -> [neg]optimal_placements;
        optimal_placements = difference::<'tick,'static>();
        optimal_placements
            -> inspect(|_|{debug_println!("Found optimal placement");})
            -> map(|_|1) -> reduce::<'static>(|accum: &mut _, elem| {*accum += elem})
            -> for_each(|acc|{
                debug_println!("Number of optimal placements: {}", acc);
                tx_optimal_count.send(acc).unwrap();
        });
    }
}

pub(crate) async fn create_and_run_pipelined_hf<'a>(
    expected_optimal_count: &'a usize,
    num_d: &'a usize,
    num_c: &'a usize,
    dist: &'a [usize],
    latency: &'a [LatencyType],
    dist_constraint: &'a usize,
    num_filter_threads: &'a usize,
    num_dominated_threads: &'a usize,
) -> std::time::Duration {
    /* Pipeline structure
    Macro:
    data_centers -> filter_valid_placements -> find_dominated_placements -> [neg]difference
                                            ->                              [pos]difference

    Micro:
    (1) data_centers -> broadcast-join (1:filter_threads) -> valid_placements ->
            broadcast-join (filter_threads:dominated_threads) -> find_dominated_placements ->
        incast (dominated_threads:1) -> difference -> result_count
     */

    // Channels
    // data_centers: Broadcast build and probe channels, num_filter_threads
    let (tx_data_center_build_vec, rx_data_center_build_vec): (Vec<_>, Vec<_>) = (0
        ..*num_filter_threads)
        .map(|_| hydroflow::util::unbounded_channel())
        .unzip();
    let (tx_data_center_probe_vec, rx_data_center_probe_vec): (Vec<_>, Vec<_>) = (0
        ..*num_filter_threads)
        .map(|_| hydroflow::util::unbounded_channel())
        .unzip();

    // valid_placements: Broadcast build and probe channels, num_dominated_threads
    let (tx_valid_placements_build_vec, rx_valid_placements_build_vec): (Vec<_>, Vec<_>) = (0
        ..*num_dominated_threads)
        .map(|_| hydroflow::util::unbounded_channel())
        .unzip();
    let (tx_valid_placements_probe_vec, rx_valid_placements_probe_vec): (Vec<_>, Vec<_>) = (0
        ..*num_dominated_threads)
        .map(|_| hydroflow::util::unbounded_channel())
        .unzip();

    // find_dominated_placements: Incast build and probe channels, 1
    let (tx_dominated_placements_build, rx_dominated_placements_build) =
        hydroflow::util::unbounded_channel();
    let (tx_dominated_placements_probe, rx_dominated_placements_probe) =
        hydroflow::util::unbounded_channel();

    // Collect number of optimal placements
    let (tx_optimal_count, rx_optimal_count) = hydroflow::util::unbounded_channel();

    // Copy latency and dist for each filter thread
    let latency_clones = (0..*num_filter_threads)
        .map(|_| latency.to_vec().clone())
        .collect::<Vec<_>>();
    let dist_clones = (0..*num_filter_threads)
        .map(|_| dist.to_vec().clone())
        .collect::<Vec<_>>();

    // Atomic flag for thread to wait for executing and when to stop
    let coordination_flag_main = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Create hydroflows and spawn threads

    // Data centers -> broadcast-join
    let coordination_flag_thread = coordination_flag_main.clone();
    let num_d_clone = num_d.clone();
    let data_center_pairs_thread = thread::spawn(move || {
        // Create hydroflow
        let mut hf = create_data_centers_hf(&num_d_clone, tx_data_center_build_vec, tx_data_center_probe_vec);

        // Coordinate start by waiting for true
        while coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == false {}

        debug_println!("Enumerating data centers");

        let mut worked = true;
        while worked && coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == true {
            worked = hf.run_available();
            debug_println!("Iteration of data_center_pairs returned {}", worked);
        }
    });

    // Broadcast-join -> valid_placements -> broadcast-join
    //                                    -> incast -> difference
    let valid_placements_threads = rx_data_center_build_vec
        .into_iter()
        .zip(rx_data_center_probe_vec.into_iter())
        .zip(latency_clones.into_iter())
        .zip(dist_clones.into_iter())
        .map(|(((rx_build, rx_probe), latency), dist)| {
            
            // Append tx_valid_placements_build_vec and tx_dominated_placements_build to the end of the vector
            let txs_build = tx_valid_placements_build_vec
                .clone()
                .into_iter()
                .chain(std::iter::once(tx_dominated_placements_build.clone()))
                .collect::<Vec<_>>();
            let txs_probe = tx_valid_placements_probe_vec.clone();

            let num_d = num_d.clone();
            let num_c = num_c.clone();
            let dist_constraint = dist_constraint.clone();

            let coordination_flag_thread = coordination_flag_main.clone();

            let executor = thread::spawn(move || {
                // Create hydroflow
                let mut hf = create_valid_placements_hf(
                    &num_d,
                    &num_c,
                    &dist_constraint,
                    &dist,
                    &latency,
                    rx_build,
                    rx_probe,
                    txs_build,
                    txs_probe,
                );

                // Coordinate start by waiting for true
                while coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == false {
                }

                debug_println!("Filtering valid placements");

                while coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == true {
                    let worked = hf.run_available();
                    debug_println!("Iteration of valid_placement returned {}", worked);
                }
            });

            executor
        })
        .collect::<Vec<_>>();

    // Broadcast-join -> find_dominated_placements -> incast
    /* let find_dominated_placements_threads = rx_valid_placements_build_vec
        .into_iter()
        .zip(rx_valid_placements_probe_vec.into_iter())
        .map(|(rx_build, rx_probe)| {
            
            let tx_dominated_placements_probe = tx_dominated_placements_probe.clone();
            let coordination_flag_thread = coordination_flag_main.clone();

            thread::spawn(move || {
                // Create hydroflow
                let mut hf =
                    create_find_dominated_hf(rx_probe, rx_build, tx_dominated_placements_probe);

                // Coordinate start by waiting for true
                while coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == false {
                }

                debug_println!("Finding dominated placements");

                // Run until flag is false
                while coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == true {
                    //let _worked = hf.run_available();
                    let _worked = hf.run_stratum();
                    debug_println!("Find dominated returned {}", _worked);
                }
            })
        }); */

    // Incast-join -> difference
    let coordination_flag_thread = coordination_flag_main.clone();
    let optimal_placements_thread = thread::spawn(move || {
        // Create hydroflow
        let mut optimal_placements_hf = create_difference_hf(
            rx_dominated_placements_build,
            rx_dominated_placements_probe,
            tx_optimal_count,
        );

        // Coordinate start by waiting for true
        while coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == false {}

        debug_println!("Computing optimal placements");

        while coordination_flag_thread.load(std::sync::atomic::Ordering::Relaxed) == true {
            let _worked = optimal_placements_hf.run_available();
            debug_println!("Iteration of dominated returned {}", _worked);
        }
    });

    let mut find_dominated_placements_hf = rx_valid_placements_build_vec
    .into_iter()
    .zip(rx_valid_placements_probe_vec.into_iter())
    .map(|(rx_build, rx_probe)| {
        
        let tx_dominated_placements_probe = tx_dominated_placements_probe.clone();
        let coordination_flag_thread = coordination_flag_main.clone();        
        // Create hydroflow
        
        create_find_dominated_hf(rx_probe, rx_build, tx_dominated_placements_probe)
    }).collect::<Vec<_>>();

    // Wait for threads to get ready
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Start all threads
    let start_time = Instant::now();
    coordination_flag_main.store(true, std::sync::atomic::Ordering::Release);

    // Sleep for 10 seconds
    //std::thread::sleep(std::time::Duration::from_secs(2));
    println!("Waking up");

    for hf in &mut find_dominated_placements_hf {
        let worked = hf.run_available();
        debug_println!("Iteration of find_dominated returned {}", worked);
    }
    
    let mut rx_optimal_count = rx_optimal_count;
    while let Some(element) = rx_optimal_count.next().await {
        println!("Found {} optimal placements", element);
        if element >= *expected_optimal_count {
            println!("Found {} optimal placements", element);
            break;
        }
    }

    coordination_flag_main.store(false, std::sync::atomic::Ordering::Release);

    let end_time = Instant::now();
    let duration = end_time - start_time;

    // Wait for threads to finish
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Joining threads");
    data_center_pairs_thread
        .join()
        .expect("Failed to join data center thread");
    valid_placements_threads.into_iter().for_each(|thread| {
        thread.join().expect("Failed to join valid thread");
    });
    /* find_dominated_placements_threads
        .into_iter()
        .for_each(|thread| {
            thread.join().expect("Failed to join dominated thread");
        }); */
    optimal_placements_thread
        .join()
        .expect("Failed to join optimal thread");

    return duration;
}
