use std::hash::{Hash, Hasher};

const USE_LT_TYPE: usize = 0;

pub(crate) type DataCenterIndexType = u16;
pub(crate) type LatencyType = f32;
//type LatencyArray = [(LatencyType, LatencyType); NO_DATA_CENTERS];
pub(crate) type LatencyArray = Vec<(LatencyType, LatencyType)>;
pub(crate) type PlacementType = Placement<DataCenterIndexType, LatencyArray>;

#[derive(Debug, Clone)]
pub(crate) struct Placement<DataCenterIndexType, LatencyArray>
where
    DataCenterIndexType: Hash,
{
    pub d_0: DataCenterIndexType,
    pub d_1: DataCenterIndexType,
    pub latency: LatencyArray,
}

impl<DataCenterIndexType, LatencyArray> Placement<DataCenterIndexType, LatencyArray>
where
    DataCenterIndexType: Hash + std::cmp::PartialEq,
{
    pub fn has_same_dcs(&self, other: &Self) -> bool {
        self.d_0 == other.d_0 && self.d_1 == other.d_1
    }
}

impl<DataCenterIndexType, LatencyArray> PartialEq for Placement<DataCenterIndexType, LatencyArray>
where
    DataCenterIndexType: Hash + std::cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.has_same_dcs(other)
    }
}

impl<DataCenterIndexType, LatencyArray> Eq for Placement<DataCenterIndexType, LatencyArray> where
    DataCenterIndexType: Hash + std::cmp::PartialEq
{
}

impl<DataCenterIndexType, LatencyArray> Hash for Placement<DataCenterIndexType, LatencyArray>
where
    DataCenterIndexType: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.d_0.hash(state);
        self.d_1.hash(state);
    }
}

fn lt_slice_loop(a: &[(f32,f32)], b: &[(f32,f32)]) -> bool {
    let mut i = 0;
    while i < a.len() {
        if a[i] >= b[i] {
            return false;
        }
        i += 1;
    }
    true
}

fn lt_slice_iter_all(a: &[(f32,f32)], b: &[(f32,f32)]) -> bool {
    a.iter().zip(b.iter()).all(|(a, b)| a < b)
}

fn lt_slice_iter_reduce(a: &[(f32,f32)], b: &[(f32,f32)]) -> bool {
    a.iter().zip(b.iter())
        .map(|(a,b)| *a < *b)
        .reduce(|a,b| a && b).unwrap()
}

pub(crate) fn lt_slice(a: &[(f32,f32)], b: &[(f32,f32)]) -> bool {
    assert_eq!(a.len(), b.len());

    if USE_LT_TYPE == 0 {
        lt_slice_iter_all(a, b)
    } else if USE_LT_TYPE == 1 {
        lt_slice_iter_reduce(a, b)
    } else {
        lt_slice_loop(a, b)
    }
}

pub(crate) enum HfImplementation {
    Monolith,
    Monolith_Enumeration,
    Pipelined,
}

impl std::fmt::Display for HfImplementation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HfImplementation::Monolith => write!(f, "Monolith"),
            HfImplementation::Monolith_Enumeration => write!(f, "Monolith - Enumeration only"),
            HfImplementation::Pipelined => write!(f, "Pipelined"),
        }
    }
}

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        println!($($arg)*)
    };
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        {}// Noop in release
    };
}

pub(crate) fn compute_combinations(n: usize, r: usize) -> usize {
    if r > n {
        0
    } else {
        (1..=r.min(n - r)).fold(1, |acc, val| acc * (n - val + 1) / val)
    }
}