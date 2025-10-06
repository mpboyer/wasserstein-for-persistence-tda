mod structs;
use crate::structs::*;

mod auction;
pub use auction::AuctionAlgorithm;
fn main() {
    println!("Hello World!");
}

mod benchmarks;
#[cfg(test)]
mod tests;
