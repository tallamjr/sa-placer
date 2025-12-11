//! Bookshelf format parser for loading real circuit benchmarks.
//!
//! The Bookshelf format is a standard format for VLSI placement benchmarks.
//! It consists of:
//! - .nodes file: defines cells and their types
//! - .nets file: defines hyperedge connectivity
//!
//! This module parses these files and converts them to our internal
//! FPGALayout and NetlistGraph representations.

use crate::{FPGALayout, FPGALayoutType, MacroType, NetlistGraph, NetlistNode};
use rustc_hash::FxHashMap;
use rustworkx_core::petgraph;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parsed node from a .nodes file
#[derive(Debug, Clone)]
struct BookshelfNode {
    name: String,
    #[allow(dead_code)]
    width: f32,
    #[allow(dead_code)]
    height: f32,
    is_terminal: bool,
}

/// Parsed net (hyperedge) from a .nets file
#[derive(Debug, Clone)]
struct BookshelfNet {
    pins: Vec<String>, // Node names connected by this net
}

/// Parse a Bookshelf .nodes file
fn parse_nodes_file(path: &Path) -> Result<(Vec<BookshelfNode>, usize, usize), String> {
    let file = File::open(path).map_err(|e| format!("Failed to open nodes file: {}", e))?;
    let reader = BufReader::new(file);

    let mut nodes = Vec::new();
    let mut num_nodes = 0;
    let mut num_terminals = 0;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') || line.starts_with("UCLA") {
            continue;
        }

        // Parse header lines
        if line.starts_with("NumNodes") {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() == 2 {
                num_nodes = parts[1].trim().parse().unwrap_or(0);
            }
            continue;
        }
        if line.starts_with("NumTerminals") {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() == 2 {
                num_terminals = parts[1].trim().parse().unwrap_or(0);
            }
            continue;
        }

        // Parse node lines: name width height [terminal]
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let name = parts[0].to_string();
            let width: f32 = parts[1].parse().unwrap_or(1.0);
            let height: f32 = parts[2].parse().unwrap_or(1.0);
            let is_terminal = parts.len() >= 4 && parts[3] == "terminal";

            nodes.push(BookshelfNode {
                name,
                width,
                height,
                is_terminal,
            });
        }
    }

    Ok((nodes, num_nodes, num_terminals))
}

/// Parse a Bookshelf .nets file
fn parse_nets_file(path: &Path) -> Result<Vec<BookshelfNet>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open nets file: {}", e))?;
    let reader = BufReader::new(file);

    let mut nets = Vec::new();
    let mut current_net: Option<BookshelfNet> = None;
    #[allow(unused_assignments)]
    let mut expected_pins = 0;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') || line.starts_with("UCLA") {
            continue;
        }

        // Skip NumPins header
        if line.starts_with("NumPins") || line.starts_with("NumNets") {
            continue;
        }

        // NetDegree starts a new net
        if line.starts_with("NetDegree") {
            // Save previous net if exists
            if let Some(net) = current_net.take() {
                if !net.pins.is_empty() {
                    nets.push(net);
                }
            }

            // Parse degree
            let parts: Vec<&str> = line.split(':').collect();
            expected_pins = if parts.len() == 2 {
                parts[1].trim().parse().unwrap_or(0)
            } else {
                // Format might be "NetDegree : N" or "NetDegree N"
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    parts[parts.len() - 1].parse().unwrap_or(0)
                } else {
                    0
                }
            };

            current_net = Some(BookshelfNet { pins: Vec::with_capacity(expected_pins) });
            continue;
        }

        // Parse pin line: node_name direction : x_offset y_offset
        if let Some(ref mut net) = current_net {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if !parts.is_empty() {
                net.pins.push(parts[0].to_string());
            }
        }
    }

    // Save last net
    if let Some(net) = current_net {
        if !net.pins.is_empty() {
            nets.push(net);
        }
    }

    Ok(nets)
}

/// Statistics about a loaded benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    pub name: String,
    pub num_nodes: usize,
    pub num_terminals: usize,
    pub num_movable: usize,
    pub num_nets: usize,
    pub num_pins: usize,
    pub grid_size: u32,
}

/// Load a Bookshelf benchmark and convert to our internal format.
///
/// Since Bookshelf is for ASIC placement and we're doing FPGA placement,
/// we make some simplifications:
/// - All movable cells become CLBs
/// - All terminals become IOs placed on the border
/// - Grid size is computed to fit all cells
///
/// Returns (FPGALayout, NetlistGraph, BenchmarkStats)
pub fn load_bookshelf_benchmark(
    nodes_path: &Path,
    nets_path: &Path,
) -> Result<(FPGALayout, NetlistGraph, BenchmarkStats), String> {
    // Parse files
    let (nodes, _, _) = parse_nodes_file(nodes_path)?;
    let nets = parse_nets_file(nets_path)?;

    let num_terminals = nodes.iter().filter(|n| n.is_terminal).count();
    let num_movable = nodes.len() - num_terminals;

    // Create node name to index mapping
    let mut name_to_idx: FxHashMap<String, petgraph::graph::NodeIndex> = FxHashMap::default();

    // Calculate grid size: we need enough CLB slots for movable cells
    // and enough IO slots around the border for terminals
    // Grid has (width-2) * (height-2) CLB slots and 2*(width + height - 4) IO slots
    let grid_size = compute_grid_size(num_movable, num_terminals);

    // Create FPGA layout
    let layout = build_benchmark_fpga_layout(grid_size, grid_size);

    // Create netlist graph
    let mut graph = petgraph::graph::DiGraph::<NetlistNode, ()>::new();

    // Add nodes
    for (id, node) in nodes.iter().enumerate() {
        let macro_type = if node.is_terminal {
            MacroType::IO
        } else {
            MacroType::CLB
        };

        let idx = graph.add_node(NetlistNode {
            id: id as u32,
            macro_type,
        });
        name_to_idx.insert(node.name.clone(), idx);
    }

    // Add edges from nets (hyperedges)
    // For a hyperedge connecting nodes {A, B, C, D}, we create edges:
    // A->B, A->C, A->D (star model with first node as center)
    let mut num_pins = 0;
    for net in &nets {
        num_pins += net.pins.len();
        if net.pins.len() < 2 {
            continue;
        }

        // Get indices for all pins in this net
        let pin_indices: Vec<_> = net
            .pins
            .iter()
            .filter_map(|name| name_to_idx.get(name).copied())
            .collect();

        // Create star topology: connect first pin to all others
        if pin_indices.len() >= 2 {
            let center = pin_indices[0];
            for &other in &pin_indices[1..] {
                // Add edge if it doesn't exist (avoid duplicates)
                if !graph.contains_edge(center, other) {
                    graph.add_edge(center, other, ());
                }
            }
        }
    }

    let netlist = NetlistGraph { graph };

    let stats = BenchmarkStats {
        name: nodes_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        num_nodes: nodes.len(),
        num_terminals,
        num_movable,
        num_nets: nets.len(),
        num_pins,
        grid_size,
    };

    Ok((layout, netlist, stats))
}

/// Compute grid size needed to fit the benchmark
fn compute_grid_size(num_movable: usize, num_terminals: usize) -> u32 {
    // We need:
    // - (grid_size - 2)^2 >= num_movable (for CLB area)
    // - 4 * (grid_size - 1) >= num_terminals (for IO perimeter, excluding corners)

    // Solve for CLB requirement
    let clb_grid = ((num_movable as f64).sqrt().ceil() as u32) + 2;

    // Solve for IO requirement: perimeter = 4 * (grid_size - 1)
    // So grid_size >= num_terminals / 4 + 1
    let io_grid = (num_terminals as u32 / 4) + 2;

    // Take maximum and add some margin
    let grid = clb_grid.max(io_grid).max(8);

    // Round up to nice number
    ((grid + 7) / 8) * 8
}

/// Build an FPGA layout for benchmark testing
/// Similar to build_simple_fpga_layout but without BRAM columns
fn build_benchmark_fpga_layout(width: u32, height: u32) -> FPGALayout {
    let mut layout = FPGALayout::new(width, height);

    // IO on border
    layout.config_border(FPGALayoutType::MacroType(MacroType::IO));
    layout.config_corners(FPGALayoutType::EMPTY);

    // All interior is CLB (no BRAM for simplicity since benchmark doesn't have BRAM)
    layout.config_repeat(
        1,
        1,
        width - 2,
        height - 2,
        1,
        1,
        FPGALayoutType::MacroType(MacroType::CLB),
    );

    assert!(layout.valid());
    layout
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_grid_size() {
        // 752 movable cells, 81 terminals (primary1)
        let grid = compute_grid_size(752, 81);
        assert!(grid >= 30); // sqrt(752) ~ 27.4, so need at least 30

        // Check CLB capacity
        let clb_slots = (grid - 2) * (grid - 2);
        assert!(clb_slots >= 752);

        // Check IO capacity
        let io_slots = 4 * (grid - 1);
        assert!(io_slots >= 81);
    }

    #[test]
    fn test_build_benchmark_layout() {
        let layout = build_benchmark_fpga_layout(32, 32);
        let summary = layout.count_summary();

        // Should have IO on border, CLB inside
        let io_count = *summary
            .get(&FPGALayoutType::MacroType(MacroType::IO))
            .unwrap_or(&0);
        let clb_count = *summary
            .get(&FPGALayoutType::MacroType(MacroType::CLB))
            .unwrap_or(&0);

        // Border: top row (32) + bottom row (32) + left col (30) + right col (30) = 124
        // But corners are set to EMPTY, so 124 - 4 = 120 IO slots
        assert_eq!(io_count, 120);

        // Interior: 30 * 30 = 900 CLB slots
        assert_eq!(clb_count, 30 * 30);
    }
}
