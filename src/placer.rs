use std::process::Command;

use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use rustworkx_core::petgraph::visit::EdgeRef;
use tempfile::tempdir;

use super::fpga_layout::*;
use super::netlist::*;

use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use itertools::Itertools;

#[derive(Debug, Clone, Copy)]
pub enum PlacementAction {
    Move,
    Swap,
    MoveDirected,
}

#[derive(Debug, Clone)]
pub struct PlacementSolution<'a> {
    pub layout: &'a FPGALayout,
    pub netlist: &'a NetlistGraph,
    pub solution_map: FxHashMap<NetlistNode, FPGALayoutCoordinate>,
}

impl<'a> PlacementSolution<'a> {
    pub fn new(layout: &'a FPGALayout, netlist: &'a NetlistGraph) -> Self {
        Self {
            layout,
            netlist,
            solution_map: FxHashMap::default(),
        }
    }

    pub fn action_move(&mut self) {
        let mut rng = rand::thread_rng();

        // Randomly select a node
        let node = match self.netlist.all_nodes().choose(&mut rng) {
            Some(n) => *n,
            None => return,
        };

        // Get possible sites
        let possible_sites = self.get_possible_sites(node.macro_type);

        // Return if there are no possible sites
        if possible_sites.is_empty() {
            return;
        }

        // Randomly select a location
        let location = match possible_sites.choose(&mut rng) {
            Some(l) => *l,
            None => return,
        };

        self.solution_map.insert(*node, location);
    }

    pub fn action_swap(&mut self) {
        let mut rng = rand::thread_rng();

        // Randomly select a node (node_a)
        let node_a = match self.netlist.all_nodes().choose(&mut rng) {
            Some(n) => *n,
            None => return,
        };

        // Filter nodes of the same type as node_a
        let nodes_same_type = self
            .netlist
            .all_nodes()
            .into_iter()
            .filter(|node| node.macro_type == node_a.macro_type)
            .collect_vec();

        // If no nodes of the same type, return
        if nodes_same_type.is_empty() {
            return;
        }

        // Randomly select another node (node_b) of the same type
        let node_b = match nodes_same_type.choose(&mut rng) {
            Some(n) => *n,
            None => return,
        };

        // Clone the locations first to avoid borrowing issues
        let loc_a = self.solution_map.get(node_a).cloned();
        let loc_b = self.solution_map.get(node_b).cloned();

        // Perform the swap
        if let (Some(loc_a), Some(loc_b)) = (loc_a, loc_b) {
            self.solution_map.insert(*node_a, loc_b);
            self.solution_map.insert(*node_b, loc_a);
        }
    }

    pub fn action_move_directed(&mut self) {
        let node_count = self.netlist.graph.node_count() as u32;

        if node_count == 0 {
            panic!("No nodes in netlist; cannot compute mean for MOVE_DIRECTED");
        }
        let x_mean = self
            .netlist
            .all_nodes()
            .iter()
            .map(|node| self.solution_map.get(node).unwrap().x)
            .sum::<u32>()
            / node_count;

        let y_mean = self
            .netlist
            .all_nodes()
            .iter()
            .map(|node| self.solution_map.get(node).unwrap().y)
            .sum::<u32>()
            / node_count;

        let mut rng = rand::thread_rng();

        // pick a random node
        let node = self.netlist.all_nodes()[rng.gen_range(0..node_count as usize)];

        let valid_locations = self.get_possible_sites(node.macro_type);
        let valid_closest_location = valid_locations
            .iter()
            .min_by(|a, b| {
                let a_distance =
                    (a.x as i32 - x_mean as i32).abs() + (a.y as i32 - y_mean as i32).abs();
                let b_distance =
                    (b.x as i32 - x_mean as i32).abs() + (b.y as i32 - y_mean as i32).abs();
                a_distance.cmp(&b_distance)
            })
            .unwrap();

        // if the new location is futher away from the mean than the current location, return
        let current_location = self.solution_map.get(node).unwrap();
        let current_distance = (current_location.x as i32 - x_mean as i32).abs()
            + (current_location.y as i32 - y_mean as i32).abs();
        let new_distance = (valid_closest_location.x as i32 - x_mean as i32).abs()
            + (valid_closest_location.y as i32 - y_mean as i32).abs();
        if new_distance > current_distance {
            return;
        }

        self.solution_map.insert(*node, *valid_closest_location);
    }

    pub fn action(&mut self, action: PlacementAction) {
        match action {
            PlacementAction::Move => self.action_move(),
            PlacementAction::Swap => self.action_swap(),
            PlacementAction::MoveDirected => self.action_move_directed(),
        }
    }

    /// Bounding-box cost: sum of Manhattan distances for all edges
    ///
    /// This is the original cost function treating each edge independently.
    /// Time complexity: O(E) where E is the number of edges.
    pub fn cost_bb(&self) -> f32 {
        let mut cost = 0;

        for edge in self.netlist.graph.edge_references() {
            let source_idx = edge.source();
            let target_idx = edge.target();

            let source = self.netlist.graph.node_weight(source_idx).unwrap();
            let target = self.netlist.graph.node_weight(target_idx).unwrap();

            let source_location = self.solution_map.get(source).unwrap();
            let target_location = self.solution_map.get(target).unwrap();

            let x_distance = source_location.x.abs_diff(target_location.x);
            let y_distance = source_location.y.abs_diff(target_location.y);

            let distance = x_distance + y_distance;
            cost += distance;
        }

        cost as f32
    }

    /// Half-Perimeter Wirelength (HPWL) cost function
    ///
    /// Groups edges by source node to form pseudo-nets and computes
    /// the half-perimeter of the bounding box for each net.
    ///
    /// HPWL = sum over nets of (max_x - min_x + max_y - min_y)
    ///
    /// This is more accurate for multi-fanout nets than simple edge-based cost.
    /// Time complexity: O(E) where E is the number of edges.
    pub fn cost_hpwl(&self) -> f32 {
        use rustworkx_core::petgraph::graph::NodeIndex;

        // Group nodes by their source (forming pseudo-nets)
        // Each source node and its destinations form a net
        let mut nets: FxHashMap<NodeIndex, Vec<FPGALayoutCoordinate>> = FxHashMap::default();

        for edge in self.netlist.graph.edge_references() {
            let source_idx = edge.source();
            let target_idx = edge.target();

            let source = self.netlist.graph.node_weight(source_idx).unwrap();
            let target = self.netlist.graph.node_weight(target_idx).unwrap();

            let source_loc = *self.solution_map.get(source).unwrap();
            let target_loc = *self.solution_map.get(target).unwrap();

            // Add source and target to this net
            let net = nets.entry(source_idx).or_default();
            if net.is_empty() {
                net.push(source_loc);
            }
            net.push(target_loc);
        }

        let mut total_hpwl: u32 = 0;

        for coords in nets.values() {
            if coords.len() < 2 {
                continue;
            }

            let x_min = coords.iter().map(|c| c.x).min().unwrap();
            let x_max = coords.iter().map(|c| c.x).max().unwrap();
            let y_min = coords.iter().map(|c| c.y).min().unwrap();
            let y_max = coords.iter().map(|c| c.y).max().unwrap();

            total_hpwl += (x_max - x_min) + (y_max - y_min);
        }

        total_hpwl as f32
    }

    /// Incremental cost update after moving a single node
    ///
    /// Instead of recomputing the full cost, this computes only the
    /// delta caused by moving one node. Much faster for SA where
    /// we make one move per iteration.
    ///
    /// Returns the new cost given the current cost and the move.
    pub fn incremental_cost_delta(
        &self,
        node: &NetlistNode,
        old_location: &FPGALayoutCoordinate,
        new_location: &FPGALayoutCoordinate,
    ) -> i32 {
        use rustworkx_core::petgraph::Direction;

        let mut delta: i32 = 0;

        // Find the node index for this node
        let node_idx = self
            .netlist
            .graph
            .node_indices()
            .find(|&idx| self.netlist.graph[idx] == *node);

        if let Some(idx) = node_idx {
            // Check all edges connected to this node (both incoming and outgoing)
            for neighbor_idx in self
                .netlist
                .graph
                .neighbors_directed(idx, Direction::Outgoing)
                .chain(self.netlist.graph.neighbors_directed(idx, Direction::Incoming))
            {
                let neighbor = &self.netlist.graph[neighbor_idx];
                let neighbor_loc = self.solution_map.get(neighbor).unwrap();

                // Old contribution
                let old_dist = old_location.x.abs_diff(neighbor_loc.x) as i32
                    + old_location.y.abs_diff(neighbor_loc.y) as i32;

                // New contribution
                let new_dist = new_location.x.abs_diff(neighbor_loc.x) as i32
                    + new_location.y.abs_diff(neighbor_loc.y) as i32;

                delta += new_dist - old_dist;
            }
        }

        delta
    }

    pub fn render_svg(&self) -> String {
        let mut svg = String::new();

        svg.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" style=\"background-color:white\" viewBox=\"0 0 {} {}\">\n",
            self.layout.width * 100,
            self.layout.height * 100
        ));

        // draw the white background manually
        svg.push_str(&format!(
            "\t<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"white\"/>\n",
            0,
            0,
            self.layout.width * 100,
            self.layout.height * 100
        ));

        // draw boxes for each location
        for x in 0..self.layout.width {
            for y in 0..self.layout.height {
                let layout_type = self.layout.get(&FPGALayoutCoordinate::new(x, y)).unwrap();

                let color = match layout_type {
                    FPGALayoutType::MacroType(MacroType::CLB) => "red",
                    FPGALayoutType::MacroType(MacroType::DSP) => "blue",
                    FPGALayoutType::MacroType(MacroType::BRAM) => "green",
                    FPGALayoutType::MacroType(MacroType::IO) => "yellow",
                    FPGALayoutType::EMPTY => "gray",
                };

                svg.push_str(&format!(
                    "\t<rect x=\"{}\" y=\"{}\" width=\"100\" height=\"100\" fill=\"{}\" fill-opacity=\"0.25\" stroke=\"black\" stroke-width=\"2\"/>\n",
                    x * 100,
                    y * 100,
                    color
                ));
            }
        }

        // draw boxes for each netlist node
        for (node, location) in self.solution_map.iter() {
            let color = match node.macro_type {
                MacroType::CLB => "red",
                MacroType::DSP => "blue",
                MacroType::BRAM => "green",
                MacroType::IO => "yellow",
            };

            svg.push_str(&format!(
                "\t<rect x=\"{}\" y=\"{}\" width=\"100\" height=\"100\" fill=\"{}\"/>\n",
                location.x * 100,
                location.y * 100,
                color
            ));

            svg.push_str(&format!(
                "\t<text x=\"{}\" y=\"{}\" fill=\"black\" font-size=\"50\">{}</text>\n",
                location.x * 100 + 10,
                location.y * 100 + 70,
                node.id
            ));
        }

        // draw lines for each netlist edge
        for edge in self.netlist.graph.edge_references() {
            let source_idx = edge.source();
            let target_idx = edge.target();

            let source = self.netlist.graph.node_weight(source_idx).unwrap();
            let target = self.netlist.graph.node_weight(target_idx).unwrap();

            let source_location = self.solution_map.get(source).unwrap();
            let target_location = self.solution_map.get(target).unwrap();

            svg.push_str(&format!(
                "\t<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" style=\"stroke:rgb(0,0,0);stroke-width:4\" />\n",
                source_location.x * 100 + 50,
                source_location.y * 100 + 50,
                target_location.x * 100 + 50,
                target_location.y * 100 + 50
            ));
        }

        svg.push_str("</svg>\n");

        svg
    }

    pub fn get_unplaced_nodes(&self) -> Vec<NetlistNode> {
        let mut unplaced_nodes: Vec<NetlistNode> = Vec::new();

        for node in self.netlist.graph.node_weights() {
            if !self.solution_map.contains_key(node) {
                unplaced_nodes.push(*node);
            }
        }

        unplaced_nodes
    }

    pub fn get_possible_sites(&self, macro_type: MacroType) -> Vec<FPGALayoutCoordinate> {
        let mut possible_sites = Vec::new();

        let mut placed_locations = FxHashSet::default();
        for location in self.solution_map.values() {
            placed_locations.insert(*location);
        }

        for x in 0..self.layout.width {
            for y in 0..self.layout.height {
                // check if the location is unplaced
                if placed_locations.contains(&FPGALayoutCoordinate::new(x, y)) {
                    continue;
                }

                let layout_type = self
                    .layout
                    .map
                    .get(&FPGALayoutCoordinate::new(x, y))
                    .unwrap();

                match layout_type {
                    FPGALayoutType::MacroType(layout_macro_type) => {
                        if layout_macro_type == &macro_type {
                            possible_sites.push(FPGALayoutCoordinate::new(x, y));
                        }
                    }
                    FPGALayoutType::EMPTY => {}
                }
            }
        }

        possible_sites
    }

    pub fn place_node(&mut self, node: NetlistNode, location: FPGALayoutCoordinate) {
        self.solution_map.insert(node, location);
    }

    pub fn valid(&self) -> bool {
        // let netlist_nodes: Vec<NetlistNode> = self.netlist.all_nodes();
        let netlist_nodes = self.netlist.graph.node_weights().collect_vec();
        let netlist_nodes_ids = netlist_nodes
            .iter()
            .map(|node| node.id)
            .collect::<Vec<u32>>();

        // Check that all the nodes in the netlist are in the solution map
        for node in self.netlist.graph.node_weights() {
            if !self.solution_map.contains_key(node) {
                return false;
            }
        }

        // Check that all the nodes in the solution map are in the netlist
        for node in self.solution_map.keys() {
            if !netlist_nodes_ids.contains(&node.id) {
                return false;
            }
        }

        // check that each location in the layout is only used once
        let mut used_locations = FxHashSet::default();
        for location in self.solution_map.values() {
            if used_locations.contains(location) {
                return false;
            }
            used_locations.insert(*location);
        }

        // check that each node in the netlist is only used once
        let mut used_nodes = FxHashSet::default();
        for node in self.solution_map.keys() {
            if used_nodes.contains(node) {
                return false;
            }
            used_nodes.insert(*node);
        }

        // check that nodes are placed on the correct type of macro
        for (node, location) in self.solution_map.iter() {
            let layout_type: FPGALayoutType = self.layout.get(location).unwrap();

            match layout_type {
                FPGALayoutType::MacroType(macro_type) => {
                    if node.macro_type != macro_type {
                        return false;
                    }
                }
                FPGALayoutType::EMPTY => return false,
            }
        }

        true
    }
}

pub enum InitialPlacerMethod {
    Random,
    Greedy,
}

pub fn gen_random_placement<'a>(
    layout: &'a FPGALayout,
    netlist: &'a NetlistGraph,
) -> PlacementSolution<'a> {
    let mut solution = PlacementSolution::new(layout, netlist);

    let mut rng = rand::thread_rng();

    let count_summary_layout = solution.layout.count_summary();
    let count_summary_netlist = solution.netlist.count_summary();

    for &macro_type in &[
        MacroType::CLB,
        MacroType::DSP,
        MacroType::BRAM,
        MacroType::IO,
    ] {
        assert!(
            count_summary_layout
                .get(&FPGALayoutType::MacroType(macro_type))
                .unwrap()
                >= count_summary_netlist.get(&macro_type).unwrap()
        );
    }

    for node in solution.netlist.all_nodes() {
        let possible_sites = solution.get_possible_sites(node.macro_type);
        let location: FPGALayoutCoordinate = possible_sites[rng.gen_range(0..possible_sites.len())];
        solution.place_node(*node, location);
    }

    assert!(solution.valid());

    solution
}

pub fn gen_greedy_placement<'a>(
    layout: &'a FPGALayout,
    netlist: &'a NetlistGraph,
) -> PlacementSolution<'a> {
    // place nodes in the first spot in the layout closest to the origin (0,0) which is the top left corner

    let mut solution = PlacementSolution::new(layout, netlist);

    let count_summary_layout = solution.layout.count_summary();
    let count_summary_netlist = solution.netlist.count_summary();

    for &macro_type in &[
        MacroType::CLB,
        MacroType::DSP,
        MacroType::BRAM,
        MacroType::IO,
    ] {
        assert!(
            count_summary_layout
                .get(&FPGALayoutType::MacroType(macro_type))
                .unwrap()
                >= count_summary_netlist.get(&macro_type).unwrap()
        );
    }

    let nodes = solution.netlist.all_nodes();
    for node in nodes {
        let possible_sites = solution.get_possible_sites(node.macro_type);
        // get the site with the min manhattan distance to the origin (0,0)
        let location = possible_sites
            .iter()
            .min_by(|a, b| {
                let a_distance: u32 = a.x + a.y;
                let b_distance = b.x + b.y;
                a_distance.cmp(&b_distance)
            })
            .unwrap();

        solution.place_node(*node, *location);
    }

    assert!(solution.valid());

    solution
}

pub fn gen_initial_placement<'a>(
    layout: &'a FPGALayout,
    netlist: &'a NetlistGraph,
    method: InitialPlacerMethod,
) -> PlacementSolution<'a> {
    match method {
        InitialPlacerMethod::Random => gen_random_placement(layout, netlist),
        InitialPlacerMethod::Greedy => gen_greedy_placement(layout, netlist),
    }
}

#[derive(Clone)]
pub struct Renderer {
    pub svg_renders: Vec<String>,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            svg_renders: Vec::new(),
        }
    }

    pub fn add_frame(&mut self, svg: String) {
        self.svg_renders.push(svg);
    }

    pub fn render_to_video(
        self,
        output_name: &str,
        output_dir: &str,
        framerate: f64,
        every_n_frames: usize,
        make_gif: bool,
    ) {
        let dir = tempdir().unwrap();
        let frame_dir = dir.path().join("frames");
        std::fs::create_dir(&frame_dir).unwrap();

        let mut input_frames_svg_paths = Vec::new();

        for (frame_number, svg) in self.svg_renders.iter().enumerate() {
            if frame_number % every_n_frames != 0 {
                continue;
            }
            let frame_fp = frame_dir.join(format!("frame_{}.svg", frame_number));
            input_frames_svg_paths.push(frame_fp.clone());
            std::fs::write(&frame_fp, svg).expect("Unable to write file");
        }

        // rename the every_n_frames frames to be sequential to not confuse ffmpeg
        let mut input_frames_svg_paths_renumbered = Vec::new();
        for (frame_number, svg_fp) in input_frames_svg_paths.iter().enumerate() {
            let new_fp = frame_dir.join(format!("frame_{}.svg", frame_number));
            std::fs::rename(svg_fp, &new_fp).expect("Unable to rename file");
            input_frames_svg_paths_renumbered.push(new_fp);
        }

        // convert the frames to pngs
        input_frames_svg_paths_renumbered
            .par_iter()
            .for_each(|svg_fp| {
                let png_fp = svg_fp.with_extension("png");
                println!("Converting {:?} to {:?} ... ", svg_fp, png_fp);
                let _output: std::process::Output = std::process::Command::new("magick")
                    .arg("convert")
                    .arg("-size")
                    .arg("800x800")
                    .arg(svg_fp)
                    .arg(png_fp)
                    .output()
                    .expect("failed to execute magick");
            });

        // use ffmpeg to convert the frames to a video
        let mut ffmpeg_cmd = Command::new("ffmpeg");
        ffmpeg_cmd.arg("-y");
        ffmpeg_cmd.arg("-framerate");
        ffmpeg_cmd.arg(format!("{}", framerate));
        ffmpeg_cmd.arg("-i");
        ffmpeg_cmd.arg(frame_dir.join("frame_%d.png").to_str().unwrap());
        ffmpeg_cmd.arg("-c:v");
        ffmpeg_cmd.arg("libx264");
        ffmpeg_cmd.arg("-pix_fmt");
        ffmpeg_cmd.arg("yuv420p");
        ffmpeg_cmd.arg(format!("{}/{}.mp4", output_dir, output_name));

        let child = ffmpeg_cmd.spawn().expect("failed to execute ffmpeg");
        child.wait_with_output().expect("failed to wait on ffmpeg");

        if make_gif {
            // use ffmpeg to convert the frames to a gif
            let mut ffmpeg_cmd = Command::new("ffmpeg");
            ffmpeg_cmd.arg("-y");
            ffmpeg_cmd.arg("-framerate");
            ffmpeg_cmd.arg(format!("{}", framerate));
            ffmpeg_cmd.arg("-i");
            ffmpeg_cmd.arg(frame_dir.join("frame_%d.png").to_str().unwrap());

            // Optimize gif size using rescaling and color pallet reduction
            ffmpeg_cmd.arg("-filter_complex");
            ffmpeg_cmd.arg("scale=iw/2:-1,split [a][b];[a] palettegen=stats_mode=diff:max_colors=32[p]; [b][p] paletteuse=dither=bayer");

            ffmpeg_cmd.arg(format!("{}/{}.gif", output_dir, output_name));

            let child = ffmpeg_cmd.spawn().expect("failed to execute ffmpeg");
            child.wait_with_output().expect("failed to wait on ffmpeg");
        }
    }
}

pub struct PlacerOutput<'a> {
    pub initial_solution: PlacementSolution<'a>,
    pub final_solution: PlacementSolution<'a>,
    pub x_steps: Vec<u32>,
    pub y_cost: Vec<f32>,
    pub renderer: Option<Renderer>,
}

pub fn fast_sa_placer(
    initial_solution: PlacementSolution,
    n_steps: u32,
    n_neighbors: usize, // number of neighbors to explore at each step
    verbose: bool,
    render: bool,
) -> PlacerOutput {
    let mut renderer = Renderer::new();

    let mut current_solution = initial_solution.clone();

    let mut rng = rand::thread_rng();
    let actions: &[PlacementAction] = &[
        PlacementAction::Move,
        PlacementAction::Swap,
        PlacementAction::MoveDirected,
    ];

    let mut x_steps = Vec::new();
    let mut y_cost = Vec::new();

    for _i in 0..n_steps {
        x_steps.push(_i);
        y_cost.push(current_solution.cost_bb());
        if render {
            renderer.add_frame(current_solution.render_svg());
        }

        // randomly select actions
        let actions: Vec<_> = actions.choose_multiple(&mut rng, n_neighbors).collect();

        let new_solutions: Vec<_> = actions
            // .into_par_iter()
            .into_iter()
            .map(|action| {
                let mut new_solution = current_solution.clone();
                new_solution.action(*action);
                new_solution
            })
            .collect();

        let best_solution = new_solutions
            .iter()
            .min_by(|sol1, sol2| {
                (sol1.cost_bb() - current_solution.cost_bb())
                    .partial_cmp(&(sol2.cost_bb() - current_solution.cost_bb()))
                    .unwrap()
            })
            .unwrap();

        let best_delta = best_solution.cost_bb() - current_solution.cost_bb();
        let mut delta = 0.0;
        if best_delta < 0.0 {
            current_solution = best_solution.clone();
            delta = best_delta;
        }

        if verbose && _i % 10 == 0 {
            println!("Current Itteration: {:?}", _i);
            println!("Delta Cost: {:?}", delta);
            println!("Current Cost: {:?}", current_solution.cost_bb());
        }
    }

    if render {
        renderer.add_frame(current_solution.render_svg());
    }

    PlacerOutput {
        initial_solution: initial_solution.clone(),
        final_solution: current_solution.clone(),
        x_steps,
        y_cost,
        renderer: if render { Some(renderer) } else { None },
    }
}

// ============================================================================
// TRUE SIMULATED ANNEALING IMPLEMENTATION
// ============================================================================

/// Cooling schedule strategies for simulated annealing
#[derive(Debug, Clone, Copy)]
pub enum CoolingSchedule {
    /// Geometric cooling: T_{k+1} = alpha * T_k
    /// Recommended alpha in [0.9, 0.99]
    Geometric { alpha: f64 },

    /// Linear cooling: T_{k+1} = T_k - beta
    /// beta is computed from initial_temp / n_steps
    Linear { beta: f64 },

    /// Logarithmic cooling: T_k = T_0 / ln(k + 2)
    /// Theoretically optimal but very slow
    Logarithmic,

    /// Adaptive cooling based on acceptance ratio
    /// Increases temp if acceptance too low, decreases if too high
    Adaptive {
        target_acceptance: f64,
        adjustment_factor: f64,
    },
}

impl CoolingSchedule {
    /// Create a geometric cooling schedule with the given alpha
    pub fn geometric(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "Alpha must be in (0, 1)");
        CoolingSchedule::Geometric { alpha }
    }

    /// Create a linear cooling schedule given initial temp and total steps
    pub fn linear(initial_temp: f64, n_steps: u32) -> Self {
        let beta = initial_temp / n_steps as f64;
        CoolingSchedule::Linear { beta }
    }

    /// Create a logarithmic cooling schedule
    pub fn logarithmic() -> Self {
        CoolingSchedule::Logarithmic
    }

    /// Create an adaptive cooling schedule
    pub fn adaptive(target_acceptance: f64, adjustment_factor: f64) -> Self {
        CoolingSchedule::Adaptive {
            target_acceptance,
            adjustment_factor,
        }
    }

    /// Compute next temperature given current state
    pub fn next_temperature(
        &self,
        current_temp: f64,
        initial_temp: f64,
        step: u32,
        acceptance_ratio: f64,
    ) -> f64 {
        match self {
            CoolingSchedule::Geometric { alpha } => current_temp * alpha,
            CoolingSchedule::Linear { beta } => (current_temp - beta).max(0.001),
            CoolingSchedule::Logarithmic => initial_temp / ((step + 2) as f64).ln(),
            CoolingSchedule::Adaptive {
                target_acceptance,
                adjustment_factor,
            } => {
                if acceptance_ratio < *target_acceptance {
                    // Too few acceptances, increase temperature
                    current_temp * (1.0 + adjustment_factor)
                } else {
                    // Enough acceptances, decrease temperature
                    current_temp * (1.0 - adjustment_factor)
                }
            }
        }
    }
}

/// Extended output for true SA placer with additional statistics
#[derive(Clone)]
pub struct SAPlacerOutput<'a> {
    pub initial_solution: PlacementSolution<'a>,
    pub final_solution: PlacementSolution<'a>,
    pub best_solution: PlacementSolution<'a>,
    pub x_steps: Vec<u32>,
    pub y_cost: Vec<f32>,
    pub y_temperature: Vec<f64>,
    pub y_acceptance_ratio: Vec<f64>,
    pub total_accepted: u32,
    pub total_rejected: u32,
    pub uphill_accepted: u32,
    pub renderer: Option<Renderer>,
}

/// Estimate initial temperature using random walk sampling
///
/// Samples random moves and computes a temperature such that
/// the acceptance probability for an average uphill move equals
/// the target acceptance ratio (typically 0.8 for initial temp).
pub fn estimate_initial_temperature(
    solution: &PlacementSolution,
    target_acceptance: f64,
    sample_size: usize,
) -> f64 {
    let mut rng = rand::thread_rng();
    let actions = [PlacementAction::Move, PlacementAction::Swap];

    let mut positive_deltas = Vec::with_capacity(sample_size);

    for _ in 0..sample_size {
        let mut candidate = solution.clone();
        let action = actions.choose(&mut rng).unwrap();
        candidate.action(*action);

        let delta = candidate.cost_bb() - solution.cost_bb();
        if delta > 0.0 {
            positive_deltas.push(delta as f64);
        }
    }

    if positive_deltas.is_empty() {
        return 1000.0; // Default if no uphill moves found
    }

    let avg_delta: f64 = positive_deltas.iter().sum::<f64>() / positive_deltas.len() as f64;

    // Solve: exp(-avg_delta / T) = target_acceptance
    // T = -avg_delta / ln(target_acceptance)
    -avg_delta / target_acceptance.ln()
}

/// True Simulated Annealing placer with Metropolis-Hastings acceptance criterion
///
/// Unlike `fast_sa_placer`, this implementation:
/// - Accepts worse solutions with probability exp(-delta/T)
/// - Implements proper temperature cooling schedules
/// - Can escape local optima through probabilistic uphill moves
///
/// # Arguments
/// * `initial_solution` - Starting placement
/// * `n_steps` - Number of SA iterations
/// * `initial_temp` - Starting temperature (use `estimate_initial_temperature` if unsure)
/// * `cooling_schedule` - Temperature reduction strategy
/// * `verbose` - Print progress every 100 steps
/// * `render` - Generate SVG frames for animation
///
/// # Returns
/// Extended output with temperature and acceptance statistics
pub fn true_sa_placer<'a>(
    initial_solution: PlacementSolution<'a>,
    n_steps: u32,
    initial_temp: f64,
    cooling_schedule: CoolingSchedule,
    verbose: bool,
    render: bool,
) -> SAPlacerOutput<'a> {
    let mut renderer = Renderer::new();
    let mut rng = rand::thread_rng();

    let mut current_solution = initial_solution.clone();
    let mut best_solution = initial_solution.clone();
    let mut best_cost = current_solution.cost_bb();

    let mut temperature = initial_temp;

    let actions = [PlacementAction::Move, PlacementAction::Swap];

    // Statistics tracking
    let mut x_steps = Vec::with_capacity(n_steps as usize);
    let mut y_cost = Vec::with_capacity(n_steps as usize);
    let mut y_temperature = Vec::with_capacity(n_steps as usize);
    let mut y_acceptance_ratio = Vec::with_capacity(n_steps as usize);

    let mut total_accepted: u32 = 0;
    let mut total_rejected: u32 = 0;
    let mut uphill_accepted: u32 = 0;

    // Window for acceptance ratio calculation
    let window_size = 100;
    let mut recent_accepted = 0u32;
    let mut recent_total = 0u32;

    for i in 0..n_steps {
        let current_cost = current_solution.cost_bb();

        // Record statistics
        x_steps.push(i);
        y_cost.push(current_cost);
        y_temperature.push(temperature);

        let acceptance_ratio = if recent_total > 0 {
            recent_accepted as f64 / recent_total as f64
        } else {
            1.0
        };
        y_acceptance_ratio.push(acceptance_ratio);

        if render {
            renderer.add_frame(current_solution.render_svg());
        }

        // Generate single neighbour (classical SA uses one neighbour per step)
        let mut candidate = current_solution.clone();
        let action = actions.choose(&mut rng).unwrap();
        candidate.action(*action);

        let candidate_cost = candidate.cost_bb();
        let delta = candidate_cost - current_cost;

        // Metropolis-Hastings acceptance criterion
        let accept = if delta < 0.0 {
            // Always accept improvements
            true
        } else if temperature > 0.0 {
            // Accept worse solutions with probability exp(-delta/T)
            let probability = (-(delta as f64) / temperature).exp();
            rng.gen::<f64>() < probability
        } else {
            false
        };

        // Update acceptance window
        recent_total += 1;
        if recent_total > window_size {
            recent_total = window_size;
            recent_accepted = (recent_accepted as f64 * 0.99) as u32;
        }

        if accept {
            current_solution = candidate;
            total_accepted += 1;
            recent_accepted += 1;

            if delta > 0.0 {
                uphill_accepted += 1;
            }

            // Track best solution found
            if candidate_cost < best_cost {
                best_solution = current_solution.clone();
                best_cost = candidate_cost;
            }
        } else {
            total_rejected += 1;
        }

        // Update temperature according to cooling schedule
        temperature = cooling_schedule.next_temperature(
            temperature,
            initial_temp,
            i,
            acceptance_ratio,
        );

        if verbose && i % 100 == 0 {
            println!(
                "Step {}: cost={:.1}, temp={:.2}, acceptance={:.2}%, uphill={}",
                i,
                current_cost,
                temperature,
                acceptance_ratio * 100.0,
                uphill_accepted
            );
        }
    }

    if render {
        renderer.add_frame(current_solution.render_svg());
    }

    SAPlacerOutput {
        initial_solution: initial_solution.clone(),
        final_solution: current_solution,
        best_solution,
        x_steps,
        y_cost,
        y_temperature,
        y_acceptance_ratio,
        total_accepted,
        total_rejected,
        uphill_accepted,
        renderer: if render { Some(renderer) } else { None },
    }
}

/// Hybrid placer: Greedy descent followed by SA refinement
///
/// This combines the speed of greedy descent with SA's ability
/// to escape local optima.
pub fn hybrid_placer<'a>(
    initial_solution: PlacementSolution<'a>,
    greedy_steps: u32,
    sa_steps: u32,
    initial_temp: f64,
    cooling_schedule: CoolingSchedule,
    verbose: bool,
) -> SAPlacerOutput<'a> {
    // Phase 1: Fast greedy descent
    if verbose {
        println!("Phase 1: Greedy descent ({} steps)", greedy_steps);
    }
    let greedy_output = fast_sa_placer(initial_solution.clone(), greedy_steps, 3, false, false);

    // Phase 2: SA refinement
    if verbose {
        println!("Phase 2: SA refinement ({} steps)", sa_steps);
    }
    true_sa_placer(
        greedy_output.final_solution,
        sa_steps,
        initial_temp,
        cooling_schedule,
        verbose,
        false,
    )
}
