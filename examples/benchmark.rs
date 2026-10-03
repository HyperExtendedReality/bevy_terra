use bevy_terra::terrain::*;
use fast_surface_nets::ndshape::ConstShape;
use std::time::Instant;

fn main() {
    let start = Instant::now();
    let mut terrain = Terrain::new(Landscape::generate(Settings::default()));
    println!("Generation: {:.3}s", start.elapsed().as_secs_f64());
    let coords = terrain.initial_coords();
    let start = Instant::now();
    let mut triangles = 0;
    for coord in &coords {
        terrain.ensure_chunk(*coord);
        let mesh = polygonize(*coord, &terrain.chunks[coord].samples, &terrain.landscape);
        triangles += mesh.indices.len() / 3;
    }
    println!(
        "Initial cache + meshes (one thread): {:.3}s; {} chunks, {} triangles, {:.1} MiB samples",
        start.elapsed().as_secs_f64(),
        coords.len(),
        triangles,
        terrain.chunks.len() as f64 * Shape::USIZE as f64 * 4.0 / 1048576.0
    );
    let center = bevy::prelude::Vec3::new(0.0, terrain.landscape.height(0.0, 0.0) - 2.0, 0.0);
    let start = Instant::now();
    let dirty = terrain.brush(center, 7.0, false);
    let brush_time = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    for coord in &dirty {
        std::hint::black_box(polygonize(
            *coord,
            &terrain.chunks[coord].samples,
            &terrain.landscape,
        ));
    }
    println!(
        "7m excavation: {:.3}ms field update; {} chunks; {:.3}ms total CPU meshing",
        brush_time,
        dirty.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );
}
