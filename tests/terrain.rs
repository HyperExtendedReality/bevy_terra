use bevy::prelude::*;
use bevy_terra::terrain::*;
use fast_surface_nets::ndshape::ConstShape;
use noise::Perlin;
use std::collections::HashMap;

fn flat() -> Landscape {
    Landscape {
        heights: vec![0.3; SIDE * SIDE],
        flow: vec![0.0; SIDE * SIDE],
        noise: Perlin::new(1337),
    }
}

#[test]
fn terrain_generation_excavation_and_chunk_seams() {
    let settings = Settings {
        droplets: 500,
        thermal_steps: 3,
        ..default()
    };
    let a = Landscape::generate(settings);
    let b = Landscape::generate(settings);
    assert_eq!(a.heights, b.heights, "seed must reproduce the landscape");
    assert!(a.heights.iter().all(|h| h.is_finite()));
    assert!(a.flow.iter().any(|f| *f > 0.0));
    let raw = Landscape::generate(Settings {
        droplets: 0,
        thermal_steps: 0,
        ..settings
    });
    let mass_before: f64 = raw.heights.iter().map(|h| *h as f64).sum();
    let mass_after: f64 = a.heights.iter().map(|h| *h as f64).sum();
    assert!(
        (mass_before - mass_after).abs() < 0.5,
        "erosion/deposition must conserve map sediment"
    );

    let mut terrain = Terrain::new(flat());
    let hit = terrain
        .raycast(Vec3::new(0.0, 12.0, 0.0), -Vec3::Y, 30.0)
        .expect("ground hit");
    assert!(hit.y.abs() < 1.0);
    assert!(
        terrain
            .raycast(Vec3::new(0.0, 12.0, 0.0), Vec3::Y, 10.0)
            .is_none()
    );
    assert!(terrain.raycast(Vec3::ZERO, Vec3::ZERO, 10.0).is_none());
    assert!(terrain.brush(Vec3::NAN, 3.0, false).is_empty());
    assert!(terrain.brush(Vec3::ZERO, f32::INFINITY, false).is_empty());

    // Face, edge and corner overlap of signed-coordinate chunks.
    let center = Vec3::new(0.0, -3.0, 0.0);
    let dirty = terrain.brush(center, 5.0, false);
    assert!(dirty.len() >= 4);
    assert!(
        terrain.sample(center) > 4.0,
        "a cavity must be genuinely empty in 3D"
    );
    assert!(
        terrain.sample(center + Vec3::Y * (-8.0)) < 0.0,
        "ground beyond brush must remain"
    );
    let snapshot = terrain.chunks[&IVec3::new(0, -1, 0)].samples.clone();
    let revision = terrain.chunks[&IVec3::new(0, -1, 0)].revision;
    terrain.brush(center, 5.0, true);
    assert!(
        terrain.sample(center) < 0.0,
        "filling must restore solid density"
    );
    assert!(terrain.chunks[&IVec3::new(0, -1, 0)].revision > revision);
    assert_ne!(
        *snapshot,
        *terrain.chunks[&IVec3::new(0, -1, 0)].samples,
        "worker snapshot must be immutable"
    );
    terrain.brush(center, 5.0, false);
    for (&coord, chunk) in &terrain.chunks {
        for i in 0..Shape::SIZE {
            let local = IVec3::from_array(Shape::delinearize(i).map(|v| v as i32));
            let global = coord * CELLS - IVec3::ONE + local;
            assert_eq!(
                chunk.samples[i as usize],
                terrain.lattice(global),
                "halo mismatch at {global}"
            );
        }
    }

    // Weld the eight chunks of a sphere. Every undirected edge must have two incident triangles.
    let mut edges = HashMap::new();
    for z in -1..=0 {
        for y in -1..=0 {
            for x in -1..=0 {
                let coord = IVec3::new(x, y, z);
                let origin = chunk_origin(coord);
                let samples: Vec<f32> = (0..Shape::SIZE)
                    .map(|i| {
                        let p = origin + Vec3::from_array(Shape::delinearize(i).map(|v| v as f32));
                        p.length() - 5.3
                    })
                    .collect();
                let mesh = polygonize(coord, &samples, &flat());
                assert!(
                    mesh.normals
                        .iter()
                        .all(|n| Vec3::from_array(*n).is_finite())
                );
                for face in mesh.indices.as_chunks::<3>().0 {
                    for pair in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                        let key = |index: u32| {
                            mesh.positions[index as usize].map(|f| (f * 10000.0).round() as i32)
                        };
                        let (a, b) = (key(pair.0), key(pair.1));
                        *edges
                            .entry(if a < b { (a, b) } else { (b, a) })
                            .or_insert(0) += 1;
                    }
                }
            }
        }
    }
    assert!(!edges.is_empty());
    assert!(
        edges.values().all(|n| *n == 2),
        "chunked sphere must be watertight without duplicate faces"
    );
}

#[test]
fn terrain_shader_syntax() {
    // Bevy resolves imports and definitions during the GPU smoke test. Parse the local shader too.
    let source = include_str!("../assets/shaders/terrain.wgsl");
    let body = source.split("@group").nth(1).expect("material binding");
    let body = format!("@group{}", body).replace("#{MATERIAL_BIND_GROUP}", "2");
    let interfaces = "struct VertexOutput { world_position: vec4<f32>, world_normal: vec3<f32>, uv: vec2<f32> }\nstruct FragmentOutput { @location(0) color: vec4<f32> }\nstruct Material { base_color: vec4<f32>, perceptual_roughness: f32 }\nstruct Pbr { material: Material, N: vec3<f32> }\nfn pbr_input_from_standard_material(v: VertexOutput, f: bool) -> Pbr { return Pbr(Material(vec4(1.0),1.0),v.world_normal); }\nfn apply_pbr_lighting(p: Pbr) -> vec4<f32> { return p.material.base_color; }\nfn main_pass_post_lighting_processing(p: Pbr,c: vec4<f32>) -> vec4<f32> { return c; }\n";
    let result = naga::front::wgsl::parse_str(&format!("{interfaces}{body}"));
    if let Err(error) = result {
        panic!("{}", error.emit_to_string(&format!("{interfaces}{body}")));
    }
}
