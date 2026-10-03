# Validation recorded October 1, 2026

`cargo check --offline`, `cargo test --offline --test terrain`, and `cargo clippy --offline --all-targets -- -D warnings` passed. The GPU smoke test completed successfully, and its landscape and before/after excavation captures were visually inspected.

One run of `cargo run --offline --example benchmark`, seed 1337, optimized development profile (project optimization level 1, dependency level 3):

| Measurement | Result |
| --- | ---: |
| Heightfield generation, 90,000 droplets + 30 thermal passes | 0.393 s |
| Initial cached samples + all meshes, one CPU thread | 0.547 s |
| Allocated initial chunks, including conservative surface bounds | 624 |
| Initial triangles | 726,850 |
| Cached f32 samples, including halos | 41.8 MiB |
| Field update for a 7 m excavation | 0.341 ms |
| Chunks affected by that excavation | 8 |
| Combined CPU meshing for those chunks | 2.136 ms |

These are single-run CPU observations from this session, not benchmark medians, GPU timings, release-profile results or comparisons against other algorithms. The sample-memory figure excludes meshes, renderer allocations and temporary copy-on-write snapshots. The viewer reports the last accepted chunk's CPU meshing time, not end-to-end brush latency; scheduling, frame polling and GPU uploads add further delay.

The geometry test verifies deterministic generation, finite values, transported sediment, map sediment conservation, ray hits/misses, invalid brush input, 3D cavity/fill behaviour, monotonic revisions, immutable snapshots, exact shared halo values, and a watertight eight-chunk sphere after welding matching vertices. A second check parses the terrain shader. The native renderer smoke test resolves Bevy shader imports and applies an excavation to the actual scene, then checks that the cavity's center is empty.

The sphere seam check does not prove that Surface Nets produces manifold surfaces for every sampled topology. No multi-hardware FPS, kilometre-scale streaming, persistent save/load, walking collision, or comparison with SVO/NAADF/dual contouring was tested.

Captures from the smoke test are written to `captures/landscape.png`, `captures/before-dig.png`, and `captures/after-dig.png`.
