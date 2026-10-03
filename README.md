# TERRA

A runnable Bevy 0.19.1 landscape with hydraulic erosion, smooth voxel surfaces, and live excavation. No downloaded art assets are required.

```powershell
cargo run --release
# Reproducible alternative landscape:
cargo run --release -- --seed 42
```

The first build compiles Bevy. Terrain generation runs in the background; nearby chunks appear first. This machine also has the dependencies cached, so `cargo run --offline` works with the optimized development profile.

| Control | Action |
| --- | --- |
| Click | Capture mouse; the initial click does not dig |
| Mouse | Look |
| WASD / arrow keys | Fly |
| Q / E | Descend / rise |
| Shift | Fly faster |
| Hold left mouse | Dig |
| Hold right mouse | Add material |
| Mouse wheel / brackets | Brush radius, 1.5–12 m |
| L | Toggle cave headlamp |
| Esc | Release mouse; focus loss also releases it |
| F12 | Save `captures/terrain.png` |

The explorer is a free-flight camera. Digging edits a 3D scalar field, so cavities, tunnels and overhangs are supported; it does not just lower a heightmap. Excavated walls expose soil and layered rock. The landscape is 384 × 384 m, sampled at 1 m, with an editable vertical range from −48 to 144 m. Sub-voxel features are limited by sampling resolution.

The surface uses procedural world-space materials, slope and depth masks, erosion-flow wetness, surface-gradient bump mapping, atmosphere lighting, cascaded shadows, ambient occlusion and temporal antialiasing. Terrain colors and geological layers are artistic approximations, not a geological survey.

## Validation

```powershell
cargo test --offline
cargo run --offline --example benchmark
cargo run --offline -- --smoke-test
```

The integration check covers repeatable generation, sediment conservation, field sampling, digging/filling, immutable worker snapshots, shared chunk halos and a watertight chunked sphere. The smoke test exercises the GPU material, captures a landscape view and a before/after excavation, then exits. The benchmark measures generation, initial field caching and meshing, and a 7 m brush edit; it is not an FPS benchmark. Run the release version for production timing.

[Recorded validation and benchmark results](docs/VALIDATION.md).

## Implementation

- [Research and method comparison](docs/RESEARCH.md): primary papers through October 1, 2026, with separate recommendations for surface generation, storage, meshing and rendering.
- `src/terrain.rs`: erosion, sparse cached density chunks, trilinear queries, brush operations and Surface Nets extraction.
- `src/main.rs`: viewer, local editing, bounded asynchronous meshing and rendering.
- `assets/shaders/terrain.wgsl`: continuous 3D terrain material and bump detail.

Each chunk owns 24³ cells with overlapping boundary samples. An edit updates every affected halo copy, increments revisions, and queues local remeshing. Workers use immutable snapshots; outdated results are rejected. There are at most four jobs and three accepted mesh uploads per frame. The HUD reports cached sample memory and the last accepted chunk's CPU meshing time. Scheduling, polling and GPU uploads add to end-to-end edit latency.

This is a finite terrain prototype. It does not yet include distance LOD, streaming, disk persistence, player physics, full path-traced lighting, live geological erosion after excavation, or photogrammetry materials. The density is a signed implicit field rather than an exact Euclidean SDF; the picker therefore uses conservative small fixed steps and bracket refinement instead of unsafe sphere-tracing distances. Very thin grazing intersections may be missed. Surface Nets does not guarantee manifold topology for every ambiguous sampled field.
