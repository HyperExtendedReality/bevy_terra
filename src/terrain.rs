use bevy::prelude::*;
use fast_surface_nets::{
    SurfaceNetsBuffer,
    ndshape::{ConstShape, ConstShape3u32},
    surface_nets,
};
use noise::{NoiseFn, Perlin};
use std::{collections::HashMap, sync::Arc};

pub const CELLS: i32 = 24;
pub const VOXEL: f32 = 1.0;
pub const HALF_WORLD: i32 = 192;
pub const MIN_Y: i32 = -48;
pub const MAX_Y: i32 = 144;
pub const SIDE: usize = (HALF_WORLD * 2 + 1) as usize;
pub type Shape = ConstShape3u32<26, 26, 26>;

#[derive(Clone, Copy)]
pub struct Settings {
    pub seed: u32,
    pub droplets: usize,
    pub thermal_steps: usize,
    pub erosion_rate: f32,
    pub deposition_rate: f32,
    pub evaporation: f32,
    pub talus: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            seed: 1337,
            droplets: 90_000,
            thermal_steps: 30,
            erosion_rate: 0.24,
            deposition_rate: 0.18,
            evaporation: 0.025,
            talus: 1.1,
        }
    }
}

#[derive(Clone)]
pub struct Landscape {
    pub heights: Vec<f32>,
    pub flow: Vec<f32>,
    pub noise: Perlin,
}

// Deterministic sampling without an additional random-number dependency.
struct Random(u64);
impl Random {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

impl Landscape {
    pub fn generate(settings: Settings) -> Self {
        let noise = Perlin::new(settings.seed);
        let mut heights = vec![0.0; SIDE * SIDE];
        for z in 0..SIDE {
            for x in 0..SIDE {
                let wx = x as f64 - HALF_WORLD as f64;
                let wz = z as f64 - HALF_WORLD as f64;
                let warp = noise.get([wx * 0.007, wz * 0.007]) * 24.0;
                let mut ridges = 0.0;
                let mut amplitude = 1.0;
                let mut frequency = 0.008;
                for octave in 0..5 {
                    let n =
                        noise.get([(wx + warp) * frequency, wz * frequency, octave as f64 * 7.0]);
                    ridges += (1.0 - n.abs()).powi(3) * amplitude;
                    frequency *= 2.1;
                    amplitude *= 0.48;
                }
                let river_x = 28.0 * (wz * 0.016).sin() - 12.0;
                let valley = (-((wx - river_x) / 39.0).powi(2)).exp();
                let mountain =
                    (ridges * 44.0 + noise.get([wx * 0.004, wz * 0.004]) * 22.0 + 20.0).max(10.0);
                heights[z * SIDE + x] = (mountain * (1.0 - valley * 0.83) + 5.0) as f32;
            }
        }
        let mut terrain = Self {
            heights,
            flow: vec![0.0; SIDE * SIDE],
            noise,
        };
        terrain.erode(settings);
        terrain
    }

    fn map_sample(map: &[f32], p: Vec2) -> (f32, Vec2) {
        let p = p.clamp(Vec2::ZERO, Vec2::splat((SIDE - 2) as f32));
        let [x, z] = p.floor().as_uvec2().to_array();
        let f = p.fract();
        let i = z as usize * SIDE + x as usize;
        let [a, b, c, d] = [map[i], map[i + 1], map[i + SIDE], map[i + SIDE + 1]];
        let height = a.lerp(b, f.x).lerp(c.lerp(d, f.x), f.y);
        let gradient = Vec2::new((b - a).lerp(d - c, f.y), (c - a).lerp(d - b, f.x));
        (height, gradient)
    }

    fn deposit(&mut self, p: Vec2, sediment: f32) {
        let [x, z] = p.floor().as_uvec2().to_array();
        let f = p.fract();
        let i = z as usize * SIDE + x as usize;
        for (offset, weight) in [
            (0, (1.0 - f.x) * (1.0 - f.y)),
            (1, f.x * (1.0 - f.y)),
            (SIDE, (1.0 - f.x) * f.y),
            (SIDE + 1, f.x * f.y),
        ] {
            self.heights[i + offset] += sediment * weight;
        }
    }

    fn erode(&mut self, s: Settings) {
        let mut rng = Random(s.seed as u64 + 1);
        // ponytail: sequential droplets avoid write races; move transport to GPU for larger maps.
        for _ in 0..s.droplets {
            let mut p = Vec2::new(rng.unit(), rng.unit()) * (SIDE - 5) as f32 + Vec2::splat(2.0);
            let mut direction = Vec2::ZERO;
            let (mut speed, mut water, mut sediment) = (1.0f32, 1.0f32, 0.0f32);
            for _ in 0..70 {
                let (height, gradient) = Self::map_sample(&self.heights, p);
                direction = (direction * 0.35 - gradient * 0.65).normalize_or_zero();
                if direction == Vec2::ZERO {
                    break;
                }
                let next = p + direction;
                if next.min_element() < 2.0 || next.max_element() >= (SIDE - 3) as f32 {
                    break;
                }
                let dh = Self::map_sample(&self.heights, next).0 - height;
                let capacity = (-dh).max(0.02) * speed * water * 3.2;
                if dh > 0.0 || sediment > capacity {
                    let amount = if dh > 0.0 {
                        sediment.min(dh)
                    } else {
                        (sediment - capacity) * s.deposition_rate
                    };
                    sediment -= amount;
                    self.deposit(p, amount);
                } else {
                    let amount = ((capacity - sediment) * s.erosion_rate).min(-dh).min(0.65);
                    let center = p.floor().as_ivec2();
                    let mut total = 0.0;
                    for dz in -2i32..=2 {
                        for dx in -2i32..=2 {
                            total += (2.5 - Vec2::new(dx as f32, dz as f32).length()).max(0.0);
                        }
                    }
                    for dz in -2i32..=2 {
                        for dx in -2i32..=2 {
                            let weight =
                                (2.5 - Vec2::new(dx as f32, dz as f32).length()).max(0.0) / total;
                            let i = (center.y + dz) as usize * SIDE + (center.x + dx) as usize;
                            self.heights[i] -= amount * weight;
                        }
                    }
                    sediment += amount;
                }
                self.flow[p.y as usize * SIDE + p.x as usize] += water;
                speed = (speed * speed - dh * 4.0).max(0.01).sqrt();
                water *= 1.0 - s.evaporation;
                p = next;
                if water < 0.05 {
                    break;
                }
            }
            // Deposit remaining load at the terminal point to conserve sediment on the map.
            self.deposit(p, sediment);
        }
        let mut delta = vec![0.0; self.heights.len()];
        for _ in 0..s.thermal_steps {
            delta.fill(0.0);
            for z in 1..SIDE - 1 {
                for x in 1..SIDE - 1 {
                    let i = z * SIDE + x;
                    let lowest = [i - 1, i + 1, i - SIDE, i + SIDE]
                        .into_iter()
                        .min_by(|a, b| self.heights[*a].total_cmp(&self.heights[*b]))
                        .expect("four neighbours");
                    let amount =
                        ((self.heights[i] - self.heights[lowest] - s.talus) * 0.2).max(0.0);
                    delta[i] -= amount;
                    delta[lowest] += amount;
                }
            }
            for (h, d) in self.heights.iter_mut().zip(&delta) {
                *h += d;
            }
        }
        for value in &mut self.flow {
            *value = (1.0 + *value).ln() / 7.0;
        }
    }

    pub fn height(&self, x: f32, z: f32) -> f32 {
        Self::map_sample(
            &self.heights,
            Vec2::new(x, z) + Vec2::splat(HALF_WORLD as f32),
        )
        .0
    }

    pub fn wetness(&self, x: f32, z: f32) -> f32 {
        Self::map_sample(&self.flow, Vec2::new(x, z) + Vec2::splat(HALF_WORLD as f32))
            .0
            .clamp(0.0, 1.0)
    }

    pub fn density(&self, p: Vec3) -> f32 {
        let base = p.y - self.height(p.x, p.z);
        let detail =
            self.noise
                .get([p.x as f64 * 0.14, p.y as f64 * 0.14, p.z as f64 * 0.14]) as f32;
        // A signed scalar field, not an exact Euclidean SDF: never sphere-trace it unchecked.
        base + detail * 0.55 * (1.0 - base.abs() / 3.0).max(0.0)
    }
}

pub struct Chunk {
    pub samples: Arc<Vec<f32>>,
    pub revision: u64,
}

#[derive(Resource)]
pub struct Terrain {
    pub landscape: Arc<Landscape>,
    pub chunks: HashMap<IVec3, Chunk>,
}

pub fn chunk_origin(coord: IVec3) -> Vec3 {
    (coord * CELLS - IVec3::ONE).as_vec3() * VOXEL
}

pub fn valid_chunk(c: IVec3) -> bool {
    (-HALF_WORLD / CELLS..HALF_WORLD / CELLS).contains(&c.x)
        && (-HALF_WORLD / CELLS..HALF_WORLD / CELLS).contains(&c.z)
        && (MIN_Y / CELLS..MAX_Y / CELLS).contains(&c.y)
}

impl Terrain {
    pub fn new(landscape: Landscape) -> Self {
        Self {
            landscape: Arc::new(landscape),
            chunks: HashMap::new(),
        }
    }

    pub fn initial_coords(&self) -> Vec<IVec3> {
        let mut coords = Vec::new();
        for z in -HALF_WORLD / CELLS..HALF_WORLD / CELLS {
            for x in -HALF_WORLD / CELLS..HALF_WORLD / CELLS {
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for dz in -1..=CELLS + 1 {
                    for dx in -1..=CELLS + 1 {
                        let h = self
                            .landscape
                            .height((x * CELLS + dx) as f32, (z * CELLS + dz) as f32);
                        lo = lo.min(h);
                        hi = hi.max(h);
                    }
                }
                for y in MIN_Y / CELLS..MAX_Y / CELLS {
                    if ((y * CELLS - 1) as f32) < hi + 2.0
                        && (((y + 1) * CELLS + 1) as f32) > lo - 2.0
                    {
                        coords.push(IVec3::new(x, y, z));
                    }
                }
            }
        }
        coords
    }

    pub fn ensure_chunk(&mut self, coord: IVec3) {
        self.chunks.entry(coord).or_insert_with(|| {
            let origin = chunk_origin(coord);
            let mut samples = Vec::with_capacity(Shape::USIZE);
            for i in 0..Shape::SIZE {
                let p = Vec3::from_array(Shape::delinearize(i).map(|v| v as f32));
                samples.push(self.landscape.density(origin + p * VOXEL));
            }
            Chunk {
                samples: Arc::new(samples),
                revision: 0,
            }
        });
    }

    pub fn lattice(&self, p: IVec3) -> f32 {
        let coord = IVec3::new(
            p.x.div_euclid(CELLS),
            p.y.div_euclid(CELLS),
            p.z.div_euclid(CELLS),
        );
        if let Some(chunk) = self.chunks.get(&coord) {
            let q = (p - coord * CELLS + IVec3::ONE).as_uvec3().to_array();
            chunk.samples[Shape::linearize(q) as usize]
        } else {
            self.landscape.density(p.as_vec3() * VOXEL)
        }
    }

    pub fn sample(&self, world: Vec3) -> f32 {
        let p = world / VOXEL;
        let lo = p.floor().as_ivec3();
        let f = p - lo.as_vec3();
        let mut yz = [0.0; 4];
        for z in 0..2 {
            for y in 0..2 {
                yz[z * 2 + y] = self
                    .lattice(lo + IVec3::new(0, y as i32, z as i32))
                    .lerp(self.lattice(lo + IVec3::new(1, y as i32, z as i32)), f.x);
            }
        }
        yz[0].lerp(yz[1], f.y).lerp(yz[2].lerp(yz[3], f.y), f.z)
    }

    pub fn normal(&self, p: Vec3) -> Vec3 {
        let h = VOXEL * 0.5;
        Vec3::new(
            self.sample(p + Vec3::X * h) - self.sample(p - Vec3::X * h),
            self.sample(p + Vec3::Y * h) - self.sample(p - Vec3::Y * h),
            self.sample(p + Vec3::Z * h) - self.sample(p - Vec3::Z * h),
        )
        .normalize_or(Vec3::Y)
    }

    pub fn raycast(&self, origin: Vec3, direction: Vec3, reach: f32) -> Option<Vec3> {
        let direction = direction.try_normalize()?;
        let mut previous = self.sample(origin);
        // ponytail: quarter-voxel picking can miss sub-voxel grazing hits; use cell cubic roots if needed.
        let step = VOXEL * 0.25;
        let mut distance = step;
        while distance <= reach {
            let point = origin + direction * distance;
            let value = self.sample(point);
            if (value < 0.0) != (previous < 0.0) {
                let (mut a, mut b) = (distance - step, distance);
                for _ in 0..12 {
                    let mid = (a + b) * 0.5;
                    if (self.sample(origin + direction * mid) < 0.0) == (previous < 0.0) {
                        a = mid;
                    } else {
                        b = mid;
                    }
                }
                let hit = origin + direction * ((a + b) * 0.5);
                return (hit.x.abs() < HALF_WORLD as f32 - 2.0
                    && hit.z.abs() < HALF_WORLD as f32 - 2.0
                    && hit.y > MIN_Y as f32 + 2.0
                    && hit.y < MAX_Y as f32 - 2.0)
                    .then_some(hit);
            }
            previous = value;
            distance += step;
        }
        None
    }

    /// Update every duplicated halo sample, including corner neighbours, from the same brush.
    pub fn brush(&mut self, center: Vec3, radius: f32, add: bool) -> Vec<IVec3> {
        if !center.is_finite() || !radius.is_finite() || !(1.0..=12.0).contains(&radius) {
            return Vec::new();
        }
        let extent = Vec3::splat(radius + 2.0 * VOXEL);
        let lo = ((center - extent) / (CELLS as f32 * VOXEL))
            .floor()
            .as_ivec3();
        let hi = ((center + extent) / (CELLS as f32 * VOXEL))
            .floor()
            .as_ivec3();
        let mut dirty = Vec::new();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let coord = IVec3::new(x, y, z);
                    if !valid_chunk(coord) {
                        continue;
                    }
                    self.ensure_chunk(coord);
                    let origin = chunk_origin(coord);
                    let chunk = self.chunks.get_mut(&coord).expect("just inserted");
                    let samples = Arc::make_mut(&mut chunk.samples);
                    let mut changed = false;
                    for (i, d) in samples.iter_mut().enumerate() {
                        let p = origin
                            + Vec3::from_array(Shape::delinearize(i as u32).map(|v| v as f32))
                                * VOXEL;
                        let sphere = p.distance(center) - radius;
                        // Compact support keeps edits local and all halo duplicates bit-identical.
                        if sphere <= VOXEL {
                            let new = if add { d.min(sphere) } else { d.max(-sphere) };
                            if new != *d {
                                *d = new;
                                changed = true;
                            }
                        }
                    }
                    if changed {
                        chunk.revision += 1;
                        dirty.push(coord);
                    }
                }
            }
        }
        dirty
    }
}

pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub terrain_uv: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

pub fn polygonize(coord: IVec3, samples: &[f32], landscape: &Landscape) -> MeshData {
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(samples, &Shape {}, [0; 3], [25; 3], &mut buffer);
    let origin = chunk_origin(coord);
    let mut terrain_uv = Vec::with_capacity(buffer.positions.len());
    for p in &mut buffer.positions {
        let world = origin + Vec3::from_array(*p) * VOXEL;
        *p = world.to_array();
        terrain_uv.push([
            landscape.height(world.x, world.z),
            landscape.wetness(world.x, world.z),
        ]);
    }
    MeshData {
        positions: buffer.positions,
        normals: buffer.normals,
        terrain_uv,
        indices: buffer.indices,
    }
}
