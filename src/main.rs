#![recursion_limit = "256"]

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    app::AppExit,
    asset::RenderAssetUsages,
    camera::{Exposure, Hdr},
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder,
        atmosphere::ScatteringMedium,
    },
    mesh::{Indices, PrimitiveTopology},
    pbr::{AtmosphereSettings, ExtendedMaterial, MaterialExtension, ScreenSpaceAmbientOcclusion},
    prelude::*,
    render::{
        render_resource::AsBindGroup,
        view::screenshot::{Screenshot, save_to_disk},
    },
    shader::ShaderRef,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use bevy_terra::terrain::*;
use fast_surface_nets::ndshape::ConstShape;
use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

type TerrainMaterial = ExtendedMaterial<StandardMaterial, Ground>;

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct Ground {
    #[uniform(100)]
    detail_strength: f32,
}
impl MaterialExtension for Ground {
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain.wgsl".into()
    }
}

#[derive(Resource)]
struct Generation {
    task: Option<Task<Landscape>>,
    started: Instant,
}

struct Job {
    revision: u64,
    task: Task<(MeshData, f32)>,
}
struct DisplayChunk {
    entity: Entity,
    mesh: Handle<Mesh>,
    triangles: usize,
}

#[derive(Resource, Default)]
struct Meshing {
    pending: HashSet<IVec3>,
    running: HashMap<IVec3, Job>,
    displayed: HashMap<IVec3, DisplayChunk>,
    material: Handle<TerrainMaterial>,
    last_ms: f32,
    rejected: u64,
    generation_secs: f32,
}

#[derive(Component)]
struct Explorer {
    yaw: f32,
    pitch: f32,
}
#[derive(Component)]
struct Headlamp;
#[derive(Component)]
struct StatusText;

#[derive(Resource)]
struct Tool {
    radius: f32,
    cooldown: f32,
    last_center: Option<Vec3>,
    hit: Option<Vec3>,
    strokes: u64,
    notice: &'static str,
    smoke: bool,
    smoke_stage: u8,
    settle: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut settings = Settings::default();
    if let Some(i) = args.iter().position(|a| a == "--seed") {
        let Some(seed) = args.get(i + 1).and_then(|a| a.parse::<u32>().ok()) else {
            eprintln!("Usage: cargo run --release -- [--seed NUMBER] [--smoke-test]");
            std::process::exit(2);
        };
        settings.seed = seed;
    }
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(WorldSettings(settings))
        .insert_resource(Tool {
            radius: 3.5,
            cooldown: 0.0,
            last_center: None,
            hit: None,
            strokes: 0,
            notice: "Click to explore",
            smoke: args.iter().any(|a| a == "--smoke-test"),
            smoke_stage: 0,
            settle: 0.0,
        })
        .init_resource::<Meshing>()
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.68, 0.78, 1.0),
            brightness: 120.0,
            ..default()
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "TERRA — Eroded voxel landscape".into(),
                resolution: (1440, 900).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                finish_generation,
                explorer_input,
                sculpt,
                remesh,
                update_hud,
                smoke_test,
            )
                .chain(),
        )
        .run();
}

#[derive(Resource)]
struct WorldSettings(Settings);

fn setup(
    mut commands: Commands,
    settings: Res<WorldSettings>,
    mut jobs: ResMut<Meshing>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut media: ResMut<Assets<ScatteringMedium>>,
) {
    let settings = settings.0;
    commands.insert_resource(Generation {
        started: Instant::now(),
        task: Some(AsyncComputeTaskPool::get().spawn(async move { Landscape::generate(settings) })),
    });
    jobs.material = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            perceptual_roughness: 0.93,
            reflectance: 0.18,
            ..default()
        },
        extension: Ground {
            detail_strength: 0.19,
        },
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 28_000.0,
            color: Color::srgb(1.0, 0.91, 0.77),
            shadow_maps_enabled: true,
            shadow_normal_bias: 1.0,
            ..default()
        },
        Transform::from_xyz(-100.0, 125.0, 70.0).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder {
            maximum_distance: 260.0,
            first_cascade_far_bound: 35.0,
            ..default()
        }
        .build(),
    ));
    let earth = media.add(ScatteringMedium::earth(128, 128));
    commands.spawn(Atmosphere::earth(earth));
    commands
        .spawn((
            Camera3d::default(),
            Projection::Perspective(PerspectiveProjection {
                fov: 60.0_f32.to_radians(),
                far: 1200.0,
                ..default()
            }),
            Hdr,
            Msaa::Off,
            TemporalAntiAliasing::default(),
            ScreenSpaceAmbientOcclusion::default(),
            AtmosphereSettings::default(),
            AtmosphereEnvironmentMapLight::default(),
            Exposure { ev100: 12.0 },
            Tonemapping::AcesFitted,
            DistanceFog {
                color: Color::srgb(0.64, 0.72, 0.79),
                falloff: FogFalloff::Exponential { density: 0.0018 },
                ..default()
            },
            Transform::from_xyz(50.0, 70.0, 100.0).looking_at(Vec3::new(0.0, 20.0, 20.0), Vec3::Y),
            Explorer {
                yaw: 0.0,
                pitch: 0.0,
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Headlamp,
                SpotLight {
                    intensity: 220_000.0,
                    range: 60.0,
                    inner_angle: 0.35,
                    outer_angle: 0.65,
                    ..default()
                },
                Visibility::Hidden,
                Transform::default(),
            ));
        });
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(28),
            top: px(24),
            padding: UiRect::all(px(18)),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
            border_radius: BorderRadius::all(px(8)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.025, 0.042, 0.044, 0.86)),
        children![
            (
                Text::new("T E R R A   /   FIELD STUDY 01"),
                TextFont {
                    font_size: FontSize::Px(21.0),
                    ..default()
                },
                TextColor(Color::srgb(0.87, 0.91, 0.83))
            ),
            (
                Text::new("Shaping the landscape…"),
                StatusText,
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgb(0.65, 0.75, 0.70))
            )
        ],
    ));
    commands.spawn((Text::new("WASD / arrows  move     Mouse  look     Q/E  descend / rise     Shift  faster\nLeft click  dig     Right click  fill     Wheel / [ ]  brush     L  lamp     Esc  release     F12  capture"),
        TextFont { font_size: FontSize::Px(14.0), ..default() },
        TextColor(Color::srgb(0.88,0.93,0.90)),
        Node { position_type: PositionType::Absolute, left: px(28), bottom: px(24),
            padding: UiRect::all(px(14)), border_radius: BorderRadius::all(px(6)), ..default() },
        BackgroundColor(Color::srgba(0.025,0.042,0.044,0.86))));
    commands.spawn((
        Text::new("+"),
        TextFont {
            font_size: FontSize::Px(24.0),
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.65)),
        Node {
            position_type: PositionType::Absolute,
            left: percent(50),
            top: percent(50),
            ..default()
        },
    ));
}

fn finish_generation(
    mut commands: Commands,
    mut generation: ResMut<Generation>,
    mut jobs: ResMut<Meshing>,
    mut camera: Single<(&mut Transform, &mut Explorer)>,
) {
    let Some(task) = generation.task.as_mut() else {
        return;
    };
    let Some(landscape) = block_on(poll_once(task)) else {
        return;
    };
    jobs.generation_secs = generation.started.elapsed().as_secs_f32();
    info!("Erosion completed in {:.2}s", jobs.generation_secs);
    let position = Vec3::new(54.0, landscape.height(54.0, 100.0) + 18.0, 100.0);
    let target = Vec3::new(10.0, landscape.height(10.0, 35.0), 35.0);
    let transform = Transform::from_translation(position).looking_at(target, Vec3::Y);
    let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
    camera.0.clone_from(&transform);
    camera.1.yaw = yaw;
    camera.1.pitch = pitch;
    let terrain = Terrain::new(landscape);
    jobs.pending.extend(terrain.initial_coords());
    commands.insert_resource(terrain);
    generation.task = None;
}

#[allow(clippy::too_many_arguments)] // Bevy injects system parameters independently.
fn explorer_input(
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    time: Res<Time>,
    mut camera: Single<(&mut Transform, &mut Explorer)>,
    mut window: Single<(&Window, &mut CursorOptions), With<PrimaryWindow>>,
    mut lamp: Single<&mut Visibility, With<Headlamp>>,
    mut tool: ResMut<Tool>,
    mut commands: Commands,
) {
    let (window, cursor) = &mut *window;
    if !window.focused || keys.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    } else if buttons.just_pressed(MouseButton::Left)
        && cursor.grab_mode == CursorGrabMode::None
        && !tool.smoke
    {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
        tool.cooldown = time.delta_secs() + 0.2;
        tool.notice = "Exploring";
    }
    if keys.just_pressed(KeyCode::F12) {
        if let Err(error) = std::fs::create_dir_all("captures") {
            error!("Capture directory: {error}");
        } else {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("captures/terrain.png"));
        }
    }
    if keys.just_pressed(KeyCode::KeyL) {
        **lamp = if **lamp == Visibility::Hidden {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    if cursor.grab_mode != CursorGrabMode::Locked || tool.smoke {
        return;
    }
    let (transform, explorer) = &mut *camera;
    explorer.yaw -= motion.delta.x * 0.002;
    explorer.pitch = (explorer.pitch - motion.delta.y * 0.002).clamp(-1.5, 1.5);
    transform.rotation = Quat::from_euler(EulerRot::YXZ, explorer.yaw, explorer.pitch, 0.0);
    let mut movement = Vec3::ZERO;
    if keys.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]) {
        movement += *transform.forward();
    }
    if keys.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]) {
        movement -= *transform.forward();
    }
    if keys.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]) {
        movement += *transform.right();
    }
    if keys.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]) {
        movement -= *transform.right();
    }
    if keys.pressed(KeyCode::KeyE) {
        movement += Vec3::Y;
    }
    if keys.pressed(KeyCode::KeyQ) {
        movement -= Vec3::Y;
    }
    let speed = if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
        48.0
    } else {
        14.0
    };
    transform.translation += movement.normalize_or_zero() * speed * time.delta_secs().min(0.05);
    transform.translation = transform.translation.clamp(
        Vec3::new(-185.0, -43.0, -185.0),
        Vec3::new(185.0, 180.0, 185.0),
    );
}

#[allow(clippy::too_many_arguments)] // Bevy injects system parameters independently.
fn sculpt(
    mut terrain: Option<ResMut<Terrain>>,
    mut jobs: ResMut<Meshing>,
    mut tool: ResMut<Tool>,
    camera: Single<&Transform, With<Explorer>>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    mut gizmos: Gizmos,
) {
    tool.cooldown = (tool.cooldown - time.delta_secs()).max(0.0);
    let Some(terrain) = terrain.as_mut() else {
        return;
    };
    if tool.smoke {
        return;
    }
    let change = scroll.delta.y * 0.35
        + if keys.just_pressed(KeyCode::BracketRight) {
            0.5
        } else {
            0.0
        }
        - if keys.just_pressed(KeyCode::BracketLeft) {
            0.5
        } else {
            0.0
        };
    tool.radius = (tool.radius + change).clamp(1.5, 12.0);
    tool.hit = terrain.raycast(camera.translation, *camera.forward(), 90.0);
    let Some(hit) = tool.hit else {
        tool.last_center = None;
        return;
    };
    if cursor.grab_mode != CursorGrabMode::Locked {
        return;
    }
    let normal = terrain.normal(hit);
    gizmos.circle(
        Isometry3d::new(
            hit + normal * 0.06,
            Quat::from_rotation_arc(Vec3::Z, normal),
        ),
        tool.radius,
        Color::srgba(0.89, 0.94, 0.73, 0.7),
    );
    let digging = buttons.pressed(MouseButton::Left);
    let adding = buttons.pressed(MouseButton::Right);
    if !digging && !adding {
        tool.last_center = None;
        return;
    }
    if tool.cooldown > 0.0 {
        return;
    }
    let center = hit
        + normal
            * if adding {
                tool.radius * 0.4
            } else {
                -tool.radius * 0.35
            };
    if tool.last_center.is_some_and(|p| p.distance(center) < 0.25) {
        return;
    }
    let dirty = terrain.brush(center, tool.radius, adding);
    if !dirty.is_empty() {
        jobs.pending.extend(dirty);
        tool.strokes += 1;
        tool.notice = if adding {
            "Depositing terrain"
        } else {
            "Excavating"
        };
    }
    tool.cooldown = 0.08;
    tool.last_center = Some(center);
}

fn remesh(
    mut commands: Commands,
    mut terrain: Option<ResMut<Terrain>>,
    mut jobs: ResMut<Meshing>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera: Single<&Transform, With<Explorer>>,
) {
    let Some(terrain) = terrain.as_mut() else {
        return;
    };
    let mut ready = Vec::new();
    for (&coord, job) in &mut jobs.running {
        if ready.len() == 3 {
            break;
        }
        if let Some((data, cpu_ms)) = block_on(poll_once(&mut job.task)) {
            ready.push((coord, job.revision, data, cpu_ms));
        }
    }
    for (coord, revision, data, cpu_ms) in ready {
        jobs.running.remove(&coord);
        if terrain.chunks[&coord].revision != revision {
            jobs.rejected += 1;
            jobs.pending.insert(coord);
            continue;
        }
        jobs.last_ms = cpu_ms;
        let triangles = data.indices.len() / 3;
        if triangles == 0 {
            if let Some(old) = jobs.displayed.remove(&coord) {
                commands.entity(old.entity).despawn();
            }
            continue;
        }
        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, data.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, data.terrain_uv)
        .with_inserted_indices(Indices::U32(data.indices));
        if let Some(old) = jobs.displayed.get_mut(&coord) {
            if let Some(mut asset) = meshes.get_mut(&old.mesh) {
                *asset = mesh;
            }
            old.triangles = triangles;
        } else {
            let handle = meshes.add(mesh);
            let entity = commands
                .spawn((
                    Mesh3d(handle.clone()),
                    MeshMaterial3d(jobs.material.clone()),
                    Transform::default(),
                ))
                .id();
            jobs.displayed.insert(
                coord,
                DisplayChunk {
                    entity,
                    mesh: handle,
                    triangles,
                },
            );
        }
    }
    // ponytail: nearest-first scan is cheap for this finite world; use a heap for streaming worlds.
    while jobs.running.len() < 4 {
        let candidate = jobs
            .pending
            .iter()
            .filter(|c| !jobs.running.contains_key(c))
            .min_by(|a, b| {
                chunk_origin(**a)
                    .distance_squared(camera.translation)
                    .total_cmp(&chunk_origin(**b).distance_squared(camera.translation))
            })
            .copied();
        let Some(coord) = candidate else {
            break;
        };
        jobs.pending.remove(&coord);
        terrain.ensure_chunk(coord);
        let chunk = &terrain.chunks[&coord];
        let (revision, samples, landscape) = (
            chunk.revision,
            chunk.samples.clone(),
            terrain.landscape.clone(),
        );
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let start = Instant::now();
            let data = polygonize(coord, &samples, &landscape);
            (data, start.elapsed().as_secs_f32() * 1000.0)
        });
        jobs.running.insert(coord, Job { revision, task });
    }
}

fn update_hud(
    jobs: Res<Meshing>,
    tool: Res<Tool>,
    terrain: Option<Res<Terrain>>,
    time: Res<Time>,
    mut text: Single<&mut Text, With<StatusText>>,
    mut elapsed: Local<f32>,
) {
    *elapsed += time.delta_secs();
    if *elapsed < 0.25 {
        return;
    }
    *elapsed = 0.0;
    if let Some(terrain) = terrain {
        let triangles: usize = jobs.displayed.values().map(|c| c.triangles).sum();
        let memory = terrain.chunks.len() * Shape::USIZE * 4;
        text.0 = format!(
            "{}   /   brush {:.1} m\n384 m landscape  /  {} surfaces  /  {:.0}k triangles\nField {:.1} MiB   /   CPU mesh {:.2} ms   /   pending {}\n{} edits   /   {} stale jobs rejected",
            tool.notice,
            tool.radius,
            jobs.displayed.len(),
            triangles as f32 / 1000.0,
            memory as f32 / 1048576.0,
            jobs.last_ms,
            jobs.pending.len() + jobs.running.len(),
            tool.strokes,
            jobs.rejected
        );
    }
}

/// Exercise the actual renderer and excavation path; captures are saved before and after an edit.
fn smoke_test(
    mut commands: Commands,
    time: Res<Time>,
    mut tool: ResMut<Tool>,
    mut terrain: Option<ResMut<Terrain>>,
    mut jobs: ResMut<Meshing>,
    mut camera: Single<(&mut Transform, &mut Explorer)>,
    mut exit: MessageWriter<AppExit>,
) {
    if !tool.smoke {
        return;
    }
    let Some(terrain) = terrain.as_mut() else {
        return;
    };
    if !jobs.pending.is_empty() || !jobs.running.is_empty() {
        tool.settle = 0.0;
        return;
    }
    tool.settle += time.delta_secs();
    if tool.settle < 2.0 {
        return;
    }
    match tool.smoke_stage {
        0 => {
            if let Err(error) = std::fs::create_dir_all("captures") {
                error!("{error}");
                exit.write(AppExit::error());
                return;
            }
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("captures/landscape.png"));
            tool.smoke_stage = 1;
            tool.settle = 0.0;
        }
        1 => {
            let target = Vec3::new(24.0, terrain.landscape.height(24.0, 52.0), 52.0);
            let position = target + Vec3::new(10.0, 12.0, 18.0);
            *camera.0 = Transform::from_translation(position).looking_at(target, Vec3::Y);
            tool.smoke_stage = 2;
            tool.settle = 0.0;
        }
        2 => {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("captures/before-dig.png"));
            tool.smoke_stage = 3;
            tool.settle = 0.0;
        }
        3 => {
            let Some(hit) = terrain.raycast(camera.0.translation, *camera.0.forward(), 90.0) else {
                error!("Smoke test failed to hit terrain");
                exit.write(AppExit::error());
                return;
            };
            let center = hit - Vec3::Y * 2.0;
            let dirty = terrain.brush(center, 7.0, false);
            assert!(!dirty.is_empty(), "smoke excavation must change samples");
            assert!(
                terrain.sample(center) > 0.0,
                "excavation must remove solid terrain"
            );
            jobs.pending.extend(dirty);
            tool.strokes += 1;
            tool.notice = "Excavation verified";
            tool.smoke_stage = 4;
            tool.settle = 0.0;
        }
        4 => {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("captures/after-dig.png"));
            info!(
                "Smoke test: {} chunks, {} triangles, field {:.1} MiB, erosion {:.2}s, last remesh {:.2}ms",
                jobs.displayed.len(),
                jobs.displayed.values().map(|c| c.triangles).sum::<usize>(),
                terrain.chunks.len() as f32 * Shape::USIZE as f32 * 4.0 / 1048576.0,
                jobs.generation_secs,
                jobs.last_ms
            );
            tool.smoke_stage = 5;
            tool.settle = 0.0;
        }
        _ => {
            exit.write(AppExit::Success);
        }
    }
}
