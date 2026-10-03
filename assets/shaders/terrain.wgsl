#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> detail_strength: f32;

fn hash(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f*f*(3.0-2.0*f);
    return mix(mix(mix(hash(i), hash(i+vec3(1.0,0.0,0.0)), u.x),
        mix(hash(i+vec3(0.0,1.0,0.0)), hash(i+vec3(1.0,1.0,0.0)), u.x), u.y),
        mix(mix(hash(i+vec3(0.0,0.0,1.0)), hash(i+vec3(1.0,0.0,1.0)), u.x),
        mix(hash(i+vec3(0.0,1.0,1.0)), hash(i+vec3(1.0,1.0,1.0)), u.x), u.y), u.z);
}

fn micro_height(p: vec3<f32>) -> f32 {
    return noise(p*1.7)*0.55 + noise(p*6.0)*0.28 + noise(p*18.0)*0.12;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, front);
    let p = in.world_position.xyz;
    let n = normalize(in.world_normal);
    let wet = clamp(in.uv.y,0.0,1.0);
    let depth = max(in.uv.x-p.y,0.0);
    let broad = noise(p*0.055);
    let grain = micro_height(p);
    let strata = smoothstep(0.2,0.8,noise(vec3(p.x*0.08,p.y*0.8 + broad*2.0,p.z*0.08)));
    let rock = mix(vec3(0.13,0.12,0.105),vec3(0.36,0.32,0.26),strata) * (0.72+grain*0.50);
    let dirt = mix(vec3(0.19,0.11,0.055),vec3(0.36,0.27,0.16),broad) * (0.78+grain*0.34);
    let grass = mix(vec3(0.075,0.105,0.025),vec3(0.22,0.27,0.07),broad) * (0.70+grain*0.50);
    let cliff = 1.0-smoothstep(0.55,0.86,n.y);
    let buried = smoothstep(0.5,2.4,depth);
    let bedrock = smoothstep(2.0,5.0,depth);
    let vegetation = smoothstep(0.3,0.65,broad + wet*0.18) * (1.0-buried);
    var albedo = mix(dirt,grass,vegetation);
    albedo = mix(albedo,rock,max(cliff,bedrock));
    let snow = smoothstep(68.0,91.0,p.y + broad*7.0) * smoothstep(0.55,0.9,n.y) * (1.0-buried);
    albedo = mix(albedo,vec3(0.72,0.78,0.80)*(0.90+grain*0.10),snow);
    albedo *= 1.0 - wet*0.13;
    pbr.material.base_color=vec4(albedo,1.0);
    pbr.material.perceptual_roughness=mix(0.95,0.73,wet)*(1.0-snow*0.1);
    // Surface-gradient bump mapping: world-space detail is continuous over cliffs and excavations.
    // Fade frequencies below the pixel footprint to avoid distant specular shimmer.
    let px = dpdx(p);
    let py = dpdy(p);
    let h = grain;
    let dhdx = dpdx(h);
    let dhdy = dpdy(h);
    let r1 = cross(py,n);
    let r2 = cross(n,px);
    let det = dot(px,r1);
    let fade = 1.0-smoothstep(0.08,0.7,max(length(px),length(py)));
    let gradient = (r1*dhdx+r2*dhdy)*sign(det)/max(abs(det),0.00001);
    pbr.N=normalize(n-gradient*detail_strength*fade);
    var out: FragmentOutput;
    out.color=apply_pbr_lighting(pbr);
    out.color=main_pass_post_lighting_processing(pbr,out.color);
    return out;
}
