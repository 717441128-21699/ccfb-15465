// 西汉透光镜 WebGPU 着色器集合
// 管线:平滑 -> 雅可比求解(泊松方程) -> 归约(统计) -> 反光成像(gather) -> 可视化

export const PARAMS_WGSL = /* wgsl */ `
struct Params {
  // 标量区(16 个 f32,64 字节)
  mirrorD: f32,        // 镜直径 (m)
  dWall: f32,          // 镜到墙距离 (m)
  incidenceDeg: f32,   // 入射角 (度)
  sourceRadiusRad: f32,// 光源角半径 (rad)
  baseCurve: f32,      // 基底凸度曲率 kappa = 1/R (1/m)
  hScale: f32,         // 归一化起伏 -> 物理高度的比例 (m)
  rmsNorm: f32,        // 归一化高度场的 RMS
  grind: f32,          // 研磨程度 0..1
  wallHalfW: f32,      // 墙画幅半宽 (m)
  wallHalfH: f32,      // 墙画幅半高 (m)
  res: f32,            // 镜面网格分辨率 N
  inRes: f32,          // 归约输入分辨率
  viewMode: f32,       // 可视化模式 0厚度 1起伏 2墙面
  exposure: f32,       // 曝光
  wallRes: f32,        // 墙纹理分辨率
  alpha: f32,          // 泊松求解系数(冷却阶段使用)
  // 向量区
  iDir: vec4f,         // 入射光传播方向(单位向量) xyz
  wallB1: vec4f,       // 墙面基向量 1
  wallB2: vec4f,       // 墙面基向量 2
  n0: vec4f,           // 镜面中心法线
};
@group(0) @binding(0) var<uniform> P: Params;
`;

// ---------- 平滑(盒式滤波两次,近似高斯,模拟凝固层应力扩散) ----------
export const SMOOTH_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(1) var tIn: texture_2d<f32>;
@group(0) @binding(2) var tOut: texture_storage_2d<r32float, write>;

fn loadC(c: vec2i) -> f32 {
  let n = i32(P.res);
  let cc = clamp(c, vec2i(0), vec2i(n - 1));
  return textureLoad(tIn, cc, 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec2i(gid.xy);
  if (c.x >= i32(P.res) || c.y >= i32(P.res)) { return; }
  let v = loadC(c) * 2.0
    + loadC(c + vec2i(1, 0)) + loadC(c - vec2i(1, 0))
    + loadC(c + vec2i(0, 1)) + loadC(c - vec2i(0, 1));
  textureStore(tOut, c, vec4f(v * 0.16666667, 0.0, 0.0, 1.0));
}
`;

// ---------- 雅可比迭代:∇²h = -α(t - mean),厚薄差 -> 曲率差 -> 高度起伏 ----------
// Neumann(零通量)边界:边界外法线方向导数为 0,符合镜面边缘无外加约束的物理情形
export const JACOBI_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(1) var hIn: texture_2d<f32>;
@group(0) @binding(2) var hOut: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var tSrc: texture_2d<f32>;

fn loadH(c: vec2i) -> f32 {
  let n = i32(P.res);
  return textureLoad(hIn, clamp(c, vec2i(0), vec2i(n - 1)), 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec2i(gid.xy);
  if (c.x >= i32(P.res) || c.y >= i32(P.res)) { return; }
  let mean = P.n0.w;       // 复用 n0.w 传厚度均值
  let alpha = P.alpha;     // 泊松系数 α
  let tv = textureLoad(tSrc, c, 0).r;
  let sum = loadH(c + vec2i(1, 0)) + loadH(c - vec2i(1, 0))
          + loadH(c + vec2i(0, 1)) + loadH(c - vec2i(0, 1));
  // 网格间距取 1(格点单位),h = 1/4 * (邻域和 + α(t-mean))
  let h = 0.25 * (sum + alpha * (tv - mean));
  textureStore(hOut, c, vec4f(h, 0.0, 0.0, 1.0));
}
`;

// ---------- 归约:16x16 工作组 -> 单像素(sum, sumSq),逐级 16 倍下采样 ----------
// 输入尺寸 N 编译期固化,完全不读 uniform,避免与主参数缓冲的写入排队竞争

// 首层级:r32float 输入,v 与 v² 现场计算
export const REDUCE_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(1) var tIn: texture_2d<f32>;
@group(0) @binding(2) var tOut: texture_storage_2d<rg32float, write>;
const N = 512u;

var<workgroup> shA: array<f32, 256>;
var<workgroup> shB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let src = wid.xy * vec2u(16u) + lid.xy;
  var v = 0.0;
  if (src.x < N && src.y < N) { v = textureLoad(tIn, vec2i(src), 0).r; }
  let idx = lid.y * 16u + lid.x;
  shA[idx] = v;
  shB[idx] = v * v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (idx < s) {
      shA[idx] = shA[idx] + shA[idx + s];
      shB[idx] = shB[idx] + shB[idx + s];
    }
    workgroupBarrier();
  }
  if (idx == 0u) {
    textureStore(tOut, vec2i(wid.xy), vec4f(shA[0], shB[0], 0.0, 1.0));
  }
}
`;

// 后续层级:rg32float 输入,.r 为部分和、.g 为部分平方和,尺寸编译期固化
export function reduceRGWGSL(n: number): string {
  return /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(1) var tIn: texture_2d<f32>;
@group(0) @binding(2) var tOut: texture_storage_2d<rg32float, write>;
const N = ${n}u;

var<workgroup> shA: array<f32, 256>;
var<workgroup> shB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let src = wid.xy * vec2u(16u) + lid.xy;
  var v = 0.0;
  var v2 = 0.0;
  if (src.x < N && src.y < N) {
    let z = textureLoad(tIn, vec2i(src), 0);
    v = z.r;
    v2 = z.g;
  }
  let idx = lid.y * 16u + lid.x;
  shA[idx] = v;
  shB[idx] = v2;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (idx < s) {
      shA[idx] = shA[idx] + shA[idx + s];
      shB[idx] = shB[idx] + shB[idx + s];
    }
    workgroupBarrier();
  }
  if (idx == 0u) {
    textureStore(tOut, vec2i(wid.xy), vec4f(shA[0], shB[0], 0.0, 1.0));
  }
}
`;
}

// ---------- 反光成像:逐墙像素反向映射到镜面,按局部微起伏曲率计算汇聚/发散亮度 ----------
// 直接由高度场生成墙面光斑,无需原子累加;凹区汇聚增亮、凸区发散减暗
export const WALL_CLEAR_WGSL = /* wgsl */ `
@group(0) @binding(0) var tOut: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= 512u || gid.y >= 512u) { return; }
  textureStore(tOut, vec2i(gid.xy), vec4f(0.028, 0.032, 0.042, 1.0));
}
`;

// 主投影:每个墙像素读取对应镜面位置的高度与曲率,生成暖色调明暗
export const PROJECT_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(1) var tH: texture_2d<f32>;
@group(0) @binding(2) var tOut: texture_storage_2d<rgba16float, write>;

fn hload(c: vec2i) -> f32 {
  let n = i32(P.res);
  return textureLoad(tH, clamp(c, vec2i(0), vec2i(n - 1)), 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let W = i32(P.wallRes);
  if (gid.x >= u32(W) || gid.y >= u32(W)) { return; }
  let c = vec2i(gid.xy);
  let bg = vec3f(0.028, 0.032, 0.042);

  // 圆形光束软遮罩
  let pc = (vec2f(f32(c.x), f32(c.y)) + 0.5) / P.wallRes - vec2f(0.5);
  let beamMask = 1.0 - smoothstep(0.48, 0.5, length(pc));
  if (beamMask < 0.01) {
    textureStore(tOut, c, vec4f(bg, 1.0));
    return;
  }

  // 墙像素 -> 镜面格点(墙与镜同分辨率,中心对齐)
  let mc = clamp(c, vec2i(0), vec2i(i32(P.res) - 1));
  let hc = hload(mc);

  // 局部曲率(拉普拉斯):凹区 lap<0 汇聚增亮,凸区 lap>0 发散减暗
  let lap = hload(mc + vec2i(1, 0)) + hload(mc - vec2i(1, 0))
          + hload(mc + vec2i(0, 1)) + hload(mc - vec2i(0, 1)) - 4.0 * hc;
  let rms = max(P.rmsNorm, 1.0e-6);

  // 光源角直径 -> 模糊半径(墙像素),大光源平滑掉高频、花纹变柔
  let blurPx = clamp(i32(P.sourceRadiusRad * P.dWall / (2.0 * P.wallHalfW / P.wallRes)), 0, 12);
  var hBlur = 0.0;
  var lapBlur = 0.0;
  var wsum = 0.0;
  for (var by = -4; by <= 4; by++) {
    for (var bx = -4; bx <= 4; bx++) {
      let take = select(0, 1, abs(bx) <= blurPx && abs(by) <= blurPx);
      let sp = mc + vec2i(bx, by) * take;
      let hh = hload(sp);
      let ll = hload(sp + vec2i(1, 0)) + hload(sp - vec2i(1, 0))
             + hload(sp + vec2i(0, 1)) + hload(sp - vec2i(0, 1)) - 4.0 * hh;
      hBlur = hBlur + hh * f32(take);
      lapBlur = lapBlur + ll * f32(take);
      wsum = wsum + f32(take);
    }
  }
  hBlur = hBlur / max(wsum, 1.0);
  lapBlur = lapBlur / max(wsum, 1.0);

  // 距离放大因子:反射偏转随墙距 L 增大,远墙花纹对比更强(也更发散)
  let distGain = clamp(0.4 + P.dWall / 2.0, 0.5, 2.5);
  let heightTone = 0.5 + 0.9 * hBlur / (2.5 * rms);
  let curvatureTone = -1.5 * distGain * lapBlur / rms;
  let v = clamp(P.exposure * (heightTone + curvatureTone), 0.03, 0.98);
  let col = mix(bg, vec3f(v, v * 0.85, v * 0.62), beamMask);
  textureStore(tOut, c, vec4f(col, 1.0));
}
`;

// ---------- 全屏渲染:厚度 / 起伏明暗 / 墙面光斑 ----------
export const RENDER_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(1) var tScalar: texture_2d<f32>;
@group(0) @binding(2) var tWall: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  // 标准全屏三角形:三条边均延伸到裁剪空间外,完整覆盖 [-1,1]²
  var p = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  var o: VSOut;
  let xy = p[vi];
  o.pos = vec4f(xy, 0.0, 1.0);
  // 裁剪空间 y 向上,纹理 v 向下
  o.uv = vec2f(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
  return o;
}

fn loadC(t: texture_2d<f32>, c: vec2i) -> f32 {
  let n = i32(P.res);
  return textureLoad(t, clamp(c, vec2i(0), vec2i(n - 1)), 0).r;
}

@fragment
fn fs(o: VSOut) -> @location(0) vec4f {
  let N = i32(P.res);
  let c = vec2i(o.uv * vec2f(P.res));
  let r = length(o.uv - vec2f(0.5)) * 2.0;
  let mask = select(0.0, 1.0, r <= 1.0);

  if (u32(P.viewMode) == 2u) {
    let wc = clamp(vec2i(o.uv * vec2f(P.wallRes)), vec2i(0), vec2i(i32(P.wallRes) - 1));
    let w = textureLoad(tWall, wc, 0);
    return vec4f(w.rgb, 1.0);
  }

  let v = loadC(tScalar, c);
  let bronzeDark = vec3f(0.16, 0.09, 0.035);
  let bronzeLight = vec3f(0.82, 0.62, 0.32);

  if (u32(P.viewMode) == 0u) {
    // 厚度:灰度映射到铜色
    let g = clamp(v, 0.0, 1.0);
    let col = mix(bronzeDark, bronzeLight, g);
    let rim = select(0.0, 1.0, r > 0.94 && r <= 1.0);
    let rimmed = col * (1.0 - 0.35 * rim) + vec3f(0.25, 0.17, 0.08) * rim;
    let inside = rimmed * mask;
    let outside = vec3f(0.05, 0.05, 0.06) * (1.0 - mask);
    return vec4f(inside + outside, 1.0);
  }

  // 起伏:斜率做方向光阴影,高度做等值线着色(幅度已放大,仅用于可视化)
  let hl = loadC(tScalar, c + vec2i(-1, 0));
  let hr = loadC(tScalar, c + vec2i(1, 0));
  let hu = loadC(tScalar, c + vec2i(0, -1));
  let hd = loadC(tScalar, c + vec2i(0, 1));
  let k = P.hScale * P.res / P.mirrorD;
  let sx = k * (hr - hl);
  let sy = k * (hd - hu);
  var n = normalize(vec3f(-sx * 3.0e4, -sy * 3.0e4, 1.0)); // 视觉放大 3 万倍
  let ldir = normalize(vec3f(0.35, -0.45, 0.82));
  let diff = clamp(dot(n, ldir), 0.0, 1.0);
  let hn = clamp(v * 0.5 + 0.5, 0.0, 1.0);
  let base = mix(bronzeDark, bronzeLight, hn);
  let contour = 0.05 * sin(v * 40.0);
  let col = base * (0.35 + 0.75 * diff) + vec3(contour);
  return vec4f(col * mask + vec3f(0.05, 0.05, 0.06) * (1.0 - mask), 1.0);
}
`;
