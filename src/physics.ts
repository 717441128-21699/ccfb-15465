// 冷却收缩的唯象物理模型
// 核心关系(见原理页):
//   凝固收缩 → 厚薄不均产生应力 → 弹性弯曲 → 表面曲率扰动 κ' ∝ 厚度差 Δt
//   泊松方程 ∇²h = -α(t - t̄)  即 曲率(高度二阶导)正比于厚度偏差
//
// 幅度标定:
//   h_rms = K · ε · D² · σ_t · g(研磨程度)
//   ε   铜锡合金(锡青铜,约 23% Sn)凝固线收缩率 ≈ 1.6%
//   D   镜直径(mm),典型西汉镜约 80 mm
//   σ_t 镜背厚度标准差(mm),由当前纹样实测得到
//   K   无量纲结构系数,由文物检测报告的 0.1–1 μm 数量级反标
//   g   研磨系数:镜面磨得越薄,残余约束越弱、起伏越明显

export const ALLOY_SHRINKAGE = 0.016; // ε:锡青铜凝固线收缩率 1.6%
export const MIRROR_DIAMETER_M = 0.08; // 80 mm
export const STRUCTURE_COEF = 0.01; // K:无量纲结构系数(实测反标)
export const MIN_AMPLITUDE_UM = 0.1;
export const MAX_AMPLITUDE_UM = 1.0;

export interface ThicknessStats {
  mean: number; // 归一化均值 0..1
  std: number; // 归一化标准差 0..1
  meanMm: number; // 平均厚度 mm
  stdMm: number; // 厚度标准差 mm
}

export function thicknessInMm(normalized: number): number {
  // 镜体厚度 0.8 mm(薄处)~ 2.2 mm(厚处/纹饰)
  return 0.8 + 1.4 * normalized;
}

/** 研磨程度 0..1 → 研磨系数 0.5..1.5(磨得越薄起伏越大) */
export function grindFactor(grind: number): number {
  return 0.5 + grind;
}

/**
 * 按合金收缩率与镜体厚薄分布推算目标起伏幅度(RMS,微米),
 * 并夹到文物检测报告的 0.1–1 μm 区间。
 */
export function targetAmplitudeUm(stdMm: number, grind: number): number {
  const dMm = MIRROR_DIAMETER_M * 1000;
  const raw =
    STRUCTURE_COEF * ALLOY_SHRINKAGE * dMm * dMm * stdMm * grindFactor(grind);
  return Math.min(MAX_AMPLITUDE_UM, Math.max(MIN_AMPLITUDE_UM, raw));
}

export interface AmplitudeReport {
  amplitudeUm: number;
  rawUm: number;
  clamped: boolean;
  pvEstimateUm: number;
}

export function amplitudeReport(stats: ThicknessStats, grind: number): AmplitudeReport {
  const raw =
    STRUCTURE_COEF *
    ALLOY_SHRINKAGE *
    (MIRROR_DIAMETER_M * 1000) ** 2 *
    stats.stdMm *
    grindFactor(grind);
  const amplitudeUm = Math.min(MAX_AMPLITUDE_UM, Math.max(MIN_AMPLITUDE_UM, raw));
  return {
    amplitudeUm,
    rawUm: raw,
    clamped: raw < MIN_AMPLITUDE_UM || raw > MAX_AMPLITUDE_UM,
    // 近似高斯场时峰峰值约为 RMS 的 5 倍
    pvEstimateUm: amplitudeUm * 5,
  };
}

/** 由入射方向(传播向量)构造垂直于它的严格正交基,墙平面法线即入射轴 */
export function wallBasis(iDir: [number, number, number]): {
  b1: [number, number, number];
  b2: [number, number, number];
} {
  const [x, y, z] = iDir;
  // 选一个不平行于 iDir 的参考轴
  const ref: [number, number, number] = Math.abs(z) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  // b1 = normalize(cross(ref, iDir)),垂直于 iDir
  let b1x = ref[1] * z - ref[2] * y;
  let b1y = ref[2] * x - ref[0] * z;
  let b1z = ref[0] * y - ref[1] * x;
  const l1 = Math.hypot(b1x, b1y, b1z);
  b1x /= l1; b1y /= l1; b1z /= l1;
  // b2 = cross(iDir, b1),与 iDir、b1 都正交,构成右手系
  const b2x = y * b1z - z * b1y;
  const b2y = z * b1x - x * b1z;
  const b2z = x * b1y - y * b1x;
  return { b1: [b1x, b1y, b1z], b2: [b2x, b2y, b2z] };
}

/** 入射角(度,相对镜面法线)→ 入射光传播单位向量 */
export function incidentDirection(deg: number): [number, number, number] {
  const t = (deg * Math.PI) / 180;
  return [Math.sin(t), 0, Math.cos(t)];
}

/**
 * 自动取景:墙垂直于入射轴,光束在墙上的半径 =
 * 镜盘投影半径 + 凸度/微起伏发散 + 光源角展宽,沿入射轴距离为 dWall
 */
export function wallHalfSize(
  dWall: number,
  mirrorD: number,
  sourceAngularRadiusRad: number,
): number {
  const divergenceSlope = 0.06; // 凸面镜曲率 + 微起伏典型发散斜率
  const mirrorProjected = mirrorD / 2;
  // 留足余量,保证整个光束(含边缘光线与光源展宽)都落在画幅内,避免直线裁切
  return (mirrorProjected + dWall * (sourceAngularRadiusRad + divergenceSlope)) * 1.6;
}
