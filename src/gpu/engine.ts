import {
  JACOBI_WGSL,
  PROJECT_WGSL,
  REDUCE_WGSL,
  RENDER_WGSL,
  SMOOTH_WGSL,
  WALL_CLEAR_WGSL,
  reduceRGWGSL,
} from './shaders';
import {
  ThicknessStats,
  amplitudeReport,
  thicknessInMm,
  wallBasis,
  wallHalfSize,
  incidentDirection,
} from '../physics';

export const GRID = 512; // 镜面网格 512×512
const JACOBI_ITERS = 48; // 泊松方程迭代次数
const WALL_RES = 512;

export interface SimParams {
  dWall: number; // 镜-墙距离 (m)
  incidenceDeg: number; // 入射角
  sourceAngularDiamDeg: number; // 光源角直径(度)
  baseCurve: number; // 基底凸度曲率 1/R (1/m)
  grind: number; // 研磨程度 0..1
  exposure: number;
}

export interface CoolingResult {
  stats: ThicknessStats;
  rmsNorm: number;
  hScale: number; // m per normalized unit
  amplitudeUm: number;
  pvUm: number;
  rawUm: number;
  clamped: boolean;
  totalMs: number;
}

const FMT_R = 'r32float' as GPUTextureFormat;
const FMT_RG = 'rg32float' as GPUTextureFormat;
const FMT_WALL = 'rgba16float' as GPUTextureFormat;

function makeStorageTexture(device: GPUDevice, format: GPUTextureFormat, size = GRID): GPUTexture {
  return device.createTexture({
    size: { width: size, height: size },
    format,
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.COPY_SRC,
  });
}

export class MirrorEngine {
  device!: GPUDevice;
  format!: GPUTextureFormat;

  private thickness!: GPUTexture;
  private smooth!: GPUTexture;
  private smoothTmp!: GPUTexture;
  private hA!: GPUTexture;
  private hB!: GPUTexture;
  private wall!: GPUTexture;
  private reduceL0!: GPUTexture; // 32×32 rg
  private reduceL1!: GPUTexture; // 2×2 rg
  private reduceL2!: GPUTexture; // 1×1 rg
  private uniform!: GPUBuffer;
  private sampler!: GPUSampler;

  private smoothPipe!: GPUComputePipeline;
  private jacobiPipe!: GPUComputePipeline;
  private reduceRPipe!: GPUComputePipeline; // r(512) -> rg(32)
  private reduceRG32Pipe!: GPUComputePipeline; // rg(32) -> rg(2)
  private reduceRG2Pipe!: GPUComputePipeline; // rg(2) -> rg(1)
  private projectPipe!: GPUComputePipeline;
  private wallClearPipe!: GPUComputePipeline;
  private clearPipe!: GPUComputePipeline;
  private renderPipe!: GPURenderPipeline;

  private bglSmooth!: GPUBindGroupLayout;
  private bglJacobi!: GPUBindGroupLayout;
  private bglReduceR!: GPUBindGroupLayout;
  private bglReduceRG!: GPUBindGroupLayout;
  private bglProject!: GPUBindGroupLayout;
  private bglWallClear!: GPUBindGroupLayout;
  private bglClear!: GPUBindGroupLayout;
  private bglRender!: GPUBindGroupLayout;

  private canvasCfgKeys = new Map<HTMLCanvasElement, string>();
  private rmsNorm = 0;
  private stats: ThicknessStats = { mean: 0.5, std: 0.1, meanMm: 1.5, stdMm: 0.14 };
  private wallHalfW = 0.1;

  static async create(): Promise<MirrorEngine> {
    if (!('gpu' in navigator)) {
      throw new Error('当前浏览器不支持 WebGPU,请使用最新版 Chrome/Edge。');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('找不到可用的 GPU 适配器。');
    const device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', (e) => {
      console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error);
    });
    const engine = new MirrorEngine();
    engine.device = device;
    MirrorEngine.WRITE_ACCESS = MirrorEngine.detectWriteAccess(device);
    engine.init();
    await engine.ready();
    return engine;
  }

  // 输出:write-only storage texture。
  // Chrome 113–151 枚举为 'write',152+ 按新规范改名为 'write-only',首次拿到设备时探测
  private static WRITE_ACCESS: GPUStorageTextureAccess = 'write' as GPUStorageTextureAccess;

  private static detectWriteAccess(device: GPUDevice): GPUStorageTextureAccess {
    for (const v of ['write-only', 'write']) {
      try {
        device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              storageTexture: { format: 'r32float' as GPUTextureFormat, access: v as unknown as GPUStorageTextureAccess },
            },
          ],
        });
        return v as GPUStorageTextureAccess;
      } catch {
        // 试下一个候选
      }
    }
    return 'write' as GPUStorageTextureAccess;
  }

  private storageEntry(binding: number, format: GPUTextureFormat) {
    return {
      binding,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { format, access: MirrorEngine.WRITE_ACCESS },
    };
  }

  // 输入:只读纹理。r32float / rg32float 不可过滤,用 unfilterable-float
  private textureEntry(binding: number, sampleType: GPUTextureSampleType = 'unfilterable-float') {
    return { binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType } };
  }

  private init() {
    const d = this.device;
    this.format = navigator.gpu.getPreferredCanvasFormat();

    this.thickness = makeStorageTexture(d, FMT_R);
    this.smooth = makeStorageTexture(d, FMT_R);
    this.smoothTmp = makeStorageTexture(d, FMT_R);
    this.hA = makeStorageTexture(d, FMT_R);
    this.hB = makeStorageTexture(d, FMT_R);
    this.wall = makeStorageTexture(d, FMT_WALL, WALL_RES);
    this.reduceL0 = makeStorageTexture(d, FMT_RG, 32);
    this.reduceL1 = makeStorageTexture(d, FMT_RG, 2);
    this.reduceL2 = makeStorageTexture(d, FMT_RG, 1);

    this.uniform = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = d.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    } as GPUSamplerDescriptor);

    const uComp = { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {} };

    this.bglSmooth = d.createBindGroupLayout({
      entries: [uComp, this.textureEntry(1), this.storageEntry(2, FMT_R)],
    });
    this.bglJacobi = d.createBindGroupLayout({
      entries: [uComp, this.textureEntry(1), this.storageEntry(2, FMT_R), this.textureEntry(3)],
    });
    this.bglReduceR = d.createBindGroupLayout({
      entries: [uComp, this.textureEntry(1), this.storageEntry(2, FMT_RG)],
    });
    this.bglReduceRG = d.createBindGroupLayout({
      entries: [uComp, this.textureEntry(1), this.storageEntry(2, FMT_RG)],
    });
    // 投影:uniform + 高度纹理(只读)+ 墙面输出纹理(只写)
    this.bglProject = d.createBindGroupLayout({
      entries: [
        uComp,
        this.textureEntry(1),
        this.storageEntry(2, FMT_WALL),
      ],
    });
    // 清墙:只写墙面纹理
    this.bglWallClear = d.createBindGroupLayout({
      entries: [this.storageEntry(0, FMT_WALL)],
    });
    this.bglClear = d.createBindGroupLayout({
      entries: [this.storageEntry(0, FMT_R)],
    });
    this.bglRender = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    const compPipe = (code: string, layout: GPUBindGroupLayout, label: string) => {
      const mod = d.createShaderModule({ code, label });
      this.modules.push(mod);
      this.moduleLabels.set(mod, label);
      return d.createComputePipeline({
        label,
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: mod, entryPoint: 'main' },
      });
    };

    this.smoothPipe = compPipe(SMOOTH_WGSL, this.bglSmooth, 'smooth');
    this.jacobiPipe = compPipe(JACOBI_WGSL, this.bglJacobi, 'jacobi');
    this.reduceRPipe = compPipe(REDUCE_WGSL, this.bglReduceR, 'reduceR');
    this.reduceRG32Pipe = compPipe(reduceRGWGSL(32), this.bglReduceRG, 'reduceRG32');
    this.reduceRG2Pipe = compPipe(reduceRGWGSL(2), this.bglReduceRG, 'reduceRG2');
    this.projectPipe = compPipe(PROJECT_WGSL, this.bglProject, 'project');
    this.wallClearPipe = compPipe(WALL_CLEAR_WGSL, this.bglWallClear, 'wallClear');
    this.clearPipe = compPipe(
      /* wgsl */ `
        @group(0) @binding(0) var tOut: texture_storage_2d<r32float, write>;
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid: vec3u) {
          if (gid.x >= ${GRID}u || gid.y >= ${GRID}u) { return; }
          textureStore(tOut, vec2i(gid.xy), vec4f(0.0));
        }`,
      this.bglClear,
      'clear',
    );

    const renderModule = d.createShaderModule({ code: RENDER_WGSL });
    this.modules.push(renderModule);
    this.renderPipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.bglRender] }),
      vertex: { module: renderModule, entryPoint: 'vs' },
      fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });

    // 初始化 uniform,确保平滑/清零等早期 pass 在首次 writeUniforms 之前也能读到合法的 P.res
    const seed = new Float32Array(32);
    seed[10] = GRID;
    seed[11] = GRID;
    seed[14] = WALL_RES;
    seed[30] = 1;
    d.queue.writeBuffer(this.uniform, 0, seed);
  }

  /** 显式等待全部着色器模块编译完成,失败时抛出带模块名的具体错误 */
  private modules: GPUShaderModule[] = [];
  private moduleLabels = new Map<GPUShaderModule, string>();

  async ready(): Promise<void> {
    const results = await Promise.all(
      this.modules.map(async (m) => ({ m, info: await m.getCompilationInfo() })),
    );
    const failures: string[] = [];
    for (const { m, info } of results) {
      const errs = info.messages.filter((x) => x.type === 'error');
      if (errs.length) {
        failures.push(`[${this.moduleLabels.get(m) || 'unknown'}] ` + errs.map((x) => x.message).join('; '));
      }
    }
    // getCompilationInfo 个别驱动可能不及时回调,加 2 秒超时兜底
    await Promise.race([
      Promise.all(this.modules.map((m) => m.getCompilationInfo())),
      new Promise((r) => setTimeout(r, 2000)),
    ]);
    if (failures.length) throw new Error('WGSL 编译失败:\n' + failures.join('\n'));
  }

  private bg(layout: GPUBindGroupLayout, entries: GPUBindGroupEntry[]): GPUBindGroup {
    return this.device.createBindGroup({ layout, entries });
  }

  /** 写入全部参数 uniform */
  private writeUniforms(p: SimParams, viewMode: number, extra?: { alpha?: number; meanT?: number; inRes?: number }) {
    const radiusRad = ((p.sourceAngularDiamDeg / 2) * Math.PI) / 180;
    const iDir = incidentDirection(p.incidenceDeg);
    const { b1, b2 } = wallBasis(iDir);
    this.wallHalfW = wallHalfSize(p.dWall, 0.08, radiusRad);

    const f = new Float32Array(32);
    f[0] = 0.08;
    f[1] = p.dWall;
    f[2] = p.incidenceDeg;
    f[3] = radiusRad;
    f[4] = p.baseCurve;
    f[5] = this.hScaleNow(p);
    f[6] = this.rmsNorm;
    f[7] = p.grind;
    f[8] = this.wallHalfW;
    f[9] = this.wallHalfW;
    f[10] = GRID;
    f[11] = extra?.inRes ?? GRID;
    f[12] = viewMode;
    f[13] = p.exposure;
    f[14] = WALL_RES;
    f[15] = extra?.alpha ?? 0;
    f[16] = iDir[0]; f[17] = iDir[1]; f[18] = iDir[2]; f[19] = 0;
    f[20] = b1[0]; f[21] = b1[1]; f[22] = b1[2]; f[23] = 0;
    f[24] = b2[0]; f[25] = b2[1]; f[26] = b2[2]; f[27] = 0;
    f[28] = 0; f[29] = 0; f[30] = 1; f[31] = extra?.meanT ?? 0;
    this.device.queue.writeBuffer(this.uniform, 0, f);
  }

  private hScaleNow(p: SimParams): number {
    if (this.rmsNorm <= 0) return 0;
    return (amplitudeReport(this.stats, p.grind).amplitudeUm * 1e-6) / this.rmsNorm;
  }

  /** 上传观众编辑的镜背厚度场(Float32,0..1,512²)并更新厚度统计 */
  uploadPattern(data: Float32Array) {
    this.device.queue.writeTexture(
      { texture: this.thickness },
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength) as unknown as BufferSource,
      { bytesPerRow: GRID * 4, rowsPerImage: GRID },
      { width: GRID, height: GRID },
    );
    let sum = 0;
    let sumSq = 0;
    for (const v of data) {
      sum += v;
      sumSq += v * v;
    }
    const n = data.length;
    const mean = sum / n;
    const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
    this.stats = { mean, std, meanMm: thicknessInMm(mean), stdMm: 1.4 * std };
  }

  /** 三级归约:512(r32)→32(rg)→2(rg)→1(rg),回读 (sum, sumSq)。尺寸编译期固化,不写 uniform */
  private async reduce(src: GPUTexture): Promise<{ mean: number; rms: number; count: number }> {
    const d = this.device;
    const stages: { inp: GPUTexture; out: GPUTexture; pipe: GPUComputePipeline; groups: number; rg: boolean }[] = [
      { inp: src, out: this.reduceL0, pipe: this.reduceRPipe, groups: GRID / 16, rg: false },
      { inp: this.reduceL0, out: this.reduceL1, pipe: this.reduceRG32Pipe, groups: 2, rg: true },
      { inp: this.reduceL1, out: this.reduceL2, pipe: this.reduceRG2Pipe, groups: 1, rg: true },
    ];
    // 三级放进同一 encoder,执行顺序确定
    const enc = d.createCommandEncoder();
    for (const s of stages) {
      const pass = enc.beginComputePass();
      pass.setPipeline(s.pipe);
      pass.setBindGroup(
        0,
        this.bg(s.rg ? this.bglReduceRG : this.bglReduceR, [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: s.inp.createView() },
          { binding: 2, resource: s.out.createView() },
        ]),
      );
      pass.dispatchWorkgroups(s.groups, s.groups);
      pass.end();
    }
    // 每次归约用一次性缓冲,避免复用 MAP_READ 缓冲造成的读写竞态
    const rb = d.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer(
      { texture: this.reduceL2, origin: [0, 0, 0] },
      { buffer: rb, bytesPerRow: 256, rowsPerImage: 1 },
      { width: 1, height: 1 },
    );
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
    await rb.mapAsync(GPUMapMode.READ);
    const mapped = rb.getMappedRange();
    const view = new Float32Array(mapped.slice(0, 8));
    rb.unmap();
    rb.destroy();
    const sum = view[0];
    const sumSq = view[1];
    const count = GRID * GRID;
    const mean = sum / count;
    const rms = Math.sqrt(Math.max(0, sumSq / count - mean * mean));
    return { mean, rms, count };
  }

  /** 冷却推算:统计 → 平滑 → 泊松求解 → 高度统计标定。纹样每改一笔调用一次 */
  async runCooling(p: SimParams, iters: number = JACOBI_ITERS): Promise<CoolingResult> {
    const t0 = performance.now();
    const d = this.device;

    // 厚度统计
    const ts = await this.reduce(this.thickness);
    this.stats = {
      mean: ts.mean,
      std: ts.rms,
      meanMm: thicknessInMm(ts.mean),
      stdMm: 1.4 * ts.rms,
    };

    // 两次盒式平滑(应力扩散),thickness → smoothTmp → smooth,避免同纹理读写
    {
      // 两次平滑拆为独立提交并等待完成,规避部分驱动在同一 encoder 内
      // 跨 pass 读写同一纹理时缺失隐式同步的问题
      const pairs: [GPUTexture, GPUTexture][] = [
        [this.thickness, this.smoothTmp],
        [this.smoothTmp, this.smooth],
      ];
      for (const [inp, out] of pairs) {
        const enc = d.createCommandEncoder();
        const cp = enc.beginComputePass();
        cp.setPipeline(this.smoothPipe);
        cp.setBindGroup(
          0,
          this.bg(this.bglSmooth, [
            { binding: 0, resource: { buffer: this.uniform } },
            { binding: 1, resource: inp.createView() },
            { binding: 2, resource: out.createView() },
          ]),
        );
        cp.dispatchWorkgroups(GRID / 8, GRID / 8);
        cp.end();
        d.queue.submit([enc.finish()]);
        await d.queue.onSubmittedWorkDone();
      }
    }

    // 清零两个高度场
    {
      const enc = d.createCommandEncoder();
      for (const t of [this.hA, this.hB]) {
        const cp = enc.beginComputePass();
        cp.setPipeline(this.clearPipe);
        cp.setBindGroup(0, this.bg(this.bglClear, [{ binding: 0, resource: t.createView() }]));
        cp.dispatchWorkgroups(GRID / 8, GRID / 8);
        cp.end();
      }
      d.queue.submit([enc.finish()]);
    }

    // 雅可比迭代:∇²h = -α(t - mean),全部迭代合并到单个 encoder 保证确定的执行顺序
    const alpha = 0.02;
    this.writeUniforms(p, 1, { alpha, meanT: this.stats.mean });
    await d.queue.onSubmittedWorkDone();
    let cur = this.hA;
    let nxt = this.hB;
    {
      const enc = d.createCommandEncoder();
      for (let i = 0; i < iters; i++) {
        const cp = enc.beginComputePass();
        cp.setPipeline(this.jacobiPipe);
        cp.setBindGroup(
          0,
          this.bg(this.bglJacobi, [
            { binding: 0, resource: { buffer: this.uniform } },
            { binding: 1, resource: cur.createView() },
            { binding: 2, resource: nxt.createView() },
            { binding: 3, resource: this.smooth.createView() },
          ]),
        );
        cp.dispatchWorkgroups(GRID / 8, GRID / 8);
        cp.end();
        [cur, nxt] = [nxt, cur];
      }
      // 奇数迭代后结果在 hB,同一 encoder 内拷回 hA
      if (iters % 2 === 1) {
        enc.copyTextureToTexture({ texture: this.hB }, { texture: this.hA }, { width: GRID, height: GRID });
      }
      d.queue.submit([enc.finish()]);
    }
    await d.queue.onSubmittedWorkDone();

    // 高度 RMS 归约(读 hA)
    const hr = await this.reduce(this.hA);
    this.rmsNorm = Math.max(hr.rms, 1e-9);

    const report = amplitudeReport(this.stats, p.grind);
    return {
      stats: this.stats,
      rmsNorm: this.rmsNorm,
      hScale: this.hScaleNow(p),
      amplitudeUm: report.amplitudeUm,
      pvUm: report.pvEstimateUm,
      rawUm: report.rawUm,
      clamped: report.clamped,
      totalMs: performance.now() - t0,
    };
  }

  /** 反光成像:清背景 → 逐墙像素按微起伏曲率生成明暗。只依赖光学参数,改距离/角度/光源无需重算冷却 */
  runProjection(p: SimParams): void {
    const d = this.device;
    this.writeUniforms(p, 2);
    const enc = d.createCommandEncoder();

    // 1. 清墙为背景色
    {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.wallClearPipe);
      cp.setBindGroup(0, this.bg(this.bglWallClear, [{ binding: 0, resource: this.wall.createView() }]));
      cp.dispatchWorkgroups(WALL_RES / 8, WALL_RES / 8);
      cp.end();
    }

    // 2. 直接由高度场曲率生成墙面光斑
    {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.projectPipe);
      cp.setBindGroup(
        0,
        this.bg(this.bglProject, [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: this.hA.createView() },
          { binding: 2, resource: this.wall.createView() },
        ]),
      );
      cp.dispatchWorkgroups(WALL_RES / 8, WALL_RES / 8);
      cp.end();
    }
    d.queue.submit([enc.finish()]);
  }

  /** 渲染某一视图到 canvas:mode 0 厚度 / 1 起伏明暗 / 2 墙面 */
  render(canvas: HTMLCanvasElement, mode: 0 | 1 | 2, p: SimParams) {
    const d = this.device;
    const ctx = canvas.getContext('webgpu') as GPUCanvasContext;
    const cfgKey = `${canvas.width}x${canvas.height}`;
    const prevKey = this.canvasCfgKeys.get(canvas);
    if (prevKey !== cfgKey) {
      ctx.configure({ device: d, format: this.format, alphaMode: 'opaque' });
      this.canvasCfgKeys.set(canvas, cfgKey);
    }
    this.writeUniforms(p, mode);
    const scalarTex = mode === 0 ? this.thickness : this.hA;
    const bgroup = this.bg(this.bglRender, [
      { binding: 0, resource: { buffer: this.uniform } },
      { binding: 1, resource: scalarTex.createView() },
      { binding: 2, resource: this.wall.createView() },
      { binding: 3, resource: this.sampler },
    ]);
    const enc = d.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [
        {
          view: ctx.getCurrentTexture().createView(),
          clearValue: { r: 0.04, g: 0.04, b: 0.05, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    rp.setPipeline(this.renderPipe);
    rp.setBindGroup(0, bgroup);
    rp.draw(3);
    rp.end();
    d.queue.submit([enc.finish()]);
  }

  get wallHalfWidth() {
    return this.wallHalfW;
  }
}
