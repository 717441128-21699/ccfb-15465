import './styles.css';
import { MirrorEngine, SimParams, CoolingResult, GRID } from './gpu/engine';
import {
  PatternEditor,
  PATTERN_SIZE,
  PRESETS,
  PresetId,
  drawPreset,
  readThickness,
  Tool,
} from './pattern';
import { ALLOY_SHRINKAGE, MIRROR_DIAMETER_M, STRUCTURE_COEF, grindFactor } from './physics';


const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const DEFAULT_PARAMS: SimParams = {
  dWall: 2,
  incidenceDeg: 0,
  sourceAngularDiamDeg: 0.9,
  baseCurve: 0.9,
  grind: 0.5,
  exposure: 1.0,
};

class App {
  private engine!: MirrorEngine;
  private editor!: PatternEditor;
  private params: SimParams = { ...DEFAULT_PARAMS };
  private lastCooling: CoolingResult | null = null;
  private activeTab = 'pattern';
  private rerunTimer: number | null = null;

  async start() {
    const err = $('webgpu-error');
    try {
      this.engine = await MirrorEngine.create();
      err.hidden = true;
    } catch (e) {
      err.hidden = false;
      err.querySelector('p')!.textContent = (e as Error).message;
      return;
    }

    const patternCanvas = $('pattern-canvas') as HTMLCanvasElement;
    patternCanvas.width = PATTERN_SIZE;
    patternCanvas.height = PATTERN_SIZE;
    drawPreset(patternCanvas, 'lianhua');
    this.editor = new PatternEditor(patternCanvas);
    this.editor.onChange = () => this.scheduleRerun();

    this.wireControls();
    this.wireTabs();
    this.wirePresets();
    this.wireTools();

    // 隐藏标签里 clientWidth 为 0,统一用固定逻辑尺寸,避免画布退化
    for (const c of ['thickness-canvas', 'relief-canvas', 'wall-canvas']) {
      const cv = $(c) as HTMLCanvasElement;
      const cssSize = cv.clientWidth && cv.clientWidth > 0 ? cv.clientWidth : 460;
      const size = Math.round(Math.min(460, cssSize) * Math.min(window.devicePixelRatio, 1.5));
      cv.width = size;
      cv.height = size;
    }

    await this.rerunCooling();
    this.refreshAll();
  }

  private wireControls() {
    const bind = (id: string, key: keyof SimParams, fmt: (v: number) => string, needsCooling = false) => {
      const input = $(id) as HTMLInputElement;
      const label = $(id + '-val');
      const apply = () => {
        const v = parseFloat(input.value);
        this.params[key] = v as never;
        label.textContent = fmt(v);
        if (needsCooling) this.scheduleRerun();
        else this.projectionOnly();
      };
      input.addEventListener('input', apply);
    };
    bind('ctl-distance', 'dWall', (v) => `${v.toFixed(2)} m`);
    bind('ctl-angle', 'incidenceDeg', (v) => `${v.toFixed(0)}°`);
    bind('ctl-source', 'sourceAngularDiamDeg', (v) => `${v.toFixed(2)}°`);
    bind('ctl-curve', 'baseCurve', (v) => `R = ${(1 / v).toFixed(2)} m`);
    bind('ctl-grind', 'grind', (v) => `${Math.round(v * 100)}%`, true);
    bind('ctl-exposure', 'exposure', (v) => `${v.toFixed(2)}×`);

    $('btn-reset')!.addEventListener('click', () => {
      this.params = { ...DEFAULT_PARAMS };
      this.syncControlInputs();
      this.rerunCooling().then(() => this.refreshAll());
    });
  }

  private syncControlInputs() {
    const set = (id: string, v: number) => (($(id) as HTMLInputElement).value = String(v));
    set('ctl-distance', this.params.dWall);
    set('ctl-angle', this.params.incidenceDeg);
    set('ctl-source', this.params.sourceAngularDiamDeg);
    set('ctl-curve', this.params.baseCurve);
    set('ctl-grind', this.params.grind);
    set('ctl-exposure', this.params.exposure);
    ($('ctl-distance-val')!).textContent = `${this.params.dWall.toFixed(2)} m`;
    ($('ctl-angle-val')!).textContent = `${this.params.incidenceDeg.toFixed(0)}°`;
    ($('ctl-source-val')!).textContent = `${this.params.sourceAngularDiamDeg.toFixed(2)}°`;
    ($('ctl-curve-val')!).textContent = `R = ${(1 / this.params.baseCurve).toFixed(2)} m`;
    ($('ctl-grind-val')!).textContent = `${Math.round(this.params.grind * 100)}%`;
    ($('ctl-exposure-val')!).textContent = `${this.params.exposure.toFixed(2)}×`;
  }

  private wireTabs() {
    document.querySelectorAll<HTMLElement>('.tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.activeTab = btn.dataset.tab!;
        document.querySelectorAll<HTMLElement>('.tab-btn').forEach((b) => b.classList.toggle('active', b === btn));
        document.querySelectorAll<HTMLElement>('.tab-panel').forEach((p) => {
          p.classList.toggle('active', p.id === `tab-${this.activeTab}`);
        });
        this.refreshAll();
      });
    });
  }

  private wirePresets() {
    const box = $('preset-buttons')!;
    box.innerHTML = '';
    for (const pr of PRESETS) {
      const b = document.createElement('button');
      b.className = 'preset-btn';
      b.innerHTML = `<strong>${pr.name}</strong><span>${pr.dynasty}</span>`;
      b.title = pr.desc;
      b.addEventListener('click', () => {
        drawPreset($('pattern-canvas') as HTMLCanvasElement, pr.id as PresetId);
        box.querySelectorAll('.preset-btn').forEach((x) => x.classList.remove('active'));
        b.classList.add('active');
        this.rerunCooling().then(() => this.refreshAll());
      });
      box.appendChild(b);
    }
    (box.firstElementChild as HTMLElement)?.classList.add('active');
  }

  private wireTools() {
    document.querySelectorAll<HTMLButtonElement>('.tool-btn').forEach((b) => {
      b.addEventListener('click', () => {
        this.editor.tool = b.dataset.tool as Tool;
        document.querySelectorAll('.tool-btn').forEach((x) => x.classList.toggle('active', x === b));
      });
    });
    const brush = $('ctl-brush') as HTMLInputElement;
    brush.addEventListener('input', () => (this.editor.radius = parseInt(brush.value, 10)));
    const strength = $('ctl-strength') as HTMLInputElement;
    strength.addEventListener('input', () => (this.editor.strength = parseInt(strength.value, 10) / 100));
    $('btn-clear')!.addEventListener('click', () => this.editor.clear());
  }

  /** 一笔画完后 120ms 内合并重算,保证交互流畅且总延迟 < 1s */
  private scheduleRerun() {
    if (this.rerunTimer !== null) window.clearTimeout(this.rerunTimer);
    this.rerunTimer = window.setTimeout(() => {
      this.rerunCooling().then(() => this.refreshAll());
    }, 120);
  }

  private async rerunCooling() {
    const data = readThickness($('pattern-canvas') as HTMLCanvasElement);
    this.engine.uploadPattern(data);
    const t = performance.now();
    this.lastCooling = await this.engine.runCooling(this.params);
    const elapsed = performance.now() - t;
    this.engine.runProjection(this.params);
    this.updateStats(elapsed);
  }

  private projectionOnly() {
    this.engine.runProjection(this.params);
    this.refreshAll();
  }

  private refreshAll() {
    const p = this.params;
    this.engine.render($('thickness-canvas') as HTMLCanvasElement, 0, p);
    this.engine.render($('relief-canvas') as HTMLCanvasElement, 1, p);
    this.engine.render($('wall-canvas') as HTMLCanvasElement, 2, p);
    this.updateScaleBar();
  }

  private updateStats(elapsed?: number) {
    const c = this.lastCooling;
    if (!c) return;
    $('stat-amplitude')!.textContent = `${c.amplitudeUm.toFixed(3)} μm`;
    $('stat-pv')!.textContent = `≈ ${c.pvUm.toFixed(2)} μm`;
    $('stat-thickness')!.textContent = `${c.stats.meanMm.toFixed(2)} ± ${c.stats.stdMm.toFixed(3)} mm`;
    $('stat-rmsnorm')!.textContent = c.rmsNorm.toFixed(4);
    if (elapsed !== undefined) {
      $('stat-time')!.textContent = `${elapsed.toFixed(0)} ms`;
      $('stat-time')!.classList.toggle('good', elapsed < 1000);
    }
    const clamp = $('stat-clamp')!;
    clamp.hidden = !c.clamped;
    clamp.textContent = c.rawUm < 0.1
      ? '幅度低于检测下限,已按 0.1 μm 兜底'
      : '幅度超过 1 μm,已按文物实测上限截断';
    $('stat-grid')!.textContent = `${GRID}×${GRID}`;
    $('stat-shrink')!.textContent = `${(ALLOY_SHRINKAGE * 100).toFixed(1)}%`;
    $('stat-k')!.textContent = STRUCTURE_COEF.toFixed(2);
    $('stat-grindfactor')!.textContent = grindFactor(this.params.grind).toFixed(2);
    $('stat-diameter')!.textContent = `${MIRROR_DIAMETER_M * 1000} mm`;
  }

  private updateScaleBar() {
    const halfW = this.engine.wallHalfWidth;
    // 取一个整齐的物理长度作比例尺
    const target = halfW * 0.5;
    const steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];
    const len = steps.find((s) => s >= target) ?? 1;
    const bar = $('scale-bar')!;
    const frac = len / (2 * halfW);
    bar.style.width = `${frac * 100}%`;
    $('scale-label')!.textContent = len >= 1 ? `${len} m` : `${len * 100} cm`;
  }
}

new App().start().catch((e) => {
  console.error('[App.start fatal]', e?.stack || e);
  const banner = document.getElementById('webgpu-error');
  if (banner) {
    banner.hidden = false;
    banner.querySelector('p')!.textContent = String(e?.message || e);
  }
});
