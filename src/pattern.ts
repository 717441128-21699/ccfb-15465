// 镜背纹样:四种汉代铜镜纹饰的程序化绘制 + 观众自由编辑
// 输出 Float32Array(512²,0..1,数值=厚度),越厚的部位冷却越慢、镜面曲率扰动越弱

export const PATTERN_SIZE = 512;

type Ctx = CanvasRenderingContext2D;

function makeCtx(canvas: HTMLCanvasElement): Ctx {
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, PATTERN_SIZE, PATTERN_SIZE);
  return ctx;
}

function ring(ctx: Ctx, r: number, w: number, shade: number) {
  ctx.beginPath();
  ctx.arc(256, 256, r, 0, Math.PI * 2);
  ctx.strokeStyle = `rgb(${shade},${shade},${shade})`;
  ctx.lineWidth = w;
  ctx.stroke();
}

function filledCircle(ctx: Ctx, r: number, shade: number) {
  ctx.beginPath();
  ctx.arc(256, 256, r, 0, Math.PI * 2);
  ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
  ctx.fill();
}

function polar(cx: number, cy: number, r: number, a: number): [number, number] {
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}

function strokePath(ctx: Ctx, pts: [number, number][], shade: number, w: number, close = false) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  if (close) ctx.closePath();
  ctx.strokeStyle = `rgb(${shade},${shade},${shade})`;
  ctx.lineWidth = w;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.stroke();
}

function fillBlob(ctx: Ctx, pts: [number, number][], shade: number) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
  ctx.fill();
}

// 基础:镜缘(厚)+ 镜钮区
function baseDisc(ctx: Ctx, rimShade = 235) {
  filledCircle(ctx, 252, 26); // 镜体薄底
  ring(ctx, 244, 18, rimShade); // 厚镜缘
  ring(ctx, 228, 4, 150);
  filledCircle(ctx, 34, 205); // 钮座
  ring(ctx, 48, 5, 170);
}

// ---------- 1. 连弧纹镜(西汉早期典型,内向连弧 + 铭文带) ----------
function drawLianhu(ctx: Ctx) {
  baseDisc(ctx);
  // 内向十六连弧
  const arcs = 16;
  const rOut = 214;
  ctx.strokeStyle = 'rgb(200,200,200)';
  ctx.lineWidth = 6;
  for (let i = 0; i < arcs; i++) {
    const a0 = (i / arcs) * Math.PI * 2;
    const a1 = ((i + 1) / arcs) * Math.PI * 2;
    const am = (a0 + a1) / 2;
    const [x0, y0] = polar(256, 256, rOut, a0);
    const [x1, y1] = polar(256, 256, rOut, a1);
    const [xc, yc] = polar(256, 256, rOut - 46, am);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.quadraticCurveTo(xc, yc, x1, y1);
    ctx.stroke();
  }
  ring(ctx, 150, 3, 120);
  // 铭文带(篆隶风格吉语)
  const chars = '见日之光天下大明';
  ctx.fillStyle = 'rgb(225,225,225)';
  ctx.font = 'bold 26px "STKaiti","KaiTi",serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < chars.length; i++) {
    const a = (i / chars.length) * Math.PI * 2 - Math.PI / 2;
    const [x, y] = polar(256, 256, 184, a);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(a + Math.PI / 2);
    ctx.fillText(chars[i], 0, 0);
    ctx.restore();
  }
  // 钮座四叶
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2;
    const [x, y] = polar(256, 256, 40, a);
    ctx.beginPath();
    ctx.ellipse(x, y, 16, 7, a, 0, Math.PI * 2);
    ctx.fillStyle = 'rgb(190,190,190)';
    ctx.fill();
  }
}

// ---------- 2. 草叶纹镜(西汉早中期,中轴对称草叶 + 方钮座) ----------
function drawCaoye(ctx: Ctx) {
  baseDisc(ctx, 225);
  // 方钮座
  ctx.strokeStyle = 'rgb(205,205,205)';
  ctx.lineWidth = 6;
  ctx.strokeRect(256 - 58, 256 - 58, 116, 116);
  ring(ctx, 86, 4, 160);

  const leaf = (cx: number, cy: number, angle: number, len: number, shade: number) => {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(len * 0.45, -len * 0.28, len, 0);
    ctx.quadraticCurveTo(len * 0.45, len * 0.28, 0, 0);
    ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
    ctx.fill();
    // 叶脉
    ctx.beginPath();
    ctx.moveTo(4, 0);
    ctx.lineTo(len - 4, 0);
    ctx.strokeStyle = `rgb(${Math.max(40, shade - 70)},${Math.max(40, shade - 70)},${Math.max(40, shade - 70)})`;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  };

  // 四向草叶枝
  for (let k = 0; k < 4; k++) {
    const base = (k / 4) * Math.PI * 2;
    const [bx, by] = polar(256, 256, 92, base);
    const [tx, ty] = polar(256, 256, 205, base);
    strokePath(ctx, [[bx, by], [tx, ty]], 185, 5);
    for (let j = 1; j <= 3; j++) {
      const rr = 100 + j * 30;
      const [px, py] = polar(256, 256, rr, base);
      leaf(px, py, base + 0.9, 30, 175);
      leaf(px, py, base - 0.9, 30, 175);
    }
    // 顶端花苞
    const [fx, fy] = polar(256, 256, 214, base);
    filledCircleAt(ctx, fx, fy, 9, 210);
  }
  // 四角乳钉
  for (let k = 0; k < 4; k++) {
    const a = (k / 4) * Math.PI * 2 + Math.PI / 4;
    const [x, y] = polar(256, 256, 128, a);
    filledCircleAt(ctx, x, y, 7, 220);
  }
}

function filledCircleAt(ctx: Ctx, x: number, y: number, r: number, shade: number) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
  ctx.fill();
}

// ---------- 3. 星云纹镜(西汉中期,多乳钉 + 曲线云气) ----------
function drawXingyun(ctx: Ctx) {
  baseDisc(ctx, 230);
  ring(ctx, 196, 5, 170);
  ring(ctx, 70, 4, 150);

  // 云气:每组一条 S 形卷云
  const groups = 6;
  for (let g = 0; g < groups; g++) {
    const base = (g / groups) * Math.PI * 2;
    const pts: [number, number][] = [];
    for (let s = -2; s <= 2; s++) {
      const rr = 130 + s * 22;
      const aa = base + s * 0.22;
      pts.push(polar(256, 256, rr, aa));
    }
    strokePath(ctx, pts, 165, 4);
    // 卷云头
    const [hx, hy] = polar(256, 256, 175, base - 0.45);
    ctx.beginPath();
    ctx.arc(hx, hy, 12, 0, Math.PI * 1.5);
    ctx.strokeStyle = 'rgb(175,175,175)';
    ctx.lineWidth = 4;
    ctx.stroke();
  }

  // 乳钉阵列(大小两层)
  for (let g = 0; g < groups; g++) {
    const a = (g / groups) * Math.PI * 2 + Math.PI / groups;
    for (const rr of [105, 160]) {
      const [x, y] = polar(256, 256, rr, a);
      filledCircleAt(ctx, x, y, 10, 235);
      filledCircleAt(ctx, x - 3, y - 3, 3, 255);
    }
  }
  for (let g = 0; g < 8; g++) {
    const a = (g / 8) * Math.PI * 2;
    const [x, y] = polar(256, 256, 210, a);
    filledCircleAt(ctx, x, y, 6, 200);
  }
}

// ---------- 4. 蟠螭纹镜(战国末-西汉初,交缠兽形纹) ----------
function drawPanchi(ctx: Ctx) {
  baseDisc(ctx, 240);
  ring(ctx, 208, 4, 160);
  ring(ctx, 66, 4, 150);

  const beasts = 4;
  for (let g = 0; g < beasts; g++) {
    const base = (g / beasts) * Math.PI * 2;
    // 兽身:交缠的双弧
    const body: [number, number][] = [];
    for (let s = 0; s <= 24; s++) {
      const t = s / 24;
      const rr = 95 + 85 * t;
      const aa = base + t * 1.7 + 0.35 * Math.sin(t * Math.PI * 2);
      body.push(polar(256, 256, rr, aa));
    }
    strokePath(ctx, body, 195, 7);
    // 兽身第二道交缠线
    const body2: [number, number][] = [];
    for (let s = 0; s <= 24; s++) {
      const t = s / 24;
      const rr = 100 + 80 * t;
      const aa = base + 0.5 + t * 1.5 - 0.3 * Math.sin(t * Math.PI * 2);
      body2.push(polar(256, 256, rr, aa));
    }
    strokePath(ctx, body2, 150, 4);
    // 兽首(水滴状)
    const [hx, hy] = polar(256, 256, 188, base + 1.75);
    const ha = base + 1.75;
    const head: [number, number][] = [];
    for (let s = 0; s <= 10; s++) {
      const phi = (s / 10) * Math.PI * 2;
      head.push([hx + 16 * Math.cos(phi) * Math.cos(ha) - 9 * Math.sin(phi) * Math.sin(ha),
      hy + 16 * Math.cos(phi) * Math.sin(ha) + 9 * Math.sin(phi) * Math.cos(ha)]);
    }
    fillBlob(ctx, head, 205);
    // 爪
    for (const t of [0.35, 0.7]) {
      const rr = 95 + 85 * t;
      const aa = base + t * 1.7 + 0.35 * Math.sin(t * Math.PI * 2);
      const [px, py] = polar(256, 256, rr, aa);
      strokePath(ctx, [[px, py], [px + 10 * Math.cos(aa + 1.2), py + 10 * Math.sin(aa + 1.2)]], 180, 3);
      strokePath(ctx, [[px, py], [px + 10 * Math.cos(aa - 1.2), py + 10 * Math.sin(aa - 1.2)]], 180, 3);
    }
  }
}

export type PresetId = 'lianhua' | 'caoye' | 'xingyun' | 'panchi';

export const PRESETS: { id: PresetId; name: string; dynasty: string; desc: string }[] = [
  { id: 'lianhua', name: '连弧铭文镜', dynasty: '西汉早期', desc: '内向十六连弧,「见日之光,天下大明」铭文带' },
  { id: 'caoye', name: '草叶纹镜', dynasty: '西汉早中期', desc: '方钮座,四向对称草叶枝与乳钉' },
  { id: 'xingyun', name: '星云纹镜', dynasty: '西汉中期', desc: '多枚乳钉连缀如星,衬卷云气纹' },
  { id: 'panchi', name: '蟠螭纹镜', dynasty: '战国末—西汉初', desc: '四兽交缠的蟠螭,线条繁密' },
];

const DRAWERS: Record<PresetId, (ctx: Ctx) => void> = {
  lianhua: drawLianhu,
  caoye: drawCaoye,
  xingyun: drawXingyun,
  panchi: drawPanchi,
};

export function drawPreset(canvas: HTMLCanvasElement, id: PresetId) {
  const ctx = makeCtx(canvas);
  DRAWERS[id](ctx);
}

/** 从编辑画布读出厚度场(Float32,0..1) */
export function readThickness(canvas: HTMLCanvasElement): Float32Array {
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  const img = ctx.getImageData(0, 0, PATTERN_SIZE, PATTERN_SIZE);
  const out = new Float32Array(PATTERN_SIZE * PATTERN_SIZE);
  for (let i = 0; i < out.length; i++) {
    out[i] = img.data[i * 4] / 255;
  }
  return out;
}

// ---------- 观众编辑:软笔刷的增厚 / 削薄 / 橡皮 ----------
export type Tool = 'raise' | 'lower' | 'erase';

export class PatternEditor {
  private ctx: Ctx;
  private drawing = false;
  private last: [number, number] | null = null;
  tool: Tool = 'raise';
  radius = 18;
  strength = 0.5;
  onChange: () => void = () => {};

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    canvas.addEventListener('pointerdown', (e) => {
      this.drawing = true;
      canvas.setPointerCapture(e.pointerId);
      this.last = this.toLocal(e);
      this.stamp(this.last);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.drawing) return;
      const p = this.toLocal(e);
      this.stroke(this.last!, p);
      this.last = p;
    });
    const stop = () => {
      if (!this.drawing) return;
      this.drawing = false;
      this.last = null;
      this.onChange();
    };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
  }

  private toLocal(e: PointerEvent): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [
      ((e.clientX - r.left) / r.width) * PATTERN_SIZE,
      ((e.clientY - r.top) / r.height) * PATTERN_SIZE,
    ];
  }

  private stamp(p: [number, number]) {
    const ctx = this.ctx;
    const g = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], this.radius);
    if (this.tool === 'erase') {
      ctx.globalCompositeOperation = 'source-over';
      g.addColorStop(0, 'rgba(26,26,26,1)');
      g.addColorStop(1, 'rgba(26,26,26,0)');
    } else {
      // 用 lighter / darker 在灰度上加减厚度
      ctx.globalCompositeOperation = this.tool === 'raise' ? 'lighter' : 'multiply';
      const v = Math.round(this.strength * 90);
      const col = this.tool === 'raise' ? v : Math.round(255 - this.strength * 110);
      g.addColorStop(0, `rgba(${col},${col},${col},1)`);
      g.addColorStop(1, `rgba(${col},${col},${col},0)`);
    }
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(p[0], p[1], this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
  }

  private stroke(a: [number, number], b: [number, number]) {
    const dist = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const steps = Math.max(1, Math.ceil(dist / (this.radius * 0.4)));
    for (let i = 1; i <= steps; i++) {
      this.stamp([a[0] + ((b[0] - a[0]) * i) / steps, a[1] + ((b[1] - a[1]) * i) / steps]);
    }
  }

  clear() {
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, PATTERN_SIZE, PATTERN_SIZE);
    this.onChange();
  }
}
