import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu-blocklist', '--window-size=1440,1000'],
  defaultViewport: { width: 1440, height: 1000, deviceScaleFactor: 1 },
});
try {
  const page = await browser.newPage();
  await page.setCacheEnabled(false);
  await page.goto('http://localhost:5199/', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction('document.querySelector("#stat-amplitude")?.textContent?.includes("μm")', { timeout: 60000 });

  const before = await page.$eval('#stat-amplitude', (el) => el.textContent);
  // 在圆形镜背画布中心拖动绘制一笔
  const box = await page.$eval('#pattern-canvas', (c) => {
    const r = c.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  });
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  await page.mouse.move(cx - 60, cy);
  await page.mouse.down();
  for (let i = 0; i <= 10; i++) {
    await page.mouse.move(cx - 60 + i * 12, cy + Math.sin(i) * 8);
    await new Promise((r) => setTimeout(r, 30));
  }
  await page.mouse.up();
  // 等待防抖(120ms)+ 重算
  await new Promise((r) => setTimeout(r, 800));
  const after = await page.$eval('#stat-amplitude', (el) => el.textContent);
  const time = await page.$eval('#stat-time', (el) => el.textContent);
  console.log('amplitude before:', before, ' after edit:', after, ' recompute:', time);

  await page.click('.tab-btn[data-tab="wall"]');
  await new Promise((r) => setTimeout(r, 400));
  await page.screenshot({ path: 'shots/edited.png' });
  console.log('edited screenshot saved');
} finally {
  await browser.close();
}
