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
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(process.env.SHOT_URL || 'http://localhost:5199/', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction('document.querySelector("#stat-amplitude")?.textContent?.includes("μm")', { timeout: 60000 });
  await new Promise((r) => setTimeout(r, 1500));
  // 逐标签截图
  const tabs = ['pattern', 'relief', 'wall', 'theory'];
  for (const t of tabs) {
    await page.click(`.tab-btn[data-tab="${t}"]`);
    await new Promise((r) => setTimeout(r, 600));
    await page.screenshot({ path: `shots/${t}.png` });
    console.log(`saved shots/${t}.png`);
  }
} finally {
  await browser.close();
}
