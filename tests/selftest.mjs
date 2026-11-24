// headless 自测:用本机 Chrome + SwiftShader 软件 WebGPU 跑完整计算管线
import puppeteer from 'puppeteer-core';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const URL = process.env.SELFTEST_URL || 'http://localhost:5199/selftest.html';

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu-blocklist'],
});
try {
  const page = await browser.newPage();
  page.on('console', (m) => console.log('[browser]', m.text()));
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForFunction('window.__selftest !== undefined', { timeout: 120000 });
  const text = await page.$eval('#out', (el) => el.textContent);
  console.log(text);
  const ok = await page.evaluate(() => window.__selftest);
  process.exit(ok ? 0 : 1);
} finally {
  await browser.close();
}
