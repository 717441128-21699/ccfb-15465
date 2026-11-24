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
  await page.click('.tab-btn[data-tab="wall"]');
  await new Promise((r) => setTimeout(r, 400));
  const btns = await page.$$('.preset-btn');
  for (let i = 0; i < btns.length; i++) {
    // 预设按钮在 pattern 标签内,切过去点击,再回 wall 看投影
    await page.click('.tab-btn[data-tab="pattern"]');
    await new Promise((r) => setTimeout(r, 200));
    const fresh = await page.$$('.preset-btn');
    await fresh[i].click();
    await new Promise((r) => setTimeout(r, 700));
    await page.click('.tab-btn[data-tab="wall"]');
    await new Promise((r) => setTimeout(r, 400));
    await page.screenshot({ path: `shots/preset-${i}.png` });
    console.log('saved preset', i);
  }
} finally {
  await browser.close();
}
