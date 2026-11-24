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
  await new Promise((r) => setTimeout(r, 500));

  async function setSlider(id, value) {
    await page.$eval(`#${id}`, (el, val) => {
      const input = el;
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(input, String(val));
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    }, value);
    await new Promise((r) => setTimeout(r, 600));
  }

  // 近距离(模糊)
  await setSlider('ctl-distance', 0.5);
  await page.screenshot({ path: 'shots/param-near.png' });
  // 远距离(清晰但发散)
  await setSlider('ctl-distance', 5);
  await page.screenshot({ path: 'shots/param-far.png' });
  // 恢复距离,大光源(柔和)
  await setSlider('ctl-distance', 2);
  await setSlider('ctl-source', 3);
  await page.screenshot({ path: 'shots/param-bigsource.png' });
  console.log('parameter screenshots saved');
} finally {
  await browser.close();
}
