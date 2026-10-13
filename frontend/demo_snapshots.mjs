// 静态预览图: 截 4 张关键画面, 不依赖完整录屏流程, 快速看到效果
import { chromium } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";

const OUT_DIR = path.resolve(process.cwd(), "demo_output");
fs.mkdirSync(OUT_DIR, { recursive: true });
const FRONTEND = "http://localhost:3000";

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

(async () => {
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--force-color-profile=srgb"],
  });

  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await ctx.newPage();

  console.log("[snap] 打开聊天首页");
  await page.goto(`${FRONTEND}/chat`, { waitUntil: "networkidle" });
  await sleep(2500);
  await page.screenshot({ path: path.join(OUT_DIR, "01_chat_hero_dark.png") });

  console.log("[snap] 切浅色主题");
  await page.locator(".theme-toggle").click();
  await sleep(1500);
  await page.screenshot({ path: path.join(OUT_DIR, "02_chat_hero_light.png") });

  // 切回深色
  await page.locator(".theme-toggle").click();
  await sleep(800);

  // 触发一次问答, 等回答出来再截图
  console.log("[snap] 提问并等待回答");
  const sugg = page.locator(".sugg", { hasText: "支持哪些 LLM" }).first();
  if (await sugg.count()) {
    await sugg.click();
    await sleep(600);
  } else {
    await page.fill("textarea", "KnowHub 支持哪些 LLM?");
  }
  await page.keyboard.press("Control+Enter");
  // 等待流式回答
  for (let i = 0; i < 40; i++) {
    await sleep(1500);
    const loading = await page.locator(".spinner, .typing").count();
    const streaming = await page.locator(".stream-cursor").count();
    if (loading === 0 && streaming === 0) break;
  }
  await sleep(1500);
  console.log("[snap] 截 markdown 渲染后的回答");
  await page.screenshot({ path: path.join(OUT_DIR, "03_chat_answer_md.png"), fullPage: false });

  // 悬停消息显示操作按钮
  await page.locator(".bubble.assistant").last().hover();
  await sleep(800);
  await page.screenshot({ path: path.join(OUT_DIR, "04_msg_actions.png") });

  // 展开引用
  const sources = page.locator("details.sources summary").first();
  if (await sources.count()) {
    await sources.click();
    await sleep(800);
    await page.screenshot({ path: path.join(OUT_DIR, "05_sources_expanded.png") });
  }

  // Upload 页
  console.log("[snap] Upload 页");
  await page.click("a.tab[href='/upload']");
  await page.waitForLoadState("networkidle");
  await sleep(1500);
  await page.screenshot({ path: path.join(OUT_DIR, "06_upload.png") });

  await browser.close();
  console.log("[snap] 完成, 输出到", OUT_DIR);
})();
