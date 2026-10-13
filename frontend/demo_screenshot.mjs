// 录屏脚本: 用 Playwright 驱动 chromium headless 录制 KnowHub 演示流程
// 输出 webm (Playwright 原生支持), 在 chromium 浏览器里可直接播放
import { chromium } from "@playwright/test";
import path from "node:path";

const OUT_DIR = path.resolve(process.cwd(), "demo_output");
import fs from "node:fs";
fs.mkdirSync(OUT_DIR, { recursive: true });

const FRONTEND = "http://localhost:3000";

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

(async () => {
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--force-color-profile=srgb"],
  });

  // Playwright 录屏: 保存到 OUT_DIR, 每次新建 context 都开启 video
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: OUT_DIR, size: { width: 1280, height: 800 } },
  });
  const page = await context.newPage();

  const actions = [];
  function log(msg) {
    console.log(`  [demo] ${msg}`);
    actions.push(`${new Date().toISOString().slice(11, 23)}  ${msg}`);
  }

  try {
    // ====== Scene 1: 打开聊天页 ======
    log("打开 KnowHub 聊天首页");
    await page.goto(`${FRONTEND}/chat`, { waitUntil: "networkidle" });
    await sleep(2500); // 让 hero 动画展开

    // ====== Scene 2: 主题切换演示 ======
    log("切换到浅色主题");
    const toggle = page.locator(".theme-toggle");
    await toggle.click();
    await sleep(2000);
    log("切回深色主题");
    await toggle.click();
    await sleep(1500);

    // ====== Scene 3: 点击快捷问题 ======
    log("点击快捷问题: KnowHub 支持哪些 LLM?");
    const sugg = page.locator(".sugg", { hasText: "支持哪些 LLM" }).first();
    if (await sugg.count()) {
      await sugg.click();
      await sleep(800);
    } else {
      log("未找到快捷按钮, 用键盘输入");
      await page.fill("textarea", "KnowHub 支持哪些 LLM？");
    }

    // ====== Scene 4: 发送并等待流式回答 ======
    log("发送问题, 等待流式回答...");
    await page.keyboard.press("Control+Enter");
    // 等首字 + 流式打字动画
    await sleep(8000);
    // 等回答基本完成(气泡内出现内容且不在 loading)
    for (let i = 0; i < 30; i++) {
      const loading = await page.locator(".spinner, .typing").count();
      const streaming = await page.locator(".stream-cursor").count();
      if (loading === 0 && streaming === 0) break;
      await sleep(1500);
    }
    await sleep(2000);
    log("回答完成");

    // ====== Scene 5: 展开引用 ======
    const sources = page.locator("details.sources summary").first();
    if (await sources.count()) {
      log("展开引用来源");
      await sources.click();
      await sleep(2500);
    }

    // ====== Scene 6: 鼠标悬停展示消息操作 ======
    log("悬停最后一条 assistant 消息(展示复制/重新生成按钮)");
    const lastBubble = page.locator(".bubble.assistant").last();
    if (await lastBubble.count()) {
      await lastBubble.hover();
      await sleep(2000);
    }

    // ====== Scene 7: 切换到 Upload 页 ======
    log("导航到 Upload 页");
    await page.click("a.tab[href='/upload']");
    await page.waitForLoadState("networkidle");
    await sleep(2000);

    // ====== Scene 8: 导航到 Sources 页 ======
    log("导航到 Sources 页");
    await page.click("a.tab[href='/sources']");
    await page.waitForLoadState("networkidle");
    await sleep(2000);

    // ====== Scene 9: 导航到 History 页 ======
    log("导航到 History 页");
    await page.click("a.tab[href='/history']");
    await page.waitForLoadState("networkidle");
    await sleep(2500);

    // 回到 chat 收尾
    log("回到 Chat 页收尾");
    await page.click("a.tab[href='/chat']");
    await sleep(2500);

    log("录屏结束");
  } catch (e) {
    log(`ERROR: ${e.message}`);
    await page.screenshot({ path: path.join(OUT_DIR, "error.png"), fullPage: true });
  } finally {
    // 关闭 context 时 Playwright 才会把 video flush 到磁盘
    const vid = page.video();
    await vid?.saveAs(path.join(OUT_DIR, "KnowHub_demo.webm")).catch(() => {});
    await context.close();
    await browser.close();
    fs.writeFileSync(path.join(OUT_DIR, "timeline.txt"), actions.join("\n"));
    console.log(`\n=> 录屏已保存到: ${OUT_DIR}`);
    try {
      const f = fs.readdirSync(OUT_DIR);
      console.log("   文件列表:", f.join(", "));
    } catch {}
  }
})();
