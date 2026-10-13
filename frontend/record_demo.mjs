// frontend/record_demo.mjs
// 用 Playwright 无头 Chromium 录制 KnowHub 演示视频。
//
// 运行前提: 后端 :8000 与前端 :3000 已启动, 且演示知识库已灌好示例文档。
//   cd backend  && ... uvicorn app.main:app --port 8000
//   cd backend  && python run_demo.py kb-KnowHub-demo      # 灌示例文档
//   cd frontend && NEXT_PUBLIC_DEMO_KB=kb-KnowHub-demo npm run dev
//   cd frontend && node record_demo.mjs
//
// 产物: frontend/demo_output/KnowHub_demo.webm (VP8 1440x900)
// 再用 transcode_demo.mjs 转成 MP4 (H.264) + GIF。
//
// 注意: Playwright 的 page.video().saveAs() 在 Windows 下会挂死, 这里不调用它,
// 而是让 Playwright 自己写随机名 webm, 关闭 context 后用 fs.renameSync 改名。
import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs";

const ROOT = path.resolve(process.cwd());
const FRONTEND = process.env.DEMO_FRONTEND || "http://localhost:3000";
const OUT_DIR = path.join(ROOT, "demo_output");

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function log(actions, msg) {
  actions.push(`${new Date().toISOString().slice(11, 23)}  ${msg}`);
  console.log(`  [demo] ${msg}`);
}

function latestWebm(dir) {
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".webm") && !f.startsWith("_") && f !== "KnowHub_demo.webm")
    .map((f) => ({ f, stat: fs.statSync(path.join(dir, f)) }))
    .sort((a, b) => b.stat.mtimeMs - a.stat.mtimeMs)[0]?.f;
}

/** 直接读 DOM 判断回答状态。流式输出期间不要动 locator —— 节点频繁重渲染会让
 *  Playwright 的自动重试挂住, 所以全程用 page.evaluate。 */
function readBubble() {
  return (() => {
    const bubbles = document.querySelectorAll(".bubble.assistant");
    const bubble = bubbles[bubbles.length - 1];
    if (!bubble) return { text: "", streaming: false, typing: false, count: 0 };
    const md = bubble.querySelector(".md, .content");
    return {
      text: md ? md.textContent || "" : "",
      streaming: !!bubble.querySelector(".stream-cursor"),
      typing: !!bubble.querySelector(".typing"),
      count: bubbles.length,
    };
  })();
}

/** 等新的一条回答开始输出首字。返回是否等到。
 *
 *  `baseline` 必须是**点发送之前**的助手气泡数。前端在点击发送的同一时刻就把
 *  空气泡插进 DOM (只有三点省略号), 如果在发送之后才采集基线, 就会把这条空
 *  气泡算进基线, 于是 "count 变大" 永远不成立, 第二轮提问会一直空等到超时。
 *
 *  判定条件用 "气泡数 +1 且正文非空": 空气泡正文是 "", 首字到达后才成立。
 */
async function waitForAnswer(page, actions, { baseline = 0, timeoutMs = 25000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await sleep(300);
    const st = await page.evaluate(readBubble).catch(() => ({ text: "", count: 0, streaming: false }));
    const text = (st.text || "").trim();
    if (st.count >= baseline + 1 && text.length > 0) {
      log(actions, `回答开始输出 (${text.length} 字)`);
      return true;
    }
  }
  log(actions, `WARN: ${timeoutMs / 1000}s 内未出现回答内容`);
  return false;
}

/** 等流式结束 (光标与打字点都消失)。 */
async function waitStreamEnd(page, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const st = await page.evaluate(readBubble).catch(() => ({ streaming: false, typing: false }));
    if (!st.streaming && !st.typing) return true;
    await sleep(400);
  }
  return false;
}

/** 提问并发送。
 *  @param keyword  匹配快捷问题的关键字
 *  @param fullText 快捷问题消失时手动输入的完整问句。
 *                  注意: 不能只输 keyword —— 第二轮开始时 hero 区(含快捷问题)
 *                  已经收起, 退化成"20 个账号"这种残缺问句会检索不到东西,
 *                  回答变成"未找到相关信息", 演示效果直接废掉。
 */
async function ask(page, actions, keyword, fullText) {
  const sugg = page.locator(".sugg", { hasText: keyword }).first();
  let question = fullText || keyword;
  if ((await sugg.count()) > 0) {
    question = (await sugg.textContent())?.trim() || question;
    await sugg.click();
  } else {
    await page.fill("textarea", question);
  }
  await sleep(300);
  log(actions, `提问: ${question}`);

  // 基线必须在发送之前采集, 原因见 waitForAnswer 的注释。
  const baseline = (await page.evaluate(readBubble).catch(() => ({ count: 0 }))).count || 0;

  await page.locator("textarea").focus();
  await page.click(".composer button");
  log(actions, "已发送，等待流式回答...");
  return { question, baseline };
}

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const actions = [];

  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--force-color-profile=srgb"],
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    recordVideo: { dir: OUT_DIR, size: { width: 1440, height: 900 } },
  });
  const page = await context.newPage();

  let lastErr = "";
  page.on("pageerror", (err) => log(actions, `PAGE-ERROR: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      const t = msg.text();
      if (t === lastErr) return;
      lastErr = t;
      log(actions, `BROWSER-ERROR: ${t}`);
    }
  });

  try {
    // ---- Scene 1: 打开聊天首页 ----
    log(actions, "打开 KnowHub 聊天首页");
    await page.goto(`${FRONTEND}/chat`, { waitUntil: "domcontentloaded" });
    await sleep(1200);

    // ---- Scene 2: 主题切换 (暗 -> 亮 -> 暗) ----
    log(actions, "切换浅色主题");
    await page.locator(".theme-toggle").click();
    await sleep(900);
    log(actions, "切回深色主题");
    await page.locator(".theme-toggle").click();
    await sleep(700);

    // ---- Scene 3: 第一轮提问 (选流式最明显的一题) ----
    const t1 = await ask(page, actions, "混合检索", "混合检索是怎么工作的？");
    await waitForAnswer(page, actions, { baseline: t1.baseline });
    await waitStreamEnd(page);
    await sleep(700);

    // ---- Scene 4: 展开引用来源 ----
    const summary = page.locator("details.sources summary").first();
    if ((await summary.count()) > 0) {
      log(actions, "展开引用来源");
      await summary.click();
      await sleep(1400);
    }

    // ---- Scene 5: 悬停助手消息, 展示复制/重新生成操作栏 ----
    const lastBubble = page.locator(".bubble.assistant").last();
    if ((await lastBubble.count()) > 0) {
      log(actions, "悬停助手消息展示操作栏");
      await lastBubble.hover();
      await sleep(1300);
    }

    // ---- Scene 6: 第二轮提问 (多轮对话 + 计算题) ----
    await page.locator("textarea").scrollIntoViewIfNeeded().catch(() => {});
    const t2 = await ask(page, actions, "20 个账号", "企业版买 20 个账号多少钱？");
    await waitForAnswer(page, actions, { baseline: t2.baseline });
    await waitStreamEnd(page);
    await sleep(1000);

    log(actions, "录屏结束");
  } catch (e) {
    log(actions, `ERROR: ${e.message}`);
    await page.screenshot({ path: path.join(OUT_DIR, "error.png"), fullPage: false }).catch(() => {});
  } finally {
    // 先掐掉页面所有请求 (尤其是还在开的 SSE 流), 否则 context.close() 会挂死。
    try { await page.evaluate(() => window.stop()); } catch {}
    await Promise.race([context.close().catch(() => {}), sleep(5000)]);
    await Promise.race([browser.close().catch(() => {}), sleep(3000)]);

    const latest = latestWebm(OUT_DIR);
    if (latest) {
      try {
        fs.renameSync(path.join(OUT_DIR, latest), path.join(OUT_DIR, "KnowHub_demo.webm"));
        log(actions, "已重命名为 KnowHub_demo.webm");
      } catch {
        log(actions, `WARN: 无法重命名 ${latest}`);
      }
    } else {
      log(actions, "WARN: 未找到 Playwright 生成的 webm");
    }

    fs.writeFileSync(path.join(OUT_DIR, "timeline.txt"), actions.join("\n"));
    console.log(`\n=> 录屏已保存到: ${OUT_DIR}`);
    process.exit(0);
  }
})();
