"use client";
import { useEffect, useState } from "react";

type Theme = "dark" | "light";

/**
 * 深浅色切换。
 *
 * 为什么初始值写死 "dark" 而不是直接读 DOM:
 * 服务端渲染时没有 document, 首帧必须给出一个确定值, 否则 SSR 与 CSR 结果不一致会
 * hydration mismatch。真实主题由 layout.tsx 里的 inline script 在首帧前写入
 * data-theme, 这里在 useEffect 里再同步一次到本地 state。
 */
export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    const cur = document.documentElement.getAttribute("data-theme");
    if (cur === "light" || cur === "dark") setTheme(cur);
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("unikb-theme", next);
    } catch {
      /* 隐私模式下 localStorage 可能不可写, 忽略即可 */
    }
    setTheme(next);
  }

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      title={theme === "dark" ? "切换到浅色模式" : "切换到深色模式"}
      aria-label="切换深浅色主题"
    >
      {theme === "dark" ? "☾" : "☀"}
    </button>
  );
}
