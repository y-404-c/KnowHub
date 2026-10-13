import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Nav from "@/components/Nav";

// next/font 会在构建期把 Inter 下载并自托管到 /_next/static/fonts,
// 运行时不再请求 Google Fonts(国内网络也稳定), 且自动做 font-display: swap.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "知枢 KnowHub - 企业知识库",
  description: "Multi-Agent + MCP + Hybrid Search",
};

// 在首帧绘制前同步写入 data-theme, 避免刷新时深色/浅色闪一下.
// 用 inline script 而不是 useEffect, 因为 useEffect 在 hydration 之后才跑, 会闪.
const themeInitScript = `
(function () {
  try {
    var t = localStorage.getItem("unikb-theme");
    if (t !== "light" && t !== "dark") t = "dark";
    document.documentElement.setAttribute("data-theme", t);
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "dark");
  }
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" className={inter.variable} data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body>
        <Nav />
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
