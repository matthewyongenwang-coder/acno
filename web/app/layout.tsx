import type { Metadata, Viewport } from "next";
import Link from "next/link";
import { Schibsted_Grotesk } from "next/font/google";

import SmoothScroll from "./smooth-scroll";
import "./globals.css";

const grotesk = Schibsted_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "acno",
  description:
    "Scan smarter. Skin clearer. Understand your skin type and acne, right in your browser. Your photo never leaves your device.",
};

export const viewport: Viewport = {
  themeColor: "#12100f",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={grotesk.variable}>
      <body>
        <SmoothScroll />
        <header className="site-header">
          <Link href="/" className="site-mark">
            acno
          </Link>
          <nav className="site-nav">
            <Link href="/">Story</Link>
            <Link href="/scan" className="site-nav-cta">
              Scan
            </Link>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
