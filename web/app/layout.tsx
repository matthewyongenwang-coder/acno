import type { Metadata, Viewport } from "next";
import Link from "next/link";
import { Fraunces, Inter } from "next/font/google";

import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-serif",
  weight: ["500", "600"],
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "Acno",
  description:
    "Scan smarter. Skin clearer. Understand your skin type and acne, right in your browser. Your photo never leaves your device.",
};

export const viewport: Viewport = {
  themeColor: "#faf8f4",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${fraunces.variable} ${inter.variable}`}>
      <body>
        <header className="site-header">
          <Link href="/" className="site-mark">
            Acno
          </Link>
          <nav className="site-nav">
            <Link href="/">About</Link>
            <Link href="/scan">Scan</Link>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
