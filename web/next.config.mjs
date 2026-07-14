import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Pin the workspace root to this folder. Without it, Next.js can pick up a
  // stray lockfile in a parent directory and trace unrelated files into the
  // serverless bundle. Vercel deploys with this folder as the root anyway.
  outputFileTracingRoot: dirname(fileURLToPath(import.meta.url)),
  // onnxruntime-web is loaded in the browser only; keep it out of the server bundle
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals = [...(config.externals ?? []), "onnxruntime-web"];
    }
    return config;
  },
};

export default nextConfig;
