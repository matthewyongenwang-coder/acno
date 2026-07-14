/** @type {import('next').NextConfig} */
const nextConfig = {
  // onnxruntime-web is loaded in the browser only; keep it out of the server bundle
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals = [...(config.externals ?? []), "onnxruntime-web"];
    }
    return config;
  },
};

export default nextConfig;
