import type { NextConfig } from "next";
const backend =
  process.env.NEXT_PUBLIC_API_BASE ??
  process.env.BACKEND_INTERNAL_URL ??
  "http://127.0.0.1:8000";


const nextConfig: NextConfig = {
  allowedDevOrigins: ["141.54.132.236"],
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${backend}/:path*`,
      },
    ];
  },
};

export default nextConfig;