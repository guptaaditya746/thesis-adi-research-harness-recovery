import type { NextConfig } from "next";
const backend =
  process.env.BACKEND_INTERNAL_URL ??
  process.env.NEXT_PUBLIC_API_BASE ??
  "http://127.0.0.1:8000";


const nextConfig: NextConfig = {
  allowedDevOrigins: ["localhost", "127.0.0.1", "141.54.132.236", "141.54.132.137"],
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
