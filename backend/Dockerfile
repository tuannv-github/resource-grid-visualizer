# Multi-stage: cuda-build produces resgrid_plotter + libcudart; default image is the final stage (runtime).
#
# In runtime: Python/backend layers first; COPY from cuda-build last so backend/plotter/ edits mostly invalidate final layers only.
#
# Runtime base is architecture-specific: jrottenberg/ffmpeg:7.1-nvidia2404 is linux/amd64 only. On arm64, Ubuntu’s
# distro ffmpeg typically lacks a working h264_nvenc session (OpenEncodeSessionEx / “unsupported device”). We install
# Jellyfin’s aarch64 ffmpeg (ffnvcodec-enabled) and symlink it as ffmpeg on PATH. For NGC DeepStream–style full stacks
# see https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream — not used here (large, analytics-focused).

# Must not assign a default here: that overrides BuildKit’s automatic TARGETARCH and breaks arm64 builds
# (they would still pull the amd64-only ffmpeg stage). See https://docs.docker.com/reference/dockerfile/#automatic-platform-args-in-the-global-scope
ARG TARGETARCH

FROM nvidia/cuda:12.6.1-devel-ubuntu24.04 AS cuda-build

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/plotter
COPY backend/plotter/ ./
# Fresh configure even if a stale out/ slipped into the build context (host paths in CMakeCache break nvcc).
RUN rm -rf out

RUN cmake -S . -B out -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build out -j"$(nproc)" \
    && mkdir -p /cuda-artifacts \
    && cp out/resgrid_plotter /cuda-artifacts/ \
    && cp "$(readlink -f /usr/local/cuda/lib64/libcudart.so)" /cuda-artifacts/libcudart.so

FROM jrottenberg/ffmpeg:7.1-nvidia2404 AS runtime-amd64

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

FROM ubuntu:24.04 AS runtime-arm64

ENV DEBIAN_FRONTEND=noninteractive

# Noble arm64 .deb; pinned release for reproducible builds (bump when upgrading FFmpeg).
ARG JELLYFIN_FFMPEG_DEB_URL=https://github.com/jellyfin/jellyfin-ffmpeg/releases/download/v7.1.1-3/jellyfin-ffmpeg7_7.1.1-3-noble_arm64.deb

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl python3 python3-venv \
    && curl -fsSL -o /tmp/jellyfin-ffmpeg.deb "${JELLYFIN_FFMPEG_DEB_URL}" \
    && apt-get install -y --no-install-recommends /tmp/jellyfin-ffmpeg.deb \
    && rm -f /tmp/jellyfin-ffmpeg.deb \
    && ln -sf /usr/lib/jellyfin-ffmpeg/ffmpeg /usr/local/bin/ffmpeg \
    && ln -sf /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-${TARGETARCH} AS runtime

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/scripts ./scripts
COPY backend/app ./app

# CUDA I/Q plotter — last in this stage for better cache when backend/plotter/ changes.
COPY --from=cuda-build /cuda-artifacts/libcudart.so /usr/local/lib/libcudart.so.12
COPY --from=cuda-build /cuda-artifacts/resgrid_plotter /usr/local/bin/resgrid_plotter
RUN ldconfig && chmod +x /usr/local/bin/resgrid_plotter

# Base image sets ENTRYPOINT to ffmpeg; without clearing it, `command` becomes extra ffmpeg args.
ENTRYPOINT []
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]
