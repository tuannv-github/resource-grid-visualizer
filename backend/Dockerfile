# Multi-stage: cuda-build produces resgrid_plotter + libcudart; default image is the final stage (runtime).
#
# In runtime: Python/backend layers first; COPY from cuda-build last so backend/resouce-grid-plotter/ edits mostly invalidate final layers only.

FROM nvidia/cuda:12.6.1-devel-ubuntu24.04 AS cuda-build

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/resouce-grid-plotter
COPY backend/resouce-grid-plotter/ ./
# Fresh configure even if a stale out/ slipped into the build context (host paths in CMakeCache break nvcc).
RUN rm -rf out

RUN cmake -S . -B out -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build out -j"$(nproc)" \
    && mkdir -p /cuda-artifacts \
    && cp out/resgrid_plotter /cuda-artifacts/ \
    && cp "$(readlink -f /usr/local/cuda/lib64/libcudart.so)" /cuda-artifacts/libcudart.so

FROM jrottenberg/ffmpeg:7.1-nvidia2404 AS runtime

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/scripts ./scripts
COPY backend/app ./app

# CUDA I/Q plotter — last in this stage for better cache when backend/resouce-grid-plotter/ changes.
COPY --from=cuda-build /cuda-artifacts/libcudart.so /usr/local/lib/libcudart.so.12
COPY --from=cuda-build /cuda-artifacts/resgrid_plotter /usr/local/bin/resgrid_plotter
RUN ldconfig && chmod +x /usr/local/bin/resgrid_plotter

# Base image sets ENTRYPOINT to ffmpeg; without clearing it, `command` becomes extra ffmpeg args.
ENTRYPOINT []
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]
