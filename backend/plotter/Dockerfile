# Build: docker build -t resgrid-plotter .
# Run: see docker-compose profile `iq`
FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    cmake \
    ninja-build \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src

RUN cmake -S . -B out -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build out -j"$(nproc)" \
    && cp out/resgrid_plotter /usr/local/bin/

ENTRYPOINT ["resgrid_plotter"]
CMD ["--rtsp", "rtsp://mediamtx:8554/live", "--fps", "200", "--encoder-fps", "24", "--cell-pixels", "6", "--preset", "p4"]
