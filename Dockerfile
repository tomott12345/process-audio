# syntax=docker/dockerfile:1.7
#
# process-audio: the web app (Go) plus the audio pipelines (Python + ffmpeg)
# in one image.
#
#   docker build -t process-audio .
#   docker run -d --name process-audio -p 127.0.0.1:8765:8765 -v process-audio-data:/data process-audio
#   open http://127.0.0.1:8765
#
# Publish the port on 127.0.0.1 only: the app has no login yet, and anyone
# who can reach the port can upload files and run jobs.
#
# Build args:
#   WITH_DEMUCS=1   add torch (CPU) + Demucs for the nature pipeline's
#                   voice/footstep removal (several GB larger)

ARG PYTHON_VERSION=3.12
ARG GO_VERSION=1.24

# ---------------------------------------------------------------- Go build
FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-bookworm AS gobuild
ARG TARGETOS
ARG TARGETARCH
WORKDIR /src/web
COPY web/go.mod ./
COPY web/ ./
# static binary, cross-compiled for the target platform (no CGO, no QEMU)
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -trimpath -ldflags="-s -w" -o /out/paweb ./cmd/paweb

# ---------------------------------------------------------------- runtime
# Debian trixie: its ffmpeg (7.1) has every filter the pipelines use
# (adynamicequalizer, showspatial, lagfun, ...); bookworm's 5.1 is too old.
FROM python:${PYTHON_VERSION}-slim-trixie AS runtime

LABEL org.opencontainers.image.title="process-audio" \
      org.opencontainers.image.description="Web app + ffmpeg/Python pipelines for music, nature, and speech recordings" \
      org.opencontainers.image.source="https://github.com/tomott12345/process-audio" \
      org.opencontainers.image.licenses="NOASSERTION"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

# ffmpeg; DejaVu for titles; Noto Color Emoji for the visualizer's emoji
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core fonts-noto-color-emoji \
 && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

ARG WITH_DEMUCS=0
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install demucs ; \
    fi

WORKDIR /app
# the pipelines: scripts + recipes at the top of the repo, and their tests
COPY *.py *.json pytest.ini ./
COPY tests/ ./tests/
COPY --from=gobuild /out/paweb /usr/local/bin/paweb

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
 && mkdir -p /data \
 && chown app:app /data
USER app

VOLUME /data
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=4)" || exit 1

ENTRYPOINT ["paweb", "-repo", "/app", "-data", "/data", "-python", "python3", "-container"]
CMD ["-addr", "0.0.0.0:8765"]
