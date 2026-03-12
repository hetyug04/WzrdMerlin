FROM rust:1.90-slim-bookworm as builder

WORKDIR /app

# Copy Cargo.toml from current dir
COPY Cargo.toml ./
# Create a dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release

# Copy real source (main.rs is in the current dir when context is ./src/sidecar)
COPY main.rs ./src/
# Build for release
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/merlin-sidecar .

CMD ["./merlin-sidecar"]
