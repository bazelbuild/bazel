package com.google.devtools.build.lib.remote.zstd;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@BenchmarkMode(Mode.Throughput)
@State(Scope.Benchmark)
public class ZstdBenchmark {
  @Param({"4096", "4194304"})
  public int size;

  private byte[] uncompressedData;
  private byte[] compressedData;

  @Setup
  public void setup() {
    uncompressedData = new byte[size];
    for (int i = 0; i < size; i++) {
      uncompressedData[i] = (byte) (i % 256);
    }
    try {
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      ZstdCompressingInputStream zci =
          new ZstdCompressingInputStream(new ByteArrayInputStream(uncompressedData));
      zci.transferTo(baos);
      compressedData = baos.toByteArray();
    } catch (Exception e) {
      throw new RuntimeException("Failed to compress data", e);
    }
  }

  @Benchmark
  public ByteArrayOutputStream compress() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (var zci = new ZstdCompressingInputStream(new ByteArrayInputStream(uncompressedData))) {
      zci.transferTo(baos);
    }
    return baos;
  }

  @Benchmark
  public ByteArrayOutputStream decompress() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (var zdos = new ZstdDecompressingOutputStream(baos)) {
      zdos.write(compressedData);
    }
    return baos;
  }
}
