package com.google.devtools.build.lib.vfs.bazel;

import com.google.common.hash.HashCode;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import java.security.SecureRandom;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;

@BenchmarkMode(Mode.Throughput)
@State(Scope.Benchmark)
// Realistic usage within Bazel will hash files on multiple threads.
@Threads(4)
public class BazelHashFunctionsBenchmark {

  static {
    BazelHashFunctions.ensureRegistered();
  }

  public enum HashFunctionType {
    BLAKE3(new Blake3HashFunction()),
    SHA2_256(Hashing.sha256());

    final HashFunction hashFunction;

    HashFunctionType(HashFunction hashFunction) {
      this.hashFunction = hashFunction;
    }
  }

  @Param({"BLAKE3", "SHA2_256"})
  public HashFunctionType type;

  @Param({"1", "16", "128", "512", "1024", "4096", "16384", "1048576"})
  public int size;

  private byte[] data;

  @Setup(Level.Iteration)
  public void setup() {
    data = new byte[size];
    new SecureRandom().nextBytes(data);
  }

  @Benchmark
  public HashCode hashBytesOneShot() {
    return type.hashFunction.hashBytes(data);
  }

  private static final int CHUNK_SIZE = 4096;

  @Benchmark
  public HashCode hashBytesChunks() {
    Hasher hasher = type.hashFunction.newHasher();
    for (int pos = 0; pos < data.length; pos += CHUNK_SIZE) {
      hasher.putBytes(data, pos, Math.min(CHUNK_SIZE, data.length - pos));
    }
    return hasher.hash();
  }
}
