// Copyright 2026 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.remote.chunking;

import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.bazel.BazelHashFunctions;
import java.io.ByteArrayInputStream;
import java.security.SecureRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.Throughput)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 5, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 5, timeUnit = TimeUnit.SECONDS)
@Fork(3)
public class FastCDCBenchmark {
  private static final int AVG_CHUNK_SIZE = 512 * 1024;

  @Param({"1048576", "8388608", "67108864"})
  public int size;

  private byte[] data;
  private FastCDCChunker chunker;

  @Setup(Level.Iteration)
  public void setup() {
    BazelHashFunctions.ensureRegistered();
    data = new byte[size];
    new SecureRandom().nextBytes(data);

    DigestUtil digestUtil =
        new DigestUtil(SyscallCache.NO_CACHE, BazelHashFunctions.BLAKE3);
    int minSize = AVG_CHUNK_SIZE / 4;
    int maxSize = AVG_CHUNK_SIZE * 4;
    chunker = new FastCDCChunker(minSize, AVG_CHUNK_SIZE, maxSize, 2, 0, digestUtil);
  }

  @Benchmark
  public Object chunkToDigests() throws Exception {
    return chunker.chunkToDigests(new ByteArrayInputStream(data));
  }
}
