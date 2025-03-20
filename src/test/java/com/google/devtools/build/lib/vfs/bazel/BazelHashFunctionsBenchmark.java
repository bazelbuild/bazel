// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.vfs.bazel;

import com.google.common.hash.HashCode;
import com.google.common.hash.HashFunction;
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

@BenchmarkMode(Mode.Throughput)
@State(Scope.Benchmark)
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

  public enum Size {
    B,
    KB,
    MB,
    GB;

    final int bytes;

    Size() {
      bytes = 1 << (ordinal() * 10);
    }
  }

  @Param({"BLAKE3", "SHA2_256"})
  public HashFunctionType type;

  @Param({"B", "KB", "MB", "GB"})
  public Size size;

  private byte[] data;

  @Setup(Level.Iteration)
  public void setup() {
    data = new byte[size.bytes];
    new SecureRandom().nextBytes(data);
  }

  @Benchmark
  public HashCode hashBytesOneShot() {
    return type.hashFunction.hashBytes(data);
  }
}
