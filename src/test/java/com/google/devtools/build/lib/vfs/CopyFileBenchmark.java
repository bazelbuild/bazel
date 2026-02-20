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

package com.google.devtools.build.lib.vfs;

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.windows.WindowsFileSystem;
import java.io.IOException;
import java.nio.file.Files;
import java.security.SecureRandom;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.Throughput)
@State(Scope.Benchmark)
@Fork(value = 1, warmups = 1)
@Measurement(iterations = 5, time = 1)
@Warmup(iterations = 5, time = 1)
public class CopyFileBenchmark {

  private Path sourceFile;
  private Path targetFile;

  @Setup(Level.Iteration)
  public void setup() throws IOException {
    var fs =
        OS.getCurrent() == OS.WINDOWS
            ? new WindowsFileSystem(DigestHashFunction.SHA256, /* enableSymlinks= */ true)
            : new UnixFileSystem(DigestHashFunction.SHA256, "");
    var tmpDir = fs.getPath(Files.createTempDirectory("").toString());
    var destDir = tmpDir.createTempDirectory("benchmark");
    destDir.createDirectoryAndParents();

    sourceFile = tmpDir.getChild("benchmark_file");
    var buffer = new byte[1024 * 1024];
    var random = new SecureRandom();
    random.nextBytes(buffer);
    try (var out = sourceFile.getOutputStream()) {
      out.write(buffer);
    }

    targetFile = destDir.getChild("copied_benchmark_file");
  }

  @Setup(Level.Invocation)
  public void prepareTarget() throws IOException {
    targetFile.delete();
  }

  @Benchmark
  public void baselineCopy() throws IOException {
    try (var in = sourceFile.getInputStream();
        var out = targetFile.getOutputStream()) {
      ByteStreams.copy(in, out);
    }
  }

  @Benchmark
  public void transferTo() throws IOException {
    try (var in = sourceFile.getInputStream();
        var out = targetFile.getOutputStream()) {
      in.transferTo(out);
    }
  }

  @Benchmark
  public void symlink() throws IOException {
    targetFile.createSymbolicLink(sourceFile);
  }

  @Benchmark
  public void copyFile() throws IOException {
    FileSystemUtils.copyFile(sourceFile, targetFile);
  }

  @Benchmark
  public void copyRegularFile() throws IOException {
    FileSystemUtils.copyRegularFile(sourceFile, targetFile);
  }
}
