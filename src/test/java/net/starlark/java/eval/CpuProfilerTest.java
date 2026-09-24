// Copyright 2020 The Bazel Authors. All rights reserved.
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
package net.starlark.java.eval;

import com.google.common.io.ByteStreams;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.time.Duration;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;

/**
 * CpuProfilerTest is a simple integration test that the Starlark CPU profiler emits minimally
 * plausible pprof-compatible output.
 *
 * <p>It runs under Blaze only, because it requires a pprof executable.
 */
public final class CpuProfilerTest {

  private CpuProfilerTest() {} // uninstantiable

  static {
    CpuProfiler.setNativeSupport(new CpuProfilerNativeSupportImpl());
  }

  public static void main(String[] args) throws Exception {
    String pprofCmd = args.length == 0 ? "/bin/pprof" : args[0];
    if (!new File(pprofCmd).exists()) {
      throw new AssertionError("no pprof command: " + pprofCmd);
    }

    // This test will fail during profiling of the Java tests
    // because a process (the JVM) can have only one profiler.
    // That's ok; just ignore it.

    // Start writing profile to temporary file.
    File profile = java.io.File.createTempFile("pprof", ".gz", null);
    OutputStream prof = new FileOutputStream(profile);
    boolean success = Starlark.startCpuProfile(prof, Duration.ofMillis(10));

    if (!success) {
      System.err.println("Failed to start cpu profiler");
      System.exit(1);
    }

    // This program consumes about 5s of CPU.
    ParserInput input =
        ParserInput.fromLines(
            """
            x = [0]

            def f():
                for i in range(10000):
                    g()

            def g():
                for _ in range(1000):
                    list(range(10))
                int(3)
                sorted(range(10000))

            f()
            """);

    // Execute the workload.
    Module module = Module.create();
    try (Mutability mu = Mutability.create("test")) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      Starlark.execFile(input, FileOptions.DEFAULT, module, thread);
    }

    Starlark.stopCpuProfile();

    // Run pprof -top. Typical output (may vary by pprof release):
    //
    // Type: CPU
    // Time: 2026-08-04 13:40:18 PDT
    // Duration: 4.59s, Total samples = 8.27s (180.24%)
    // Showing nodes accounting for 8.26s, 99.88% of 8.27s total
    // Dropped 1 node (cum <= 0.04s)
    //       flat  flat%   sum%        cum   cum%
    //      2.69s 32.53% 32.53%      8.21s 99.27%  g
    //      2.60s 31.44% 63.97%      2.60s 31.44%  list
    //      1.61s 19.47% 83.43%      1.61s 19.47%  range
    //      1.32s 15.96% 99.40%      1.32s 15.96%  sorted
    //      0.03s  0.36% 99.76%      8.27s   100%  <unknown>
    //      0.01s  0.12% 99.88%      8.24s 99.64%  f

    // Runtime.exec is deprecated at Google but its open-source replacement is not yet available.
    @SuppressWarnings("RuntimeExec")
    Process pprof =
        Runtime.getRuntime()
            .exec(pprofCmd + " -top " + profile, /* envp= */ new String[0], /* dir= */ null);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteStreams.copy(pprof.getInputStream(), out);
    String got = out.toString(); // encoding undefined but unimportant---result is ASCII

    // We'll assert that a few key substrings are present.
    boolean ok = true;
    for (String want : new String[] {"flat%", "list", "sorted", "range"}) {
      if (!got.contains(want)) {
        System.err.println("pprof output does not contain substring: " + want);
        ok = false;
      }
    }
    if (!ok) {
      System.err.println("pprof output:\n" + out);
      System.exit(1);
    }
  }
}
