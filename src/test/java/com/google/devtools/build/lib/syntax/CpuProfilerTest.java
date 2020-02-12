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
package com.google.devtools.build.lib.syntax;

import com.google.common.io.ByteStreams;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.time.Duration;

/**
 * CpuProfilerTest is a simple integration test that the Starlark CPU profiler emits minimally
 * plausible pprof-compatible output.
 *
 * <p>It runs under Blaze only, because it requires a pprof executable.
 */
public final class CpuProfilerTest {

  private CpuProfilerTest() {} // uninstantiable

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
    Starlark.startCpuProfile(prof, Duration.ofMillis(10));

    // This program consumes about 3s of CPU.
    ParserInput input =
        ParserInput.fromLines(
            "x = [0]", //
            "",
            "def f():",
            "    for i in range(10000):",
            "        g()",
            "",
            "def g():",
            "    list(range(10000))",
            "    int(3)",
            "    sorted(range(10000))",
            "",
            "f()");

    // Execute the workload.
    StarlarkThread thread =
        StarlarkThread.builder(Mutability.create("test"))
            .setGlobals(Module.createForBuiltins(Starlark.UNIVERSE))
            .useDefaultSemantics()
            .build();
    EvalUtils.exec(input, thread.getGlobals(), thread);

    Starlark.stopCpuProfile();

    // Run pprof -top. Typical output (may vary by pprof release):
    //
    // Type: CPU
    // Time: Jan 21, 2020 at 11:08am (PST)
    // Duration: 3.26s, Total samples = 2640ms (80.97%)
    // Showing nodes accounting for 2640ms, 100% of 2640ms total
    //       flat  flat%   sum%        cum   cum%
    //     1390ms 52.65% 52.65%     1390ms 52.65%  sorted
    //      960ms 36.36% 89.02%      960ms 36.36%  list
    //      220ms  8.33% 97.35%      220ms  8.33%  range
    //       70ms  2.65%   100%       70ms  2.65%  int
    //          0     0%   100%     2640ms   100%  <unknown>
    //          0     0%   100%     2640ms   100%  f
    //          0     0%   100%     2640ms   100%  g

    // Runtime.exec is deprecated at Google but its open-source replacement is not yet available.
    @SuppressWarnings("RuntimeExec")
    Process pprof =
        Runtime.getRuntime()
            .exec(pprofCmd + " -top " + profile, /*envp=*/ new String[0], /*dir=*/ null);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteStreams.copy(pprof.getInputStream(), out);
    String got = out.toString(); // encoding undefined but unimportant---result is ASCII

    // We'll assert that a few key substrings are present.
    boolean ok = true;
    for (String want : new String[] {"flat%", "sorted", "range"}) {
      if (!got.contains(want)) {
        System.err.println("pprof output does not contain substring: " + got);
        ok = false;
      }
    }
    if (!ok) {
      System.err.println("pprof output:\n" + out);
      System.exit(1);
    }
  }
}
