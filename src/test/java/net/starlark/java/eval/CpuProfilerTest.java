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

import com.google.perftools.profiles.ProfileProto.Function;
import com.google.perftools.profiles.ProfileProto.Line;
import com.google.perftools.profiles.ProfileProto.Location;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.perftools.profiles.ProfileProto.Sample;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.Duration;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.zip.GZIPInputStream;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;

/**
 * CpuProfilerTest is a simple integration test that the Starlark CPU profiler emits minimally
 * plausible pprof-compatible output.
 */
public final class CpuProfilerTest {

  private CpuProfilerTest() {} // uninstantiable

  static {
    CpuProfiler.setNativeSupport(new CpuProfilerNativeSupportImpl());
  }

  public static void main(String[] args) throws Exception {
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

    Profile proto;
    try (InputStream in = new GZIPInputStream(new FileInputStream(profile))) {
      proto = Profile.parseFrom(in);
    }
    var functionNames = new HashMap<Long, String>();
    for (Function function : proto.getFunctionList()) {
      functionNames.put(function.getId(), proto.getStringTable((int) function.getName()));
    }
    var locations = new HashMap<Long, Location>();
    for (Location location : proto.getLocationList()) {
      locations.put(location.getId(), location);
    }
    var sampledFunctions = new TreeSet<String>();
    for (Sample sample : proto.getSampleList()) {
      for (long locationId : sample.getLocationIdList()) {
        for (Line line : locations.get(locationId).getLineList()) {
          sampledFunctions.add(functionNames.get(line.getFunctionId()));
        }
      }
    }

    // We'll assert that a few key functions have been sampled.
    boolean ok = true;
    for (String want : new String[] {"list", "sorted", "range"}) {
      if (!sampledFunctions.contains(want)) {
        System.err.println("profile contains no samples in function: " + want);
        ok = false;
      }
    }
    if (!ok) {
      System.err.println("sampled functions: " + sampledFunctions);
      System.exit(1);
    }
  }
}
