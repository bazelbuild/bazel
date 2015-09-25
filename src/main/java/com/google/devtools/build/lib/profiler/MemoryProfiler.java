// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.profiler;

import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;

/**
 * Blaze memory profiler.
 *
 * <p>At each call to {@code profile} performs garbage collection and stores
 * heap and non-heap memory usage in an external file.
 *
 * <p><em>Heap memory</em> is the runtime data area from which memory for all
 * class instances and arrays is allocated. <em>Non-heap memory</em> includes
 * the method area and memory required for the internal processing or
 * optimization of the JVM. It stores per-class structures such as a runtime
 * constant pool, field and method data, and the code for methods and
 * constructors. The Java Native Interface (JNI) code or the native library of
 * an application and the JVM implementation allocate memory from the
 * <em>native heap</em>.
 *
 * <p>The script in /devtools/blaze/scripts/blaze-memchart.sh can be used for post processing.
 */
public final class MemoryProfiler {

  private static final MemoryProfiler INSTANCE = new MemoryProfiler();

  public static MemoryProfiler instance() {
    return INSTANCE;
  }

  private PrintStream memoryProfile;
  private ProfilePhase currentPhase;

  public synchronized void start(OutputStream out) {
    this.memoryProfile = (out == null) ? null : new PrintStream(out);
    this.currentPhase = ProfilePhase.INIT;
  }

  public synchronized void stop() {
    if (memoryProfile != null) {
      memoryProfile.close();
      memoryProfile = null;
    }
  }

  public synchronized void markPhase(ProfilePhase nextPhase) {
    if (memoryProfile != null) {
      String name = currentPhase.description;
      ManagementFactory.getMemoryMXBean().gc();
      MemoryUsage memoryUsage = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
      memoryProfile.println(name + ":heap:init:" + memoryUsage.getInit());
      memoryProfile.println(name + ":heap:used:" + memoryUsage.getUsed());
      memoryProfile.println(name + ":heap:commited:" + memoryUsage.getCommitted());
      memoryProfile.println(name + ":heap:max:" + memoryUsage.getMax());

      memoryUsage = ManagementFactory.getMemoryMXBean().getNonHeapMemoryUsage();
      memoryProfile.println(name + ":non-heap:init:" + memoryUsage.getInit());
      memoryProfile.println(name + ":non-heap:used:" + memoryUsage.getUsed());
      memoryProfile.println(name + ":non-heap:commited:" + memoryUsage.getCommitted());
      memoryProfile.println(name + ":non-heap:max:" + memoryUsage.getMax());
      currentPhase = nextPhase;
    }
  }
}
