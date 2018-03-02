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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Splitter;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.time.Duration;
import java.util.Iterator;
import java.util.NoSuchElementException;
import javax.annotation.Nullable;

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
  private long heapUsedMemoryAtFinish;
  @Nullable private MemoryProfileStableHeapParameters memoryProfileStableHeapParameters;

  public synchronized void setStableMemoryParameters(
      MemoryProfileStableHeapParameters memoryProfileStableHeapParameters) {
    this.memoryProfileStableHeapParameters = memoryProfileStableHeapParameters;
  }

  public synchronized void start(OutputStream out) {
    this.memoryProfile = (out == null) ? null : new PrintStream(out);
    this.currentPhase = ProfilePhase.INIT;
    heapUsedMemoryAtFinish = 0;
  }

  public synchronized void stop() {
    if (memoryProfile != null) {
      memoryProfile.close();
      memoryProfile = null;
    }
    heapUsedMemoryAtFinish = 0;
  }

  public synchronized long getHeapUsedMemoryAtFinish() {
    return heapUsedMemoryAtFinish;
  }

  public synchronized void markPhase(ProfilePhase nextPhase) throws InterruptedException {
    if (memoryProfile != null) {
      MemoryMXBean bean = ManagementFactory.getMemoryMXBean();
      prepareBean(nextPhase, bean, (duration) -> Thread.sleep(duration.toMillis()));
      String name = currentPhase.description;
      MemoryUsage memoryUsage = bean.getHeapMemoryUsage();
      memoryProfile.println(name + ":heap:init:" + memoryUsage.getInit());
      memoryProfile.println(name + ":heap:used:" + memoryUsage.getUsed());
      memoryProfile.println(name + ":heap:commited:" + memoryUsage.getCommitted());
      memoryProfile.println(name + ":heap:max:" + memoryUsage.getMax());
      if (nextPhase == ProfilePhase.FINISH) {
        heapUsedMemoryAtFinish = memoryUsage.getUsed();
      }

      memoryUsage = bean.getNonHeapMemoryUsage();
      memoryProfile.println(name + ":non-heap:init:" + memoryUsage.getInit());
      memoryProfile.println(name + ":non-heap:used:" + memoryUsage.getUsed());
      memoryProfile.println(name + ":non-heap:commited:" + memoryUsage.getCommitted());
      memoryProfile.println(name + ":non-heap:max:" + memoryUsage.getMax());
      currentPhase = nextPhase;
    }
  }

  @VisibleForTesting
  synchronized void prepareBean(ProfilePhase nextPhase, MemoryMXBean bean, Sleeper sleeper)
      throws InterruptedException {
    bean.gc();
    if (nextPhase == ProfilePhase.FINISH && memoryProfileStableHeapParameters != null) {
      for (int i = 1; i < memoryProfileStableHeapParameters.numTimesToDoGc; i++) {
        sleeper.sleep(memoryProfileStableHeapParameters.timeToSleepBetweenGcs);
        bean.gc();
      }
    }
  }

  /**
   * Parameters that control how {@code MemoryProfiler} tries to get a stable heap at the end of the
   * build.
   */
  public static class MemoryProfileStableHeapParameters {
    private final int numTimesToDoGc;
    private final Duration timeToSleepBetweenGcs;

    private MemoryProfileStableHeapParameters(int numTimesToDoGc, Duration timeToSleepBetweenGcs) {
      this.numTimesToDoGc = numTimesToDoGc;
      this.timeToSleepBetweenGcs = timeToSleepBetweenGcs;
    }

    /** Converter for {@code MemoryProfileStableHeapParameters} option. */
    public static class Converter
        implements com.google.devtools.common.options.Converter<MemoryProfileStableHeapParameters> {
      private static final Splitter SPLITTER = Splitter.on(',');

      @Override
      public MemoryProfileStableHeapParameters convert(String input)
          throws OptionsParsingException {
        Iterator<String> values = SPLITTER.split(input).iterator();
        try {
          int numTimesToDoGc = Integer.parseInt(values.next());
          int numSecondsToSleepBetweenGcs = Integer.parseInt(values.next());
          if (values.hasNext()) {
            throw new OptionsParsingException("Expected exactly 2 comma-separated integer values");
          }
          if (numTimesToDoGc <= 0) {
            throw new OptionsParsingException("Number of times to GC must be positive");
          }
          if (numSecondsToSleepBetweenGcs < 0) {
            throw new OptionsParsingException(
                "Number of seconds to sleep between GC's must be positive");
          }
          return new MemoryProfileStableHeapParameters(
              numTimesToDoGc, Duration.ofSeconds(numSecondsToSleepBetweenGcs));
        } catch (NumberFormatException | NoSuchElementException nfe) {
          throw new OptionsParsingException(
              "Expected exactly 2 comma-separated integer values", nfe);
        }
      }

      @Override
      public String getTypeDescription() {
        return "two integers, separated by a comma";
      }
    }
  }

  @VisibleForTesting
  interface Sleeper {
    void sleep(Duration duration) throws InterruptedException;
  }
}
