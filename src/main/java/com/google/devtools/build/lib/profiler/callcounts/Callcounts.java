// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.profiler.callcounts;

import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.perftools.profiles.ProfileProto.Function;
import com.google.perftools.profiles.ProfileProto.Line;
import com.google.perftools.profiles.ProfileProto.Location;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.perftools.profiles.ProfileProto.Sample;
import com.google.perftools.profiles.ProfileProto.ValueType;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPOutputStream;

/**
 * Developer utility. Add calls to {@link Callcounts#registerCall} in places that you want to count
 * call occurrences with call stacks. At the end of the command, a pprof-compatible file will be
 * dumped to the path specified by --call_count_output_path (see also {@link CallcountsModule}).
 */
public class Callcounts {
  private static int maxCallstackDepth;
  // Every Nth call is actually logged
  private static int samplePeriod;
  /**
   * We use some variance to avoid patterns where the same calls get logged over and over.
   *
   * <p>Without a little variance, we could end up having call patterns that line up with the sample
   * period, leading to the same calls getting sampled over and over again. For instance, consider
   * 90 calls to method A followed by 10 calls to method B, over and over. If our sample period was
   * 100 then we could end up in a situation where either only method A or (worse) only method B
   * gets sampled.
   */
  private static int sampleVariance;

  private static final Random random = new Random();
  private static final AtomicLong currentSampleCount = new AtomicLong();
  private static final AtomicLong nextSampleCount = new AtomicLong();

  private static final Interner<Callstack> callstacks =
      Interners.newBuilder().weak().concurrencyLevel(8).build();
  private static final Map<Callstack, Long> callstackCounts = new ConcurrentHashMap<>();

  static class Callstack {
    final StackTraceElement[] stackTraceElements;

    Callstack(StackTraceElement[] stackTraceElements) {
      this.stackTraceElements = stackTraceElements;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      Callstack entry = (Callstack) o;
      return Arrays.equals(stackTraceElements, entry.stackTraceElements);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(stackTraceElements);
    }
  }

  /** Call to register a call trace at the current call location */
  public static void registerCall() {
    // All of this is totally thread-unsafe for speed
    // The worst that can happen is we occasionally over-sample
    if (currentSampleCount.incrementAndGet() < nextSampleCount.get()) {
      return;
    }
    long count = 0;
    synchronized (Callcounts.class) {
      // Check again if somebody else already got here first
      // This greatly improves performance compared to eager locking
      if (currentSampleCount.get() < nextSampleCount.get()) {
        return;
      }
      count = currentSampleCount.get();
      resetSampleTarget();
    }

    StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
    int skip = 2; // Otherwise we show up ourselves
    int len = Math.min(stackTrace.length - skip, maxCallstackDepth);
    if (len <= 0) {
      return;
    }

    StackTraceElement[] entries = new StackTraceElement[len];
    for (int i = 0; i < len; ++i) {
      entries[i] = stackTrace[i + skip];
    }
    Callstack callstack = new Callstack(entries);
    callstack = callstacks.intern(callstack);
    callstackCounts.put(callstack, callstackCounts.getOrDefault(callstack, 0L) + count);
  }

  static void resetSampleTarget() {
    currentSampleCount.set(0);
    nextSampleCount.set(
        samplePeriod
            + (sampleVariance > 0 ? random.nextInt(sampleVariance * 2) - sampleVariance : 0));
  }

  static void init(int samplePeriod, int sampleVariance, int maxCallstackDepth) {
    Callcounts.samplePeriod = samplePeriod;
    Callcounts.sampleVariance = sampleVariance;
    Callcounts.maxCallstackDepth = maxCallstackDepth;
    resetSampleTarget();
  }

  static void reset() {
    callstackCounts.clear();
  }

  static void dump(String path) throws IOException {
    Profile profile = createProfile();
    try (GZIPOutputStream gzipOutputStream =
        new GZIPOutputStream(Files.newOutputStream(Paths.get(path)))) {
      profile.writeTo(gzipOutputStream);
    }
  }

  static Profile createProfile() {
    Profile.Builder profile = Profile.newBuilder();
    StringTable stringTable = new StringTable(profile);
    profile.addSampleType(
        ValueType.newBuilder()
            .setType(stringTable.get("calls"))
            .setUnit(stringTable.get("count"))
            .build());
    FunctionTable functionTable = new FunctionTable(profile, stringTable);
    LocationTable locationTable = new LocationTable(profile, functionTable);
    for (Map.Entry<Callstack, Long> entry : callstackCounts.entrySet()) {
      Sample.Builder sample = Sample.newBuilder();
      sample.addValue(entry.getValue());
      for (StackTraceElement stackTraceElement : entry.getKey().stackTraceElements) {
        String name = stackTraceElement.getClassName() + "." + stackTraceElement.getMethodName();
        int line = stackTraceElement.getLineNumber();
        sample.addLocationId(locationTable.get(name, line));
      }
      profile.addSample(sample);
    }
    Instant instant = Instant.now();
    profile.setTimeNanos(instant.getEpochSecond() * 1_000_000_000);
    return profile.build();
  }

  private static class StringTable {
    final Profile.Builder profile;
    final Map<String, Long> table = new HashMap<>();
    long index = 0;

    StringTable(Profile.Builder profile) {
      this.profile = profile;
      get(""); // 0 is reserved for the empty string
    }

    long get(String str) {
      return table.computeIfAbsent(
          str,
          key -> {
            profile.addStringTable(key);
            return index++;
          });
    }
  }

  private static class FunctionTable {
    final Profile.Builder profile;
    final StringTable stringTable;
    final Map<String, Long> table = new HashMap<>();
    long index = 1; // 0 is reserved

    FunctionTable(Profile.Builder profile, StringTable stringTable) {
      this.profile = profile;
      this.stringTable = stringTable;
    }

    long get(String function) {
      return table.computeIfAbsent(
          function,
          key -> {
            Function fn =
                Function.newBuilder().setId(index).setName(stringTable.get(function)).build();
            profile.addFunction(fn);
            return index++;
          });
    }
  }

  private static class LocationTable {
    final Profile.Builder profile;
    final FunctionTable functionTable;
    final Map<String, Long> table = new HashMap<>();
    long index = 1; // 0 is reserved

    LocationTable(Profile.Builder profile, FunctionTable functionTable) {
      this.profile = profile;
      this.functionTable = functionTable;
    }

    long get(String function, long line) {
      return table.computeIfAbsent(
          function + "#" + line,
          key -> {
            Location location =
                Location.newBuilder()
                    .setId(index)
                    .addLine(
                        Line.newBuilder()
                            .setFunctionId(functionTable.get(function))
                            .setLine(line)
                            .build())
                    .build();
            profile.addLocation(location);
            return index++;
          });
    }
  }
}
