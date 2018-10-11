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

package com.google.devtools.build.lib.profiler.memory;

import com.google.common.base.Objects;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.MapMaker;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleFunction;
import com.google.devtools.build.lib.syntax.ASTNode;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Callstack;
import com.google.monitoring.runtime.instrumentation.Sampler;
import com.google.perftools.profiles.ProfileProto.Function;
import com.google.perftools.profiles.ProfileProto.Line;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.perftools.profiles.ProfileProto.Sample;
import com.google.perftools.profiles.ProfileProto.ValueType;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.zip.GZIPOutputStream;
import javax.annotation.Nullable;

/** Tracks allocations for memory reporting. */
@ConditionallyThreadCompatible
public class AllocationTracker implements Sampler {

  private static class AllocationSample {
    @Nullable final RuleClass ruleClass; // Current rule being analysed, if any
    @Nullable final AspectClass aspectClass; // Current aspect being analysed, if any
    final List<Object> callstack; // Skylark callstack, if any
    final long bytes;

    AllocationSample(
        @Nullable RuleClass ruleClass,
        @Nullable AspectClass aspectClass,
        List<Object> callstack,
        long bytes) {
      this.ruleClass = ruleClass;
      this.aspectClass = aspectClass;
      this.callstack = callstack;
      this.bytes = bytes;
    }
  }

  private final Map<Object, AllocationSample> allocations = new MapMaker().weakKeys().makeMap();
  private final int samplePeriod;
  private final int sampleVariance;
  private boolean enabled = true;

  /**
   * Cheap wrapper class for a long. Avoids having to do two thread-local lookups per allocation.
   */
  private static final class LongValue {
    long value;
  }

  private final ThreadLocal<LongValue> currentSampleBytes = ThreadLocal.withInitial(LongValue::new);
  private final ThreadLocal<Long> nextSampleBytes = ThreadLocal.withInitial(this::getNextSample);
  private final Random random = new Random();

  AllocationTracker(int samplePeriod, int variance) {
    this.samplePeriod = samplePeriod;
    this.sampleVariance = variance;
  }

  @Override
  @ThreadSafe
  public void sampleAllocation(int count, String desc, Object newObj, long size) {
    if (!enabled) {
      return;
    }
    List<Object> callstack = Callstack.get();
    RuleClass ruleClass = CurrentRuleTracker.getRule();
    AspectClass aspectClass = CurrentRuleTracker.getAspect();
    // Should we bother sampling?
    if (callstack.isEmpty() && ruleClass == null && aspectClass == null) {
      return;
    }
    // If we start getting stack overflows here, it's because the memory sampling
    // implementation has changed to call back into the sampling method immediately on
    // every allocation. Since thread locals can allocate, this can in this case lead
    // to infinite recursion. This method will then need to be rewritten to not
    // allocate, or at least not allocate to obtain its sample counters.
    LongValue bytesValue = currentSampleBytes.get();
    long bytes = bytesValue.value + size;
    if (bytes < nextSampleBytes.get()) {
      bytesValue.value = bytes;
      return;
    }
    bytesValue.value = 0;
    nextSampleBytes.set(getNextSample());
    allocations.put(
        newObj,
        new AllocationSample(ruleClass, aspectClass, ImmutableList.copyOf(callstack), bytes));
  }

  private long getNextSample() {
    return (long) samplePeriod
        + (sampleVariance > 0 ? (random.nextInt(sampleVariance * 2) - sampleVariance) : 0);
  }

  /** A pair of rule/aspect name and the bytes it consumes. */
  public static class RuleBytes {
    private final String name;
    private long bytes;

    public RuleBytes(String name) {
      this.name = name;
    }

    /** The name of the rule or aspect. */
    public String getName() {
      return name;
    }

    /** The number of bytes total occupied by this rule or aspect class. */
    public long getBytes() {
      return bytes;
    }

    public RuleBytes addBytes(long bytes) {
      this.bytes += bytes;
      return this;
    }

    @Override
    public String toString() {
      return String.format("RuleBytes(%s, %d)", name, bytes);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      RuleBytes ruleBytes = (RuleBytes) o;
      return bytes == ruleBytes.bytes && Objects.equal(name, ruleBytes.name);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(name, bytes);
    }
  }

  @Nullable
  private static RuleFunction getRuleCreationCall(AllocationSample allocationSample) {
    Object topOfCallstack = Iterables.getLast(allocationSample.callstack, null);
    if (topOfCallstack instanceof RuleFunction) {
      return (RuleFunction) topOfCallstack;
    }
    return null;
  }

  /**
   * Returns the total memory consumption for rules and aspects, keyed by {@link RuleClass#getKey}
   * or {@link AspectClass#getKey}.
   */
  public void getRuleMemoryConsumption(
      Map<String, RuleBytes> rules, Map<String, RuleBytes> aspects) {
    // Make sure we don't track our own allocations
    enabled = false;
    System.gc();

    // Get loading phase memory for rules.
    for (AllocationSample allocationSample : allocations.values()) {
      RuleFunction ruleCreationCall = getRuleCreationCall(allocationSample);
      if (ruleCreationCall != null) {
        RuleClass ruleClass = ruleCreationCall.getRuleClass();
        String key = ruleClass.getKey();
        RuleBytes ruleBytes = rules.computeIfAbsent(key, k -> new RuleBytes(ruleClass.getName()));
        rules.put(key, ruleBytes.addBytes(allocationSample.bytes));
      }
    }
    // Get analysis phase memory for rules and aspects
    for (AllocationSample allocationSample : allocations.values()) {
      if (allocationSample.ruleClass != null) {
        String key = allocationSample.ruleClass.getKey();
        RuleBytes ruleBytes =
            rules.computeIfAbsent(key, k -> new RuleBytes(allocationSample.ruleClass.getName()));
        rules.put(key, ruleBytes.addBytes(allocationSample.bytes));
      }
      if (allocationSample.aspectClass != null) {
        String key = allocationSample.aspectClass.getKey();
        RuleBytes ruleBytes =
            aspects.computeIfAbsent(
                key, k -> new RuleBytes(allocationSample.aspectClass.getName()));
        aspects.put(key, ruleBytes.addBytes(allocationSample.bytes));
      }
    }

    enabled = true;
  }

  /** Dumps all skylark analysis time allocations to a pprof-compatible file. */
  public void dumpSkylarkAllocations(String path) throws IOException {
    // Make sure we don't track our own allocations
    enabled = false;
    System.gc();
    Profile profile = buildMemoryProfile();
    try (GZIPOutputStream outputStream =
        new GZIPOutputStream(Files.newOutputStream(Paths.get(path)))) {
      profile.writeTo(outputStream);
      outputStream.finish();
    }
    enabled = true;
  }

  Profile buildMemoryProfile() {
    Profile.Builder profile = Profile.newBuilder();
    StringTable stringTable = new StringTable(profile);
    FunctionTable functionTable = new FunctionTable(profile, stringTable);
    LocationTable locationTable = new LocationTable(profile, functionTable);
    profile.addSampleType(
        ValueType.newBuilder()
            .setType(stringTable.get("memory"))
            .setUnit(stringTable.get("bytes"))
            .build());
    for (AllocationSample allocationSample : allocations.values()) {
      // Skip empty callstacks
      if (allocationSample.callstack.isEmpty()) {
        continue;
      }
      Sample.Builder sample = Sample.newBuilder().addValue(allocationSample.bytes);
      int line = -1;
      String file = null;
      for (int i = allocationSample.callstack.size() - 1; i >= 0; --i) {
        Object object = allocationSample.callstack.get(i);
        if (line == -1) {
          final Location location;
          if (object instanceof ASTNode) {
            location = ((ASTNode) object).getLocation();
          } else if (object instanceof BaseFunction) {
            location = ((BaseFunction) object).getLocation();
          } else {
            throw new IllegalStateException(
                "Unknown node type: " + object.getClass().getSimpleName());
          }
          if (location != null) {
            file = location.getPath() != null ? location.getPath().getPathString() : "<unknown>";
            line = location.getStartLine() != null ? location.getStartLine() : -1;
          } else {
            file = "<native>";
          }
        }
        String function = null;
        if (object instanceof BaseFunction) {
          BaseFunction baseFunction = (BaseFunction) object;
          function = baseFunction.getName();
        }
        if (function != null) {
          sample.addLocationId(
              locationTable.get(Strings.nullToEmpty(file), Strings.nullToEmpty(function), line));
          line = -1;
          file = null;
        }
      }
      profile.addSample(sample.build());
    }
    profile.setTimeNanos(Instant.now().getEpochSecond() * 1000000000);
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

    long get(String file, String function) {
      return table.computeIfAbsent(
          file + "#" + function,
          key -> {
            Function fn =
                Function.newBuilder()
                    .setId(index)
                    .setFilename(stringTable.get(file))
                    .setName(stringTable.get(function))
                    .build();
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

    long get(String file, String function, long line) {
      return table.computeIfAbsent(
          file + "#" + function + "#" + line,
          key -> {
            com.google.perftools.profiles.ProfileProto.Location location =
                com.google.perftools.profiles.ProfileProto.Location.newBuilder()
                    .setId(index)
                    .addLine(
                        Line.newBuilder()
                            .setFunctionId(functionTable.get(file, function))
                            .setLine(line)
                            .build())
                    .build();
            profile.addLocation(location);
            return index++;
          });
    }
  }
}
