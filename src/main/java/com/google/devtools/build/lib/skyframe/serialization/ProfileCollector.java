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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.perftools.profiles.ProfileProto.Function;
import com.google.perftools.profiles.ProfileProto.Line;
import com.google.perftools.profiles.ProfileProto.Location;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.perftools.profiles.ProfileProto.Sample;
import com.google.perftools.profiles.ProfileProto.ValueType;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Collects serialization profiling data.
 *
 * <p>This class is thread-safe.
 */
public final class ProfileCollector {
  @VisibleForTesting static final String SAMPLES = "samples";
  @VisibleForTesting static final String COUNT = "count";
  @VisibleForTesting static final String STORAGE = "storage";
  @VisibleForTesting static final String BYTES = "bytes";

  private final ConcurrentHashMap<ImmutableList<ProfilerLocationProvider>, Counts> records =
      new ConcurrentHashMap<>();

  /**
   * Records a sample.
   *
   * <p>For ease of implementation, samples are recorded here as transitive bytes. The underlying
   * proto defines samples as self-bytes so there is a cleanup step that converts the transitive
   * byte count to self-bytes by subtracting up the stack.
   *
   * @param locationStack a path of descriptions of the root object being serialized down to the
   *     current object being serialized
   * @param byteCount the transitive bytes serialized at the given object
   */
  public void recordSample(List<ProfilerLocationProvider> locationStack, int byteCount) {
    var counts = getCounts(locationStack);
    counts.count().getAndIncrement();
    counts.totalBytes().getAndAdd(byteCount);

    // Subtracts bytes from the ancestor to avoid double counting.
    if (locationStack.size() > 1) {
      List<ProfilerLocationProvider> prefix = locationStack.subList(0, locationStack.size() - 1);
      getCounts(prefix).totalBytes().getAndAdd(-byteCount);
    }
  }

  /**
   * Records a batch of samples.
   *
   * <p>This is used to merge samples from a {@link ProfileRecorder} after its novelty check
   * completes.
   *
   * @param samples a map from location stack to the number of samples and transitive bytes recorded
   *     at that location
   */
  void recordSamples(Map<ImmutableList<ProfilerLocationProvider>, Counts> samples) {
    samples.forEach(
        (stack, counts) -> {
          int count = counts.count().get();
          int byteCount = counts.totalBytes().get();
          var target = getCounts(stack);
          target.count().getAndAdd(count);
          target.totalBytes().getAndAdd(byteCount);

          // Subtracts bytes from the ancestor to avoid double counting.
          if (stack.size() > 1) {
            ImmutableList<ProfilerLocationProvider> prefix = stack.subList(0, stack.size() - 1);
            getCounts(prefix).totalBytes().getAndAdd(-byteCount);
          }
        });
  }

  /** Creates the {@link Profile} from the accumulated samples. */
  public Profile toProto() {
    var profileBuilder = new ProtoBuilder();
    records.forEach(
        (stack, counts) -> {
          int count = counts.count().get();
          int byteCount = counts.totalBytes().get();
          if (count == 0 && byteCount == 0) {
            return;
          }
          var sample = Sample.newBuilder().addValue(count).addValue(byteCount);
          for (ProfilerLocationProvider provider : Lists.reverse(stack)) {
            sample.addLocationId(profileBuilder.getOrAddLocation(provider.getLocationText()));
          }
          profileBuilder.addSample(sample);
        });
    return profileBuilder.build();
  }

  /** Stores the profiling counts associated with {@code stack}. */
  record Counts(
      ImmutableList<ProfilerLocationProvider> stack,
      AtomicInteger count,
      AtomicInteger totalBytes) {
    Counts(ImmutableList<ProfilerLocationProvider> stack) {
      this(stack, new AtomicInteger(), new AtomicInteger());
    }
  }

  /**
   * Obtains a deduplicated instance of the {@code stack}.
   *
   * <p>In practice, many parallel instances of {@link ProfileRecorder} will be in flight
   * simultaneously and each retains stack instances. This allows the memory for those stack
   * instances to be shared.
   */
  ImmutableList<ProfilerLocationProvider> getCanonicalStack(List<ProfilerLocationProvider> stack) {
    return getCounts(stack).stack();
  }

  private Counts getCounts(List<ProfilerLocationProvider> locationStack) {
    Counts counts = records.get(locationStack);
    if (counts != null) {
      return counts;
    }
    var stack = ImmutableList.copyOf(locationStack);
    // putIfAbsent has less contention than computeIfAbsent because the latter causes the allocation
    // of Counts to be inside the critical section.
    var newCounts = new Counts(stack);
    Counts previousCounts = records.putIfAbsent(stack, newCounts);
    if (previousCounts != null) {
      return previousCounts;
    }
    return newCounts;
  }

  private static class ProtoBuilder {
    private final HashMap<String, Integer> stringTableBuilder = new HashMap<>();
    private final HashMap<String, Integer> locationTableBuilder = new HashMap<>();
    private final Profile.Builder profile = Profile.newBuilder();

    private ProtoBuilder() {
      // Puts the empty string in the 0 position as required by the schema.
      int unusedEmptyId = getOrAddString("");
      int samplesId = getOrAddString(SAMPLES);
      int countId = getOrAddString(COUNT);
      int storageId = getOrAddString(STORAGE);
      int bytesId = getOrAddString(BYTES);

      // Prepopulates the schema fields. Each data point has a sample count with units "count" and a
      // storage size with units "bytes".
      profile
          .addSampleType(ValueType.newBuilder().setType(samplesId).setUnit(countId))
          .addSampleType(ValueType.newBuilder().setType(storageId).setUnit(bytesId));
    }

    private int getOrAddString(String text) {
      Integer existingId = stringTableBuilder.get(text);
      if (existingId != null) {
        return existingId;
      }
      int id = stringTableBuilder.size();
      stringTableBuilder.put(text, id);
      profile.addStringTable(text);
      return id;
    }

    private int getOrAddLocation(String name) {
      Integer existingId = locationTableBuilder.get(name);
      if (existingId != null) {
        return existingId;
      }

      int stringIndex = getOrAddString(name);
      int locationId = locationTableBuilder.size() + 1; // 0 is reserved
      locationTableBuilder.put(name, locationId);

      // Function and Location are 1-1 here so the IDs are the same.
      profile
          .addFunction(Function.newBuilder().setId(locationId).setName(stringIndex))
          .addLocation(
              Location.newBuilder()
                  .setId(locationId)
                  .addLine(Line.newBuilder().setFunctionId(locationId)));

      return locationId;
    }

    private void addSample(Sample.Builder sample) {
      profile.addSample(sample);
    }

    private Profile build() {
      return profile.build();
    }
  }
}
