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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.perftools.profiles.ProfileProto;
import com.google.perftools.profiles.ProfileProto.Line;
import com.google.perftools.profiles.ProfileProto.Location;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.perftools.profiles.ProfileProto.ValueType;
import java.util.HashMap;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ProfileCollectorTest {

  @Test
  public void toProto_hasExpectedMetadata() {
    var collector = new ProfileCollector();
    collector.recordSample(ImmutableList.of("a", "b"), 10);
    collector.recordSample(ImmutableList.of("a"), 20);

    Profile profile = collector.toProto();

    List<String> stringTable = profile.getStringTableList();
    assertThat(stringTable).hasSize(7);
    assertThat(stringTable.subList(0, 5))
        .isEqualTo(
            ImmutableList.of(
                "", // empty, required by the schema
                ProfileCollector.SAMPLES,
                ProfileCollector.COUNT,
                ProfileCollector.STORAGE,
                ProfileCollector.BYTES));
    // The records are traversed in a non-deterministic order. Depending on which one comes first,
    // "a" or "b" might be the earlier entry in the string table.
    assertThat(stringTable.subList(5, 7)).containsExactly("a", "b");

    assertThat(profile.getSampleTypeList())
        .containsExactly(
            // ProfileCollector.SAMPLES with units ProfileCollector.COUNT
            ValueType.newBuilder().setType(1).setUnit(2).build(),
            // ProfileCollector.STORAGE with units ProfileCollector.BYTES
            ValueType.newBuilder().setType(3).setUnit(4).build())
        .inOrder();

    assertThat(getSamples(profile))
        .containsExactly(
            // The stack trace is reversed with the leaf is position 0, as per the proto spec.
            new Sample(ImmutableList.of("b", "a"), 1, 10),
            // This was originally 20 but became 10 by subtracting the child.
            new Sample(ImmutableList.of("a"), 1, 10));
  }

  @Test
  public void toProto_aggregatesSamples() {
    var collector = new ProfileCollector();
    collector.recordSample(ImmutableList.of("a", "b", "c"), 10);
    collector.recordSample(ImmutableList.of("a", "b", "d"), 7);
    collector.recordSample(ImmutableList.of("a", "b"), 20);
    collector.recordSample(ImmutableList.of("a"), 25);

    collector.recordSample(ImmutableList.of("a", "b", "d"), 2);
    collector.recordSample(ImmutableList.of("a", "b"), 5);
    collector.recordSample(ImmutableList.of("a"), 10);

    collector.recordSample(ImmutableList.of("a"), 1);

    assertThat(getSamples(collector.toProto()))
        .containsExactly(
            // Only 1 entry. The stack trace is reversed with the leaf in position, as per the proto
            // spec.
            new Sample(ImmutableList.of("c", "b", "a"), 1, 10),
            // 2 samples, bytes = 2 + 7.
            new Sample(ImmutableList.of("d", "b", "a"), 2, 9),
            // 2 samples, bytes = 20 + 5 - (9 + 10) = 6.
            new Sample(ImmutableList.of("b", "a"), 2, 6),
            // 3 samples, bytes = 25 + 10 + 1 - (20 + 5) = 11.
            new Sample(ImmutableList.of("a"), 3, 11));
  }

  private record Sample(ImmutableList<String> stack, int count, int bytes) {}

  /** Converts the {@code profile} message into an easily inspectable list of {@link Sample}s. */
  private static ImmutableList<Sample> getSamples(Profile profile) {
    List<String> strings = profile.getStringTableList();
    var functionNames = new HashMap<Integer, String>();
    for (var function : profile.getFunctionList()) {
      int id = (int) function.getId();
      String previous = functionNames.putIfAbsent(id, strings.get((int) function.getName()));
      assertWithMessage("duplicate function ID %s in %s", id, profile.getFunctionList())
          .that(previous)
          .isNull();
    }
    var locationNames = new HashMap<Integer, String>();
    for (Location location : profile.getLocationList()) {
      int id = (int) location.getId();
      List<Line> lines = location.getLineList();
      assertWithMessage("location with unexpected number of lines: %s", location)
          .that(lines)
          .hasSize(1);
      assertWithMessage("location with id different from function id: %s", location)
          .that(lines.get(0).getFunctionId())
          .isEqualTo(id);
      String previous = locationNames.putIfAbsent(id, functionNames.get(id));
      assertWithMessage("duplicate location ID %s in %s", id, profile.getLocationList())
          .that(previous)
          .isNull();
    }
    assertThat(locationNames).isEqualTo(functionNames);

    var samples = ImmutableList.<Sample>builder();
    for (ProfileProto.Sample sample : profile.getSampleList()) {
      var stack =
          sample.getLocationIdList().stream()
              .map(id -> locationNames.get((int) (long) id))
              .collect(toImmutableList());
      var values = sample.getValueList();
      assertThat(values).hasSize(2);
      samples.add(new Sample(stack, (int) (long) values.get(0), (int) (long) values.get(1)));
    }
    return samples.build();
  }
}
