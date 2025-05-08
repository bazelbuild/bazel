// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ImmutableSetCodecTest {

  @Test
  @SuppressWarnings("SingletonSet")
  public void testSingleton() throws Exception {
    new SerializationTester(Collections.singleton("a"), Collections.singleton(1)).runTests();
  }

  @Test
  @SuppressWarnings("EmptySet")
  public void testEmpty() throws Exception {
    new SerializationTester(Collections.emptySet(), ImmutableSet.of()).runTests();
  }

  @Test
  public void testMultimapValueSet() throws Exception {
    // Tests the serialization of the hidden type, `LinkedHashMultimap.ValueSet`. There's no way to
    // construct instances of this type directly, so instead, constructs a `LinkedHashMultimap`,
    // then extracts and tests its values.
    LinkedHashMultimap<String, Integer> source = LinkedHashMultimap.create();
    source.putAll("a", ImmutableList.of(1, 2, 3));
    source.putAll("b", ImmutableList.of(4, 5, 6));
    source.putAll("c", ImmutableList.of(7, 8, 9));

    Map<String, Collection<Integer>> map = source.asMap();

    ArrayList<Collection<Integer>> subjects = new ArrayList<>();
    for (Map.Entry<String, Collection<Integer>> entry : map.entrySet()) {
      Collection<Integer> valueSet = entry.getValue();
      // Verifies that `valueSet` is of the special hidden `LinkedHashMultimap.ValueSet` type.
      assertThat(valueSet).isInstanceOf(ImmutableSetCodec.MULTIMAP_VALUE_SET_CLASS);
      subjects.add(valueSet);
    }

    new SerializationTester(Iterables.toArray(subjects, Object.class)).runTests();
  }

  @Test
  public void testPowerSetSubset() throws Exception {
    ArrayList<Set<String>> subsets = new ArrayList<>();
    for (Set<String> subset : Sets.powerSet(ImmutableSet.of("a", "b", "c"))) {
      if (subset.isEmpty()) {
        // The empty subset, unfortunately, does not have a stable serialized representation. The
        // first trip serializes it as a set of size 0, and the second trip serializes it as a
        // reference constant.
        continue;
      }
      subsets.add(subset);
    }
    new SerializationTester(Iterables.toArray(subsets, Object.class)).runTests();
  }

  @Test
  public void testSet() throws Exception {
    new SerializationTester(
            ImmutableSet.of(1, 2, 3, 4, 5),
            ImmutableSet.of("abc", "def", "ced"),
            ImmutableSet.of(2.5e2, 3.14159))
        .runTests();
  }
}
