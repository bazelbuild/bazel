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

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.util.Pair;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link RepositoryOptions}. */
@RunWith(JUnit4.class)
public class RepositoryResolvedEventTest {

  @Test
  public void testCompareReplace() {
    Pair<Map<String, Object>, List<String>> result =
        RepositoryResolvedEvent.compare(
            ImmutableMap.of("foo", "orig"),
            ImmutableMap.of("foo", "changed"));
    assertThat(result.getFirst()).containsExactly("foo", "changed");
    assertThat(result.getSecond()).isEmpty();
  }

  @Test
  public void testCompareDrop() {
    Pair<Map<String, Object>, List<String>> result =
        RepositoryResolvedEvent.compare(ImmutableMap.of("foo", "orig"), ImmutableMap.of());
    assertThat(result.getFirst()).isEmpty();
    assertThat(result.getSecond()).containsExactly("foo");
  }

  @Test
  public void testCompareAdd() {
    Pair<Map<String, Object>, List<String>> result =
        RepositoryResolvedEvent.compare(
            ImmutableMap.<String, Object>of(),
            ImmutableMap.of("foo", "new"));
    assertThat(result.getFirst()).containsExactly("foo", "new");
    assertThat(result.getSecond()).isEmpty();
  }

  @Test
  public void testRepresentModifications() {
    assertThat(
            RepositoryResolvedEvent.representModifications(
                ImmutableMap.<String, Object>of("a", "b", "c", "d")))
        .isEqualTo("a = \"b\", c = \"d\"");
    assertThat(
            RepositoryResolvedEvent.representModifications(
                ImmutableMap.<String, Object>of("a", 1, "b", 2)))
        .isEqualTo("a = 1, b = 2");
    assertThat(
            RepositoryResolvedEvent.representModifications(
                ImmutableMap.<String, Object>of("a", ImmutableList.<Integer>of(1, 2, 3))))
        .isEqualTo("a = [1, 2, 3]");
  }
}
