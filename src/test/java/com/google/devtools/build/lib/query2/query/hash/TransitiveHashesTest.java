// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.hash;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class TransitiveHashesTest {

  /** A mutable description of a graph: label -> (direct content, deps). */
  private final Map<String, String> content = new LinkedHashMap<>();

  private final Map<String, List<String>> deps = new LinkedHashMap<>();

  private void node(String name, String directContent, String... depNames) {
    content.put(name, directContent);
    deps.put(name, Arrays.asList(depNames));
  }

  private static Label label(String name) {
    return Label.parseCanonicalUnchecked("//pkg:" + name);
  }

  private Map<Label, byte[]> combine(Map<Label, DirectTargetHashFunction.Value> graph) {
    return TransitiveHashes.combine(graph);
  }

  private Map<Label, DirectTargetHashFunction.Value> graph(
      Map<Label, DirectTargetHashFunction.Value> into) {
    for (var entry : content.entrySet()) {
      into.put(
          label(entry.getKey()),
          new DirectTargetHashFunction.Value(
              entry.getValue().getBytes(UTF_8),
              deps.get(entry.getKey()).stream()
                  .map(TransitiveHashesTest::label)
                  .sorted()
                  .collect(ImmutableList.toImmutableList())));
    }
    return into;
  }

  private Map<String, String> hashes() {
    Map<Label, byte[]> combined = combine(graph(new HashMap<>()));
    Map<String, String> result = new HashMap<>();
    for (String name : content.keySet()) {
      byte[] hash = combined.get(label(name));
      result.put(name, hash == null ? null : Arrays.toString(hash));
    }
    return result;
  }

  @Test
  public void chain_changeInDepPropagatesToDependentsOnly() {
    node("a", "a", "b");
    node("b", "b", "c");
    node("c", "c");
    node("unrelated", "u");
    Map<String, String> before = hashes();

    node("c", "c2");
    Map<String, String> afterLeafChange = hashes();
    assertThat(afterLeafChange.get("a")).isNotEqualTo(before.get("a"));
    assertThat(afterLeafChange.get("b")).isNotEqualTo(before.get("b"));
    assertThat(afterLeafChange.get("c")).isNotEqualTo(before.get("c"));
    assertThat(afterLeafChange.get("unrelated")).isEqualTo(before.get("unrelated"));

    node("c", "c");
    node("a", "a2", "b");
    Map<String, String> afterRootChange = hashes();
    assertThat(afterRootChange.get("a")).isNotEqualTo(before.get("a"));
    assertThat(afterRootChange.get("b")).isEqualTo(before.get("b"));
    assertThat(afterRootChange.get("c")).isEqualTo(before.get("c"));
  }

  @Test
  public void hashesDoNotDependOnIterationOrder() {
    node("a", "a", "b", "c");
    node("b", "b", "c");
    node("c", "c", "a"); // A cycle a -> c -> a, plus b.
    node("d", "d", "a");
    Map<Label, byte[]> forward = combine(graph(new LinkedHashMap<>()));

    Map<String, String> reversedContent = new LinkedHashMap<>();
    ImmutableList.copyOf(content.keySet())
        .reverse()
        .forEach(name -> reversedContent.put(name, content.get(name)));
    content.clear();
    content.putAll(reversedContent);
    Map<Label, byte[]> reversed = combine(graph(new LinkedHashMap<>()));

    assertThat(reversed.keySet()).isEqualTo(forward.keySet());
    for (Label label : forward.keySet()) {
      assertThat(reversed.get(label)).isEqualTo(forward.get(label));
    }
  }

  @Test
  public void cycle_isHashedAndChangesPropagateThroughIt() {
    node("a", "a", "b");
    node("b", "b", "a");
    node("user", "user", "a");
    node("leaf", "leaf");
    Map<String, String> before = hashes();
    assertThat(before.get("a")).isNotNull();
    assertThat(before.get("b")).isNotNull();
    // Members of the same cycle still get distinct hashes.
    assertThat(before.get("a")).isNotEqualTo(before.get("b"));

    node("b", "b-changed", "a");
    Map<String, String> after = hashes();
    assertThat(after.get("a")).isNotEqualTo(before.get("a"));
    assertThat(after.get("b")).isNotEqualTo(before.get("b"));
    assertThat(after.get("user")).isNotEqualTo(before.get("user"));
    assertThat(after.get("leaf")).isEqualTo(before.get("leaf"));
  }

  @Test
  public void cycle_depOutsideCycleChanging_changesAllMembers() {
    node("a", "a", "b");
    node("b", "b", "a", "leaf");
    node("leaf", "leaf");
    Map<String, String> before = hashes();

    node("leaf", "leaf2");
    Map<String, String> after = hashes();
    assertThat(after.get("a")).isNotEqualTo(before.get("a"));
    assertThat(after.get("b")).isNotEqualTo(before.get("b"));
  }

  @Test
  public void missingDep_isHashedDeterministically() {
    node("a", "a", "missing");
    node("plain", "a");
    Map<String, String> first = hashes();
    Map<String, String> second = hashes();
    assertThat(first.get("a")).isNotNull();
    assertThat(first.get("a")).isEqualTo(second.get("a"));
    assertThat(first.get("a")).isNotEqualTo(first.get("plain"));
  }

  @Test
  public void deepChain_doesNotOverflowStack() {
    int depth = 200_000;
    for (int i = 0; i < depth; i++) {
      if (i + 1 < depth) {
        node("n" + i, "n" + i, "n" + (i + 1));
      } else {
        node("n" + i, "n" + i);
      }
    }
    assertThat(hashes().get("n0")).isNotNull();
  }
}
