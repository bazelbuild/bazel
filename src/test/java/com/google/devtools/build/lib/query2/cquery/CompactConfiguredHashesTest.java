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
package com.google.devtools.build.lib.query2.cquery;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Verifies that scheduling independent hashes preserves the ordered, serial digest. */
@RunWith(JUnit4.class)
public class CompactConfiguredHashesTest {
  private static final class Node extends CompactConfiguredHashes.Node<Node> {
    Node(int ordinal, String name) {
      super(ordinal, name);
      hash = name.getBytes(StandardCharsets.UTF_8);
    }
  }

  // Root before children, shared leaves, wide independent levels and an unavailable input.
  private static List<Node> graph() {
    var nodes = new ArrayList<Node>();
    for (int i = 0; i < 512; i++) {
      nodes.add(new Node(i, "//pkg:t" + i));
    }
    for (int i = 2; i < 216; i++) {
      nodes.get(i).deps = List.of(nodes.get(i + 216), nodes.get(511));
    }
    nodes.get(218).hash = null;
    nodes.get(0).deps = List.copyOf(nodes.subList(2, 216));
    nodes.get(1).deps = List.copyOf(nodes.subList(5, 216));
    return nodes;
  }

  private static void assertSameHashes(List<Node> actual, List<Node> expected) {
    for (int i = 0; i < expected.size(); i++) {
      assertThat(actual.get(i).hash).isEqualTo(expected.get(i).hash);
    }
  }

  @Test
  public void parallelDag_matchesSerialIncludingFailedAndMissingInputs() throws Exception {
    var expected = graph();
    CompactConfiguredHashes.combine(expected, Set.of(expected.get(3)));
    assertThat(expected.get(0).hash).isNull();
    assertThat(expected.get(1).hash).isNotNull();
    for (int workers : List.of(1, 4)) {
      var actual = graph();
      CompactConfiguredHashes.combineParallel(actual, Set.of(actual.get(3)), workers);
      assertSameHashes(actual, expected);
    }
    CompactConfiguredHashes.combineParallel(List.<Node>of(), Set.of(), 4);
  }

  @Test
  public void cycle_usesSerialSccDigestsBeforeMutatingNodes() throws Exception {
    var expected = graph();
    var actual = graph();
    for (List<Node> nodes : List.of(expected, actual)) {
      nodes.get(511).deps = List.of(nodes.get(6));
    }
    CompactConfiguredHashes.combine(expected, Set.of());
    CompactConfiguredHashes.combineParallel(actual, Set.of(), 4);
    assertThat(actual.get(6).hash).isNotNull();
    assertSameHashes(actual, expected);
  }

  @Test
  public void cycleWithDuplicateStableIdentities_failsClosed() {
    var first = new Node(0, "//pkg:same");
    var second = new Node(1, "//pkg:same");
    first.deps = List.of(second);
    second.deps = List.of(first);
    assertThrows(
        UncheckedIOException.class,
        () -> CompactConfiguredHashes.combineParallel(List.of(first, second), Set.of(), 4));
  }
}
