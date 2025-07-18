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
package com.google.devtools.build.lib.collect;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.collect.PathFragmentPrefixTrie.PathFragmentAlreadyAddedException;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PathFragmentPrefixTrie}. */
@RunWith(JUnit4.class)
public class PathFragmentPrefixTrieTest {

  @Test
  public void testEmpty() {
    PathFragmentPrefixTrie trie = new PathFragmentPrefixTrie();

    assertThat(trie.includes(PathFragment.create("a"))).isFalse();
    assertThat(trie.includes(PathFragment.create("a/b"))).isFalse();
  }

  @Test
  public void testEmptyPathFragment_isDisallowedForPut() {
    PathFragmentPrefixTrie trie = new PathFragmentPrefixTrie();

    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class, () -> trie.put(PathFragment.EMPTY_FRAGMENT, true));
    assertThat(e).hasMessageThat().contains("path fragment cannot be the empty fragment.");
  }

  @Test
  public void testSimpleInclusions() throws Exception {
    PathFragmentPrefixTrie trie = new PathFragmentPrefixTrie();

    trie.put(PathFragment.create("a"), true);
    trie.put(PathFragment.create("b"), true);

    assertThat(trie.includes(PathFragment.EMPTY_FRAGMENT)).isFalse();
    assertThat(trie.includes(PathFragment.create("a"))).isTrue();
    assertThat(trie.includes(PathFragment.create("a/b"))).isTrue();
    assertThat(trie.includes(PathFragment.create("a/b/c"))).isTrue();
    assertThat(trie.includes(PathFragment.create("b"))).isTrue();
    assertThat(trie.includes(PathFragment.create("c"))).isFalse();
  }

  @Test
  public void testSimpleExclusions() throws Exception {
    PathFragmentPrefixTrie trie = new PathFragmentPrefixTrie();

    trie.put(PathFragment.create("a"), true);
    trie.put(PathFragment.create("a/b"), false);
    trie.put(PathFragment.create("a/b/c"), true);

    assertThat(trie.includes(PathFragment.create("a"))).isTrue();
    assertThat(trie.includes(PathFragment.create("a/b"))).isFalse();
    assertThat(trie.includes(PathFragment.create("a/b/c"))).isTrue();
    assertThat(trie.includes(PathFragment.create("a/b/d"))).isFalse();
  }

  @Test
  public void testAncestors() throws Exception {
    PathFragmentPrefixTrie trie = new PathFragmentPrefixTrie();

    trie.put(PathFragment.create("a/b/c"), true);
    assertThat(trie.includes(PathFragment.create("a"))).isFalse();
    assertThat(trie.includes(PathFragment.create("a/b"))).isFalse();
    assertThat(trie.includes(PathFragment.create("a/b/c"))).isTrue(); // toggled explicitly
    assertThat(trie.includes(PathFragment.create("a/b/c/d"))).isTrue(); // toggled automatically

    trie.put(PathFragment.create("a/b/c/d"), false);
    assertThat(trie.includes(PathFragment.create("a"))).isFalse();
    assertThat(trie.includes(PathFragment.create("a/b"))).isFalse();
    assertThat(trie.includes(PathFragment.create("a/b/c"))).isTrue();
    assertThat(trie.includes(PathFragment.create("a/b/c/d"))).isFalse(); // toggled explicitly

    trie.put(PathFragment.create("a"), true);

    assertThat(trie.includes(PathFragment.create("a"))).isTrue(); // toggled explicitly
    assertThat(trie.includes(PathFragment.create("a/b"))).isTrue(); // toggled automatically
    assertThat(trie.includes(PathFragment.create("a/b/c"))).isTrue();
    assertThat(trie.includes(PathFragment.create("a/b/c/d"))).isFalse();

    trie.put(PathFragment.create("a/b"), false);

    assertThat(trie.includes(PathFragment.create("a"))).isTrue();
    assertThat(trie.includes(PathFragment.create("a/b"))).isFalse(); // toggled explicitly
    assertThat(trie.includes(PathFragment.create("a/b/c"))).isTrue();
    assertThat(trie.includes(PathFragment.create("a/b/c/d"))).isFalse();
  }

  @Test
  public void testSamePathFragmentIncludedAndExcluded_isDisallowed() throws Exception {
    PathFragmentPrefixTrie trie = new PathFragmentPrefixTrie();

    trie.put(PathFragment.create("a/b/c"), false);
    PathFragmentAlreadyAddedException e =
        assertThrows(
            PathFragmentAlreadyAddedException.class,
            () -> trie.put(PathFragment.create("a/b/c"), true));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "a/b/c has already been explicitly marked as excluded. Current state: [included:"
                + " [], excluded: [a/b/c]]");

    trie.put(PathFragment.create("a/b"), true);

    e =
        assertThrows(
            PathFragmentAlreadyAddedException.class,
            () -> trie.put(PathFragment.create("a/b"), false));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "a/b has already been explicitly marked as included. Current state: [included:"
                + " [a/b], excluded: [a/b/c]]");
  }

  @Test
  public void testStringRepr() throws Exception {
    PathFragmentPrefixTrie trie = new PathFragmentPrefixTrie();

    trie.put(PathFragment.create("a"), true);
    trie.put(PathFragment.create("a/b"), false);
    trie.put(PathFragment.create("a/b/c"), true);
    trie.put(PathFragment.create("a/b/d"), false);
    trie.put(PathFragment.create("e"), true);

    assertThat(trie.toString()).isEqualTo("[included: [a, a/b/c, e], excluded: [a/b, a/b/d]]");
  }

}
