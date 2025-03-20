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
package com.google.devtools.build.lib.remote.util;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ConcurrentPathTrieTest {

  @Test
  public void testInvalidPaths() {
    var trie = new ConcurrentPathTrie();

    assertThrows(IllegalArgumentException.class, () -> trie.contains(PathFragment.create("")));
    assertThrows(IllegalArgumentException.class, () -> trie.contains(PathFragment.create("/foo")));

    assertThrows(IllegalArgumentException.class, () -> trie.add(PathFragment.create("")));
    assertThrows(IllegalArgumentException.class, () -> trie.add(PathFragment.create("/foo")));

    assertThrows(IllegalArgumentException.class, () -> trie.addPrefix(PathFragment.create("")));
    assertThrows(IllegalArgumentException.class, () -> trie.addPrefix(PathFragment.create("/foo")));
  }

  @Test
  public void testEmpty() {
    var trie = new ConcurrentPathTrie();

    assertThat(trie.contains(PathFragment.create("foo"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isFalse();
  }

  @Test
  public void testSinglePath() {
    var trie = new ConcurrentPathTrie();
    trie.add(PathFragment.create("foo/bar"));

    assertThat(trie.contains(PathFragment.create("foo"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz"))).isFalse();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }

  @Test
  public void testNonOverlappingPath() {
    var trie = new ConcurrentPathTrie();
    trie.add(PathFragment.create("foo/bar"));
    trie.add(PathFragment.create("quux/xyzzy"));

    assertThat(trie.contains(PathFragment.create("foo"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz"))).isFalse();
    assertThat(trie.contains(PathFragment.create("quux"))).isFalse();
    assertThat(trie.contains(PathFragment.create("quux/xyzzy"))).isTrue();
    assertThat(trie.contains(PathFragment.create("quux/thud"))).isFalse();
    assertThat(trie.contains(PathFragment.create("quux/xyzzy/thud"))).isFalse();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }

  @Test
  public void testOverlappingParentPath() {
    var trie = new ConcurrentPathTrie();
    trie.add(PathFragment.create("foo"));
    trie.add(PathFragment.create("foo/bar"));

    assertThat(trie.contains(PathFragment.create("foo"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/baz/bar"))).isFalse();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }

  @Test
  public void testOverlappingSiblingPath() {
    var trie = new ConcurrentPathTrie();
    trie.add(PathFragment.create("foo/bar"));
    trie.add(PathFragment.create("foo/baz"));

    assertThat(trie.contains(PathFragment.create("foo"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/baz/bar"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/quux"))).isFalse();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }

  @Test
  public void testSinglePrefix() {
    var trie = new ConcurrentPathTrie();
    trie.addPrefix(PathFragment.create("foo/bar"));

    assertThat(trie.contains(PathFragment.create("foo"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz/quux"))).isTrue();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }

  @Test
  public void testNonOverlappingPrefix() {
    var trie = new ConcurrentPathTrie();
    trie.addPrefix(PathFragment.create("foo/bar"));
    trie.addPrefix(PathFragment.create("quux/xyzzy"));

    assertThat(trie.contains(PathFragment.create("foo"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("quux"))).isFalse();
    assertThat(trie.contains(PathFragment.create("quux/xyzzy"))).isTrue();
    assertThat(trie.contains(PathFragment.create("quux/thud"))).isFalse();
    assertThat(trie.contains(PathFragment.create("quux/xyzzy/thud"))).isTrue();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }

  @Test
  public void testOverlappingParentPrefix() {
    var trie = new ConcurrentPathTrie();
    trie.addPrefix(PathFragment.create("foo/bar"));
    trie.addPrefix(PathFragment.create("foo/baz"));

    assertThat(trie.contains(PathFragment.create("foo"))).isFalse();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/quux"))).isFalse();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }

  @Test
  public void testOverlappingSiblingPrefix() {
    var trie = new ConcurrentPathTrie();
    trie.addPrefix(PathFragment.create("foo"));
    trie.addPrefix(PathFragment.create("foo/bar"));

    assertThat(trie.contains(PathFragment.create("foo"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }

  @Test
  public void testOverlappingParentPrefixAndPath() {
    var trie = new ConcurrentPathTrie();
    trie.addPrefix(PathFragment.create("foo"));
    trie.add(PathFragment.create("foo/bar"));

    assertThat(trie.contains(PathFragment.create("foo"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }

  @Test
  public void testOverlappingSiblingPrefixAndPath() {
    var trie = new ConcurrentPathTrie();
    trie.add(PathFragment.create("foo"));
    trie.addPrefix(PathFragment.create("foo"));

    assertThat(trie.contains(PathFragment.create("foo"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("foo/bar/baz"))).isTrue();
    assertThat(trie.contains(PathFragment.create("bar"))).isFalse();
  }
}
