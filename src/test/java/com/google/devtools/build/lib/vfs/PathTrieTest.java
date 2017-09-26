// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PathTrie}. */
@RunWith(JUnit4.class)
public class PathTrieTest {
  @Test
  public void empty() {
    PathTrie<Integer> pathTrie = new PathTrie<>();
    assertThat(pathTrie.get(PathFragment.EMPTY_FRAGMENT)).isNull();
    assertThat(pathTrie.get(PathFragment.create("/x"))).isNull();
    assertThat(pathTrie.get(PathFragment.create("/x/y"))).isNull();
  }

  @Test
  public void simpleBranches() {
    PathTrie<Integer> pathTrie = new PathTrie<>();
    pathTrie.put(PathFragment.create("/a"), 1);
    pathTrie.put(PathFragment.create("/b"), 2);

    assertThat(pathTrie.get(PathFragment.EMPTY_FRAGMENT)).isNull();
    assertThat(pathTrie.get(PathFragment.create("/a"))).isEqualTo(1);
    assertThat(pathTrie.get(PathFragment.create("/a/b"))).isEqualTo(1);
    assertThat(pathTrie.get(PathFragment.create("/a/b/c"))).isEqualTo(1);
    assertThat(pathTrie.get(PathFragment.create("/b"))).isEqualTo(2);
    assertThat(pathTrie.get(PathFragment.create("/b/c"))).isEqualTo(2);
  }

  @Test
  public void nestedDirectories() {
    PathTrie<Integer> pathTrie = new PathTrie<>();
    pathTrie.put(PathFragment.create("/a/b/c"), 3);
    assertThat(pathTrie.get(PathFragment.EMPTY_FRAGMENT)).isNull();
    assertThat(pathTrie.get(PathFragment.create("/a"))).isNull();
    assertThat(pathTrie.get(PathFragment.create("/a/b"))).isNull();
    assertThat(pathTrie.get(PathFragment.create("/a/b/c"))).isEqualTo(3);
    assertThat(pathTrie.get(PathFragment.create("/a/b/c/d"))).isEqualTo(3);

    pathTrie.put(PathFragment.create("/a"), 1);
    assertThat(pathTrie.get(PathFragment.EMPTY_FRAGMENT)).isNull();
    assertThat(pathTrie.get(PathFragment.create("/b"))).isNull();
    assertThat(pathTrie.get(PathFragment.create("/a"))).isEqualTo(1);
    assertThat(pathTrie.get(PathFragment.create("/a/b"))).isEqualTo(1);
    assertThat(pathTrie.get(PathFragment.create("/a/b/c"))).isEqualTo(3);
    assertThat(pathTrie.get(PathFragment.create("/a/b/c/d"))).isEqualTo(3);

    pathTrie.put(PathFragment.ROOT_FRAGMENT, 0);
    assertThat(pathTrie.get(PathFragment.EMPTY_FRAGMENT)).isEqualTo(0);
    assertThat(pathTrie.get(PathFragment.create("/b"))).isEqualTo(0);
    assertThat(pathTrie.get(PathFragment.create("/a"))).isEqualTo(1);
    assertThat(pathTrie.get(PathFragment.create("/a/b"))).isEqualTo(1);
  }

  @Test
  public void unrootedDirectoriesError() {
    PathTrie<Integer> pathTrie = new PathTrie<>();
    assertThrows(IllegalArgumentException.class, () -> pathTrie.put(PathFragment.create("a"), 1));
  }
}
