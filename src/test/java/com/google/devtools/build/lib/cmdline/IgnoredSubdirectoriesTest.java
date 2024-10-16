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

package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class IgnoredSubdirectoriesTest {
  private static PathFragment path(String path) {
    return PathFragment.create(path);
  }
  
  private static ImmutableSet<PathFragment> prefixes(String... prefixes) {
    return Arrays.stream(prefixes)
        .map(PathFragment::create)
        .collect(ImmutableSet.toImmutableSet());
  }

  private static ImmutableList<String> patterns(String... patterns) {
    return ImmutableList.copyOf(patterns);
  }
  @Test
  public void testFilterPrefixes() {
    IgnoredSubdirectories original = IgnoredSubdirectories.of(
        prefixes("a/b", "a/b/c", "a/d", "a/d/e", "a/d2"),
        patterns());

    IgnoredSubdirectories filtered = original.filterForDirectory(path("a/d"));
    assertThat(filtered).isEqualTo(
        IgnoredSubdirectories.of(
            prefixes("a/d", "a/d/e"),
            patterns()));
  }

  @Test
  public void testFilterPatterns() {
    IgnoredSubdirectories original = IgnoredSubdirectories.of(
        prefixes(),
        patterns("**/dir", "a*/d", "a/d*", "a/d/**/e", "a/d*x/**", "b/**"));

    IgnoredSubdirectories filtered = original.filterForDirectory(path("a/d"));
    assertThat(filtered).isEqualTo(
        IgnoredSubdirectories.of(
            prefixes(),
            patterns("**/dir", "a*/d", "a/d*", "a/d/**/e")));
  }

  @Test
  public void testMatchPatterns() {
    IgnoredSubdirectories cut = IgnoredSubdirectories.of(
        prefixes(),
        patterns("**/node_modules", "a/d*/bar", "b/**/foo"));

    assertThat(cut.matchingEntry(path("foo/bar/node_modules/nodesub"))).isNotNull();
    assertThat(cut.matchingEntry(path("a/d/bar"))).isNotNull();
    assertThat(cut.matchingEntry(path("a/d/bar/sub"))).isNotNull();
    assertThat(cut.matchingEntry(path("a/ddd/bar"))).isNotNull();
    assertThat(cut.matchingEntry(path("a/ddd/bar/sub"))).isNotNull();
    assertThat(cut.matchingEntry(path("b/foo"))).isNotNull();
    assertThat(cut.matchingEntry(path("b/foo/sub"))).isNotNull();
    assertThat(cut.matchingEntry(path("b/b/foo"))).isNotNull();
    assertThat(cut.matchingEntry(path("b/b/foo/sub"))).isNotNull();
    assertThat(cut.matchingEntry(path("a/e/bar"))).isNull();
    assertThat(cut.matchingEntry(path("a/foo"))).isNull();
    assertThat(cut.matchingEntry(path("a/a/foo"))).isNull();
  }

  @Test
  public void testPrefixing() {
    IgnoredSubdirectories cut = IgnoredSubdirectories.of(
        prefixes("x", "y/z"),
        patterns("**", "a/b", "a*/b", "c/**/d"));

    assertThat(cut.withPrefix(path("foo/bar"))).isEqualTo(
        IgnoredSubdirectories.of(
            prefixes("foo/bar/x", "foo/bar/y/z"),
            patterns("foo/bar/**", "foo/bar/a/b", "foo/bar/a*/b", "foo/bar/c/**/d")));
  }

  @Test
  public void testDotfiles() {
    IgnoredSubdirectories cut = IgnoredSubdirectories.of(
        prefixes(".a", "dir/.a"),
        patterns(".b/**", "**/a", "*-c/d", ".b*/e"));

    // Exact matches on prefixes work
    assertThat(cut.matchingEntry(path(".a"))).isNotNull();
    assertThat(cut.matchingEntry(path(".a/sub"))).isNotNull();
    assertThat(cut.matchingEntry(path(".b/e/sub"))).isNotNull();
    assertThat(cut.matchingEntry(path("dir/.a/sub"))).isNotNull();

    // ** can match a dotfile
    assertThat(cut.matchingEntry(path(".b/sub"))).isNotNull();
    assertThat(cut.matchingEntry(path(".foo/a"))).isNotNull();

    // * doesn't match a leading dot, but matches subsequent characters
    assertThat(cut.matchingEntry(path(".c-c/d"))).isNull();
    assertThat(cut.matchingEntry(path(".c-c/d/sub"))).isNull();
    assertThat(cut.matchingEntry(path(".bb/e"))).isNotNull();

    IgnoredSubdirectories filteredCc = cut.filterForDirectory(path(".c-c"));
    assertThat(filteredCc).isEqualTo(IgnoredSubdirectories.of(
        prefixes(),
        patterns("**/a")));

    IgnoredSubdirectories filteredB = cut.filterForDirectory(path(".b"));
    assertThat(filteredB).isEqualTo(IgnoredSubdirectories.of(
        prefixes(),
        patterns(".b/**", "**/a", ".b*/e")));
  }
}
