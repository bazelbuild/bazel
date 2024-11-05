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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link IgnoredSubdirectories}. */
@RunWith(JUnit4.class)
public class IgnoredSubdirectoriesTest {
  private static ImmutableSet<PathFragment> prefixes(String... prefixes) {
    return Arrays.stream(prefixes).map(PathFragment::create).collect(toImmutableSet());
  }

  private static ImmutableList<String> patterns(String... patterns) {
    return ImmutableList.copyOf(patterns);
  }

  @Test
  public void testSimpleUnion() throws Exception {
    IgnoredSubdirectories one = IgnoredSubdirectories.of(prefixes("prefix1"), patterns("pattern1"));
    IgnoredSubdirectories two = IgnoredSubdirectories.of(prefixes("prefix2"), patterns("pattern2"));
    IgnoredSubdirectories union = one.union(two);
    assertThat(union)
        .isEqualTo(
            IgnoredSubdirectories.of(
                prefixes("prefix1", "prefix2"), patterns("pattern1", "pattern2")));
  }

  @Test
  public void testSelfUnionNoop() throws Exception {
    IgnoredSubdirectories ignored = IgnoredSubdirectories.of(prefixes("pre"), patterns("pat"));
    assertThat(ignored.union(ignored)).isEqualTo(ignored);
  }

  @Test
  public void filterPrefixes() throws Exception {
    IgnoredSubdirectories original =
        IgnoredSubdirectories.of(prefixes("foo", "bar", "barbaz", "bar/qux"));
    IgnoredSubdirectories filtered = original.filterForDirectory(PathFragment.create("bar"));
    assertThat(filtered).isEqualTo(IgnoredSubdirectories.of(prefixes("bar", "bar/qux")));
  }

  @Test
  public void filterPatterns() throws Exception {
    IgnoredSubdirectories original =
        IgnoredSubdirectories.of(
            prefixes(),
            patterns(
                "**/sub",
                "foo",
                "bar/*/onesub",
                "bar/qux/**",
                "bar/ba*",
                "bar/not/*/twosub",
                "bar/**/barsub",
                "bar/sub/subsub"));
    IgnoredSubdirectories filtered = original.filterForDirectory(PathFragment.create("bar/sub"));
    assertThat(filtered)
        .isEqualTo(
            IgnoredSubdirectories.of(
                prefixes(), patterns("**/sub", "bar/*/onesub", "bar/**/barsub", "bar/sub/subsub")));
  }

  @Test
  public void filterPatternsForHiddenFiles() throws Exception {
    IgnoredSubdirectories original =
        IgnoredSubdirectories.of(
            prefixes(),
            patterns("not/sub", "*dden/sub", ".hidden/**/sub", ".hi*/*/sub", "*/sub", "**/sub"));
    IgnoredSubdirectories filtered = original.filterForDirectory(PathFragment.create(".hidden"));
    // Glob semantics say that "*dden" is not supposed to match ".hidden".
    // "**" and "*" and ".hi*" should, though.
    assertThat(filtered)
        .isEqualTo(
            IgnoredSubdirectories.of(
                prefixes(), patterns(".hidden/**/sub", ".hi*/*/sub", "*/sub", "**/sub")));
  }
}
