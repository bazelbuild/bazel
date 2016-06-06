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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Tests {@link UnixGlob} recursive globs.
 */
@RunWith(JUnit4.class)
public class RecursiveGlobTest {

  private Path tmpPath;
  private FileSystem fileSystem;
  
  @Before
  public final void initializeFileSystem() throws Exception  {
    fileSystem = new InMemoryFileSystem(BlazeClock.instance());
    tmpPath = fileSystem.getPath("/rglobtmp");
    for (String dir : ImmutableList.of("foo/bar/wiz",
                         "foo/baz/wiz",
                         "foo/baz/quip/wiz",
                         "food/baz/wiz",
                         "fool/baz/wiz")) {
      FileSystemUtils.createDirectoryAndParents(tmpPath.getRelative(dir));
    }
    FileSystemUtils.createEmptyFile(tmpPath.getRelative("foo/bar/wiz/file"));
  }

  @Test
  public void testDoubleStar() throws Exception {
    assertGlobMatches("**", ".", "foo", "foo/bar", "foo/bar/wiz", "foo/baz", "foo/baz/quip",
                      "foo/baz/quip/wiz", "foo/baz/wiz", "foo/bar/wiz/file", "food", "food/baz",
                      "food/baz/wiz", "fool", "fool/baz", "fool/baz/wiz");
  }

  @Test
  public void testDoubleDoubleStar() throws Exception {
    assertGlobMatches("**/**", ".", "foo", "foo/bar", "foo/bar/wiz", "foo/baz", "foo/baz/quip",
                      "foo/baz/quip/wiz", "foo/baz/wiz", "foo/bar/wiz/file", "food", "food/baz",
                      "food/baz/wiz", "fool", "fool/baz", "fool/baz/wiz");
  }

  @Test
  public void testDirectoryWithDoubleStar() throws Exception {
    assertGlobMatches("foo/**", "foo", "foo/bar", "foo/bar/wiz", "foo/baz", "foo/baz/quip",
                      "foo/baz/quip/wiz", "foo/baz/wiz", "foo/bar/wiz/file");
  }

  @Test
  public void testIllegalPatterns() throws Exception {
    for (String prefix : Lists.newArrayList("", "*/", "**/", "ba/")) {
      String suffix = ("/" + prefix).substring(0, prefix.length());
      for (String pattern : Lists.newArrayList("**fo", "fo**", "**fo**", "fo**fo", "fo**fo**fo")) {
        assertIllegalWildcard(prefix + pattern);
        assertIllegalWildcard(pattern + suffix);
      }
    }
  }

  @Test
  public void testDoubleStarPatternWithNamedChild() throws Exception {
    assertGlobMatches("**/bar", "foo/bar");
  }

  @Test
  public void testDoubleStarPatternWithChildGlob() throws Exception {
    assertGlobMatches("**/ba*",
        "foo/bar", "foo/baz", "food/baz", "fool/baz");
  }

  @Test
  public void testDoubleStarAsChildGlob() throws Exception {
    assertGlobMatches("foo/**/wiz", "foo/bar/wiz", "foo/baz/quip/wiz", "foo/baz/wiz");
  }

  @Test
  public void testDoubleStarUnderNonexistentDirectory() throws Exception {
    assertGlobMatches("not-there/**" /* => nothing */);
  }

  @Test
  public void testDoubleStarGlobWithNonExistentBase() throws Exception {
    Collection<Path> globResult = UnixGlob.forPath(fileSystem.getPath("/does/not/exist"))
        .addPattern("**")
        .globInterruptible();
    assertThat(globResult).isEmpty();
  }

  @Test
  public void testDoubleStarUnderFile() throws Exception {
    assertGlobMatches("foo/bar/wiz/file/**" /* => nothing */);
  }

  private void assertGlobMatches(String pattern, String... expecteds)
      throws Exception {
    assertThat(
        new UnixGlob.Builder(tmpPath)
            .addPatterns(pattern)
            .globInterruptible())
    .containsExactlyElementsIn(resolvePaths(expecteds));
  }

  private Set<Path> resolvePaths(String... relativePaths) {
    Set<Path> expectedFiles = new HashSet<>();
    for (String expected : relativePaths) {
      Path file = expected.equals(".")
          ? tmpPath
          : tmpPath.getRelative(expected);
      expectedFiles.add(file);
    }
    return expectedFiles;
  }

  /**
   * Tests that a recursive glob returns files in sorted order.
   */
  @Test
  public void testGlobEntriesAreSorted() throws Exception {
    List<Path> globResult = new UnixGlob.Builder(tmpPath)
        .addPattern("**")
        .setExcludeDirectories(false)
        .globInterruptible();

    assertThat(Ordering.natural().sortedCopy(globResult)).containsExactlyElementsIn(globResult)
        .inOrder();
  }

  private void assertIllegalWildcard(String pattern)
      throws Exception {
    try {
      new UnixGlob.Builder(tmpPath)
          .addPattern(pattern)
          .globInterruptible();
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e.getMessage()).containsMatch("recursive wildcard must be its own segment");
    }
  }

}
