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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;
import static java.util.stream.Collectors.toList;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Test;

/** Tests for {@link Path}. */
public abstract class PathAbstractTest {

  private FileSystem fileSystem;
  private boolean isCaseSensitive;

  @Before
  public void setup() {
    fileSystem = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    isCaseSensitive = OsPathPolicy.getFilePathOs().isCaseSensitive();
  }

  @Test
  public void testStripsTrailingSlash() {
    // compare string forms
    assertThat(create("/foo/bar/").getPathString()).isEqualTo("/foo/bar");
    // compare fragment forms
    assertThat(create("/foo/bar/")).isEqualTo(create("/foo/bar"));
  }

  @Test
  public void testBasename() throws Exception {
    assertThat(create("/foo/bar").getBaseName()).isEqualTo("bar");
    assertThat(create("/foo/").getBaseName()).isEqualTo("foo");
    assertThat(create("/foo").getBaseName()).isEqualTo("foo");
    assertThat(create("/").getBaseName()).isEmpty();
  }

  @Test
  public void testNormalStringsDoNotAllocate() {
    String normal1 = "/a/b/hello.txt";
    assertThat(create(normal1).getPathString()).isSameInstanceAs(normal1);

    // Check our testing strategy
    String notNormal = "/a/../b";
    assertThat(create(notNormal).getPathString()).isNotSameInstanceAs(notNormal);
  }

  @Test
  public void testComparableSortOrder() {
    List<Path> list =
        Lists.newArrayList(
            create("/zzz"),
            create("/ZZZ"),
            create("/ABC"),
            create("/aBc"),
            create("/AbC"),
            create("/abc"));
    Collections.sort(list);
    List<String> result = list.stream().map(Path::getPathString).collect(toList());

    if (isCaseSensitive) {
      assertThat(result).containsExactly("/ABC", "/AbC", "/ZZZ", "/aBc", "/abc", "/zzz").inOrder();
    } else {
      // Partial ordering among case-insensitive items guaranteed by Collections.sort stability
      assertThat(result).containsExactly("/ABC", "/aBc", "/AbC", "/abc", "/zzz", "/ZZZ").inOrder();
    }
  }

  protected Path create(String path) {
    return Path.create(path, fileSystem);
  }
}
