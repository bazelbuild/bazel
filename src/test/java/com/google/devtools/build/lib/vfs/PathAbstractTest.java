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
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
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

  @Test
  public void testSerialization() throws Exception {
    FileSystem oldFileSystem = Path.getFileSystemForSerialization();
    try {
      Path.setFileSystemForSerialization(fileSystem);
      Path root = fileSystem.getPath("/");
      Path p1 = fileSystem.getPath("/foo");
      Path p2 = fileSystem.getPath("/foo/bar");

      ByteArrayOutputStream bos = new ByteArrayOutputStream();
      ObjectOutputStream oos = new ObjectOutputStream(bos);

      oos.writeObject(root);
      oos.writeObject(p1);
      oos.writeObject(p2);

      ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
      ObjectInputStream ois = new ObjectInputStream(bis);

      Path dsRoot = (Path) ois.readObject();
      Path dsP1 = (Path) ois.readObject();
      Path dsP2 = (Path) ois.readObject();

      new EqualsTester()
          .addEqualityGroup(root, dsRoot)
          .addEqualityGroup(p1, dsP1)
          .addEqualityGroup(p2, dsP2)
          .testEquals();

      assertThat(p2.startsWith(p1)).isTrue();
      assertThat(p2.startsWith(dsP1)).isTrue();
      assertThat(dsP2.startsWith(p1)).isTrue();
      assertThat(dsP2.startsWith(dsP1)).isTrue();

      // Regression test for a very specific bug in compareTo involving our incorrect usage of
      // reference equality rather than logical equality.
      String relativePathStringA = "child/grandchildA";
      String relativePathStringB = "child/grandchildB";
      assertThat(
              p1.getRelative(relativePathStringA).compareTo(dsP1.getRelative(relativePathStringB)))
          .isEqualTo(
              p1.getRelative(relativePathStringA).compareTo(p1.getRelative(relativePathStringB)));
    } finally {
      Path.setFileSystemForSerialization(oldFileSystem);
    }
  }

  protected Path create(String path) {
    return Path.create(path, fileSystem);
  }
}
