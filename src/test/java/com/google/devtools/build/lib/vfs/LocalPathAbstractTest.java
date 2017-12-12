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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static java.util.stream.Collectors.toList;

import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.vfs.LocalPath.OsPathPolicy;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Test;

/** Tests for {@link LocalPath}. */
public abstract class LocalPathAbstractTest {

  private OsPathPolicy os;

  @Before
  public void setup() {
    os = getFilePathOs();
  }

  @Test
  public void testEqualsAndHashCode() {
    new EqualsTester()
        .addEqualityGroup(
            create("../relative/path"), create("..").getRelative("relative").getRelative("path"))
        .addEqualityGroup(create("something/else"))
        .addEqualityGroup(create(""), LocalPath.EMPTY)
        .testEquals();
  }

  @Test
  public void testRelativeTo() {
    assertThat(create("").relativeTo(create("")).getPathString()).isEmpty();
    assertThat(create("foo").relativeTo(create("foo")).getPathString()).isEmpty();
    assertThat(create("foo/bar/baz").relativeTo(create("foo")).getPathString())
        .isEqualTo("bar/baz");
    assertThat(create("foo/bar/baz").relativeTo(create("foo/bar")).getPathString())
        .isEqualTo("baz");
    assertThat(create("foo").relativeTo(create("")).getPathString()).isEqualTo("foo");

    // Cannot relativize non-ancestors
    assertThrows(IllegalArgumentException.class, () -> create("foo/bar").relativeTo(create("fo")));

    // Make sure partial directory matches aren't reported
    assertThrows(
        IllegalArgumentException.class, () -> create("foo/bar").relativeTo(create("foo/ba")));
  }

  @Test
  public void testGetRelative() {
    assertThat(create("a").getRelative("b").getPathString()).isEqualTo("a/b");
    assertThat(create("a/b").getRelative("c/d").getPathString()).isEqualTo("a/b/c/d");
    assertThat(create("a").getRelative("").getPathString()).isEqualTo("a");
    assertThat(create("a/b").getRelative("../c").getPathString()).isEqualTo("a/c");
    assertThat(create("a/b").getRelative("..").getPathString()).isEqualTo("a");
  }

  @Test
  public void testEmptyPathToEmptyPath() {
    // compare string forms
    assertThat(create("").getPathString()).isEmpty();
    // compare fragment forms
    assertThat(create("")).isEqualTo(create(""));
  }

  @Test
  public void testSimpleNameToSimpleName() {
    // compare string forms
    assertThat(create("foo").getPathString()).isEqualTo("foo");
    // compare fragment forms
    assertThat(create("foo")).isEqualTo(create("foo"));
  }

  @Test
  public void testSimplePathToSimplePath() {
    // compare string forms
    assertThat(create("foo/bar").getPathString()).isEqualTo("foo/bar");
    // compare fragment forms
    assertThat(create("foo/bar")).isEqualTo(create("foo/bar"));
  }

  @Test
  public void testStripsTrailingSlash() {
    // compare string forms
    assertThat(create("foo/bar/").getPathString()).isEqualTo("foo/bar");
    // compare fragment forms
    assertThat(create("foo/bar/")).isEqualTo(create("foo/bar"));
  }

  @Test
  public void testGetParentDirectory() {
    LocalPath fooBarWiz = create("foo/bar/wiz");
    LocalPath fooBar = create("foo/bar");
    LocalPath foo = create("foo");
    LocalPath empty = create("");
    assertThat(fooBarWiz.getParentDirectory()).isEqualTo(fooBar);
    assertThat(fooBar.getParentDirectory()).isEqualTo(foo);
    assertThat(foo.getParentDirectory()).isEqualTo(empty);
    assertThat(empty.getParentDirectory()).isNull();
  }

  @Test
  public void testBasename() throws Exception {
    assertThat(create("foo/bar").getBaseName()).isEqualTo("bar");
    assertThat(create("foo/").getBaseName()).isEqualTo("foo");
    assertThat(create("foo").getBaseName()).isEqualTo("foo");
    assertThat(create("").getBaseName()).isEmpty();
  }

  @Test
  public void testStartsWith() {
    // (relative path, relative prefix) => true
    assertThat(create("foo/bar").startsWith(create("foo/bar"))).isTrue();
    assertThat(create("foo/bar").startsWith(create("foo"))).isTrue();
    assertThat(create("foot/bar").startsWith(create("foo"))).isFalse();
  }

  @Test
  public void testNormalize() {
    assertThat(create("a/b")).isEqualTo(create("a/b"));
    assertThat(create("a/../../b")).isEqualTo(create("../b"));
    assertThat(create("a/../..")).isEqualTo(create(".."));
    assertThat(create("a/../b")).isEqualTo(create("b"));
    assertThat(create("a/b/../b")).isEqualTo(create("a/b"));
  }

  @Test
  public void testNormalStringsDoNotAllocate() {
    String normal1 = "a/b/hello.txt";
    assertThat(create(normal1).getPathString()).isSameAs(normal1);

    // Sanity check our testing strategy
    String notNormal = "a/../b";
    assertThat(create(notNormal).getPathString()).isNotSameAs(notNormal);
  }

  @Test
  public void testComparableSortOrder() {
    List<LocalPath> list =
        Lists.newArrayList(
            create("zzz"),
            create("ZZZ"),
            create("ABC"),
            create("aBc"),
            create("AbC"),
            create("abc"));
    Collections.sort(list);
    List<String> result = list.stream().map(LocalPath::getPathString).collect(toList());

    if (os.isCaseSensitive()) {
      assertThat(result).containsExactly("ABC", "AbC", "ZZZ", "aBc", "abc", "zzz").inOrder();
    } else {
      // Partial ordering among case-insensitive items guaranteed by Collections.sort stability
      assertThat(result).containsExactly("ABC", "aBc", "AbC", "abc", "zzz", "ZZZ").inOrder();
    }
  }

  protected abstract OsPathPolicy getFilePathOs();

  protected LocalPath create(String path) {
    return LocalPath.createWithOs(path, os);
  }
}
