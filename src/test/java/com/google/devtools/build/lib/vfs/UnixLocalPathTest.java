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

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.vfs.LocalPath.OsPathPolicy;
import com.google.devtools.build.lib.vfs.LocalPath.UnixOsPathPolicy;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the unix implementation of {@link LocalPath}. */
@RunWith(JUnit4.class)
public class UnixLocalPathTest extends LocalPathAbstractTest {

  @Test
  public void testEqualsAndHashCodeUnix() {
    new EqualsTester()
        .addEqualityGroup(create("/something/else"))
        .addEqualityGroup(create("/"), create("//////"))
        .testEquals();
  }

  @Test
  public void testRelativeToUnix() {
    // Cannot relativize absolute and non-absolute
    assertThat(create("c/d").getRelative("/a/b").getPathString()).isEqualTo("/a/b");
    assertThat(create("/").relativeTo(create("/")).getPathString()).isEmpty();
    assertThat(create("/foo").relativeTo(create("/foo")).getPathString()).isEmpty();
    assertThat(create("/foo/bar/baz").relativeTo(create("/foo")).getPathString())
        .isEqualTo("bar/baz");
    assertThat(create("/foo/bar/baz").relativeTo(create("/foo/bar")).getPathString())
        .isEqualTo("baz");
    assertThat(create("/foo").relativeTo(create("/")).getPathString()).isEqualTo("foo");
    assertThrows(
        IllegalArgumentException.class, () -> create("/foo/bar/baz").relativeTo(create("foo")));
    assertThrows(
        IllegalArgumentException.class, () -> create("/foo").relativeTo(create("/foo/bar/baz")));
  }

  @Test
  public void testIsAbsoluteUnix() {
    assertThat(create("/absolute/test").isAbsolute()).isTrue();
    assertThat(create("relative/test").isAbsolute()).isFalse();
  }

  @Test
  public void testGetRelativeUnix() {
    assertThat(create("/a").getRelative("b").getPathString()).isEqualTo("/a/b");
    assertThat(create("/").getRelative("").getPathString()).isEqualTo("/");
    assertThat(create("c/d").getRelative("/a/b").getPathString()).isEqualTo("/a/b");
  }

  @Test
  public void testEmptyPathToEmptyPathUnix() {
    // compare string forms
    assertThat(create("/").getPathString()).isEqualTo("/");
    // compare fragment forms
    assertThat(create("/")).isEqualTo(create("/"));
  }

  @Test
  public void testRedundantSlashes() {
    // compare string forms
    assertThat(create("///").getPathString()).isEqualTo("/");
    // compare fragment forms
    assertThat(create("///")).isEqualTo(create("/"));
    // compare string forms
    assertThat(create("/foo///bar").getPathString()).isEqualTo("/foo/bar");
    // compare fragment forms
    assertThat(create("/foo///bar")).isEqualTo(create("/foo/bar"));
    // compare string forms
    assertThat(create("////foo//bar").getPathString()).isEqualTo("/foo/bar");
    // compare fragment forms
    assertThat(create("////foo//bar")).isEqualTo(create("/foo/bar"));
  }

  @Test
  public void testSimpleNameToSimpleNameUnix() {
    // compare string forms
    assertThat(create("/foo").getPathString()).isEqualTo("/foo");
    // compare fragment forms
    assertThat(create("/foo")).isEqualTo(create("/foo"));
  }

  @Test
  public void testSimplePathToSimplePathUnix() {
    // compare string forms
    assertThat(create("/foo/bar").getPathString()).isEqualTo("/foo/bar");
    // compare fragment forms
    assertThat(create("/foo/bar")).isEqualTo(create("/foo/bar"));
  }

  @Test
  public void testGetParentDirectoryUnix() {
    LocalPath fooBarWizAbs = create("/foo/bar/wiz");
    LocalPath fooBarAbs = create("/foo/bar");
    LocalPath fooAbs = create("/foo");
    LocalPath rootAbs = create("/");
    assertThat(fooBarWizAbs.getParentDirectory()).isEqualTo(fooBarAbs);
    assertThat(fooBarAbs.getParentDirectory()).isEqualTo(fooAbs);
    assertThat(fooAbs.getParentDirectory()).isEqualTo(rootAbs);
    assertThat(rootAbs.getParentDirectory()).isNull();
  }

  @Test
  public void testBasenameUnix() throws Exception {
    assertThat(create("/foo/bar").getBaseName()).isEqualTo("bar");
    assertThat(create("/foo/").getBaseName()).isEqualTo("foo");
    assertThat(create("/foo").getBaseName()).isEqualTo("foo");
    assertThat(create("/").getBaseName()).isEmpty();
  }

  @Test
  public void testStartsWithUnix() {
    LocalPath foobar = create("/foo/bar");
    LocalPath foobarRelative = create("foo/bar");

    // (path, prefix) => true
    assertThat(foobar.startsWith(foobar)).isTrue();
    assertThat(foobar.startsWith(create("/"))).isTrue();
    assertThat(foobar.startsWith(create("/foo"))).isTrue();
    assertThat(foobar.startsWith(create("/foo/"))).isTrue();
    assertThat(foobar.startsWith(create("/foo/bar/"))).isTrue(); // Includes trailing slash.

    // (prefix, path) => false
    assertThat(create("/foo").startsWith(foobar)).isFalse();
    assertThat(create("/").startsWith(foobar)).isFalse();

    // (absolute, relative) => false
    assertThat(foobar.startsWith(foobarRelative)).isFalse();
    assertThat(foobarRelative.startsWith(foobar)).isFalse();

    // relative paths start with nothing, absolute paths do not
    assertThat(foobar.startsWith(create(""))).isFalse();

    // (path, sibling) => false
    assertThat(create("/foo/wiz").startsWith(foobar)).isFalse();
    assertThat(foobar.startsWith(create("/foo/wiz"))).isFalse();
  }

  @Test
  public void testNormalizeUnix() {
    assertThat(create("/a/b")).isEqualTo(create("/a/b"));
    assertThat(create("/a/b/")).isEqualTo(create("/a/b"));
    assertThat(create("/a/./b")).isEqualTo(create("/a/b"));
    assertThat(create("/a/../b")).isEqualTo(create("/b"));
    assertThat(create("/..")).isEqualTo(create("/.."));
  }

  @Override
  protected OsPathPolicy getFilePathOs() {
    return new UnixOsPathPolicy();
  }
}
