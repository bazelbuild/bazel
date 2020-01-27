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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the unix implementation of {@link Path}. */
@RunWith(JUnit4.class)
public class UnixPathTest extends PathAbstractTest {

  @Test
  public void testEqualsAndHashCodeUnix() {
    new EqualsTester()
        .addEqualityGroup(create("/something/else"))
        .addEqualityGroup(create("/"), create("//////"))
        .testEquals();
  }

  @Test
  public void testRelativeToUnix() {
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
  public void testGetRelativeUnix() {
    assertThat(create("/a").getRelative("b").getPathString()).isEqualTo("/a/b");
    assertThat(create("/a/b").getRelative("c/d").getPathString()).isEqualTo("/a/b/c/d");
    assertThat(create("/c/d").getRelative("/a/b").getPathString()).isEqualTo("/a/b");
    assertThat(create("/a").getRelative("").getPathString()).isEqualTo("/a");
    assertThat(create("/").getRelative("").getPathString()).isEqualTo("/");
    assertThat(create("/a/b").getRelative("../foo").getPathString()).isEqualTo("/a/foo");

    // Make sure any fast path of Path#getRelative(PathFragment) works
    assertThat(create("/a/b").getRelative(PathFragment.create("../foo")).getPathString())
        .isEqualTo("/a/foo");

    // Make sure any fast path of Path#getRelative(PathFragment) works
    assertThat(create("/c/d").getRelative(PathFragment.create("/a/b")).getPathString())
        .isEqualTo("/a/b");

    // Test normalization
    assertThat(create("/a").getRelative(".").getPathString()).isEqualTo("/a");
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
    assertThat(create("/foo/bar/wiz").getParentDirectory()).isEqualTo(create("/foo/bar"));
    assertThat(create("/foo/bar").getParentDirectory()).isEqualTo(create("/foo"));
    assertThat(create("/foo").getParentDirectory()).isEqualTo(create("/"));
    assertThat(create("/").getParentDirectory()).isNull();
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
    Path foobar = create("/foo/bar");

    // (path, prefix) => true
    assertThat(foobar.startsWith(foobar)).isTrue();
    assertThat(foobar.startsWith(create("/"))).isTrue();
    assertThat(foobar.startsWith(create("/foo"))).isTrue();
    assertThat(foobar.startsWith(create("/foo/"))).isTrue();
    assertThat(foobar.startsWith(create("/foo/bar/"))).isTrue(); // Includes trailing slash.

    // (prefix, path) => false
    assertThat(create("/foo").startsWith(foobar)).isFalse();
    assertThat(create("/").startsWith(foobar)).isFalse();

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

  @Test
  public void testParentOfRootIsRootUnix() {
    assertThat(create("/..")).isEqualTo(create("/"));
    assertThat(create("/../../../../../..")).isEqualTo(create("/"));
    assertThat(create("/../../../foo")).isEqualTo(create("/foo"));
  }
}
