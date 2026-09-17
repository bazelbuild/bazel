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

package com.google.devtools.build.lib.bazel.repository.decompressor;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link StripPrefixedPath}.
 */
@RunWith(JUnit4.class)
public class StripPrefixedPathTest {
  @Test
  public void testStripPrefix() {
    StripPrefixedPath result = StripPrefixedPath.maybeDeprefix("foo/bar".getBytes(UTF_8), "foo", 0);
    assertThat(PathFragment.create("bar")).isEqualTo(result.getPathFragment());
    assertThat(result.foundPrefix()).isTrue();
    assertThat(result.skip()).isFalse();

    result = StripPrefixedPath.maybeDeprefix("foo".getBytes(UTF_8), "foo", 0);
    assertThat(result.skip()).isTrue();

    result = StripPrefixedPath.maybeDeprefix("bar/baz".getBytes(UTF_8), "foo", 0);
    assertThat(result.foundPrefix()).isFalse();

    result = StripPrefixedPath.maybeDeprefix("foof/bar".getBytes(UTF_8), "foo", 0);
    assertThat(result.foundPrefix()).isFalse();
  }

  @Test
  public void stripComponents() {
    StripPrefixedPath result = StripPrefixedPath.maybeDeprefix("foo/bar".getBytes(UTF_8), "", 1);
    assertThat(PathFragment.create("bar")).isEqualTo(result.getPathFragment());
    assertThat(result.foundPrefix()).isFalse();
    assertThat(result.skip()).isFalse();

    result = StripPrefixedPath.maybeDeprefix("foo".getBytes(UTF_8), "", 1);
    assertThat(result.foundPrefix()).isFalse();
    assertThat(result.skip()).isTrue();
  }

  @Test
  public void stripPrefixAndComponents() {
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> StripPrefixedPath.maybeDeprefix("foo/bar".getBytes(UTF_8), "foo", 1));

    assertThat(e).hasMessageThat().contains("Only one of prefix or strip_components can be set.");
  }

  @Test
  public void testAbsolute() {
    StripPrefixedPath result = StripPrefixedPath.maybeDeprefix("/foo/bar".getBytes(UTF_8), "", 0);
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("foo/bar"));

    result = StripPrefixedPath.maybeDeprefix("///foo/bar/baz".getBytes(UTF_8), "", 0);
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("foo/bar/baz"));

    result = StripPrefixedPath.maybeDeprefix("/foo/bar/baz".getBytes(UTF_8), "/foo", 0);
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("bar/baz"));

    result = StripPrefixedPath.maybeDeprefix("/foo/bar/baz".getBytes(UTF_8), "", 1);
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("bar/baz"));
  }

  @Test
  public void testWindowsAbsolute() {
    if (OS.getCurrent() != OS.WINDOWS) {
      return;
    }
    StripPrefixedPath result = StripPrefixedPath.maybeDeprefix("c:/foo/bar".getBytes(UTF_8), "", 0);
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("foo/bar"));
  }

  @Test
  public void testNormalize() {
    StripPrefixedPath result = StripPrefixedPath.maybeDeprefix("../bar".getBytes(UTF_8), "", 0);
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("../bar"));

    result = StripPrefixedPath.maybeDeprefix("foo/../baz".getBytes(UTF_8), "", 0);
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("baz"));

    result = StripPrefixedPath.maybeDeprefix("foo/../baz".getBytes(UTF_8), "foo", 0);
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("baz"));

    result = StripPrefixedPath.maybeDeprefix("foo/../baz".getBytes(UTF_8), "", 1);
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void testDeprefixSymlink() {
    InMemoryFileSystem fileSystem =
        new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);

    PathFragment relativeNoPrefix =
        StripPrefixedPath.maybeDeprefixSymlink(
            "a/b".getBytes(UTF_8), "", 0, fileSystem.getPath("/usr"));
    // there is no attempt to get absolute path for the relative symlinks target path
    assertThat(relativeNoPrefix).isEqualTo(PathFragment.create("a/b"));

    PathFragment absoluteNoPrefix =
        StripPrefixedPath.maybeDeprefixSymlink(
            "/a/b".getBytes(UTF_8), "", 0, fileSystem.getPath("/usr"));
    assertThat(absoluteNoPrefix).isEqualTo(PathFragment.create("/usr/a/b"));

    PathFragment absolutePrefix =
        StripPrefixedPath.maybeDeprefixSymlink(
            "/root/a/b".getBytes(UTF_8), "root", 0, fileSystem.getPath("/usr"));
    assertThat(absolutePrefix).isEqualTo(PathFragment.create("/usr/a/b"));

    PathFragment absoluteStripComponents =
        StripPrefixedPath.maybeDeprefixSymlink(
            "/root/a/b".getBytes(UTF_8), "", 1, fileSystem.getPath("/usr"));
    assertThat(absoluteStripComponents).isEqualTo(PathFragment.create("/usr/a/b"));

    PathFragment relativePrefix =
        StripPrefixedPath.maybeDeprefixSymlink(
            "root/a/b".getBytes(UTF_8), "root", 0, fileSystem.getPath("/usr"));
    // Only absolute paths or paths relative to extraction root are deprefixed.
    assertThat(relativePrefix).isEqualTo(PathFragment.create("root/a/b"));

    PathFragment relativeStripComponents =
        StripPrefixedPath.maybeDeprefixSymlink(
            "root/a/b".getBytes(UTF_8), "", 1, fileSystem.getPath("/usr"));
    assertThat(relativeStripComponents).isEqualTo(PathFragment.create("root/a/b"));

    PathFragment forceDeprefixRelativePrefix =
        StripPrefixedPath.maybeDeprefixSymlink(
            "root/a/b".getBytes(UTF_8), "root", 0, fileSystem.getPath("/usr"), true);
    // Forced deprefixing into root relative path.
    assertThat(forceDeprefixRelativePrefix).isEqualTo(PathFragment.create("/usr/a/b"));

    PathFragment forceDeprefixRelativeComponents =
        StripPrefixedPath.maybeDeprefixSymlink(
            "root/a/b".getBytes(UTF_8), "", 1, fileSystem.getPath("/usr"), true);
    assertThat(forceDeprefixRelativeComponents).isEqualTo(PathFragment.create("/usr/a/b"));
  }
}
