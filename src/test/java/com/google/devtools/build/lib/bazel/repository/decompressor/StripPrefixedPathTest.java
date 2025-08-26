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

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link StripPrefixedPath}.
 */
@RunWith(JUnit4.class)
public class StripPrefixedPathTest {
  @Test
  public void testStrip() {
    StripPrefixedPath result =
        StripPrefixedPath.maybeDeprefix("foo/bar".getBytes(UTF_8), Optional.of("foo"));
    assertThat(PathFragment.create("bar")).isEqualTo(result.getPathFragment());
    assertThat(result.foundPrefix()).isTrue();
    assertThat(result.skip()).isFalse();

    result = StripPrefixedPath.maybeDeprefix("foo".getBytes(UTF_8), Optional.of("foo"));
    assertThat(result.skip()).isTrue();

    result = StripPrefixedPath.maybeDeprefix("bar/baz".getBytes(UTF_8), Optional.of("foo"));
    assertThat(result.foundPrefix()).isFalse();

    result = StripPrefixedPath.maybeDeprefix("foof/bar".getBytes(UTF_8), Optional.of("foo"));
    assertThat(result.foundPrefix()).isFalse();
  }

  @Test
  public void testAbsolute() {
    StripPrefixedPath result =
        StripPrefixedPath.maybeDeprefix("/foo/bar".getBytes(UTF_8), Optional.empty());
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("foo/bar"));

    result = StripPrefixedPath.maybeDeprefix("///foo/bar/baz".getBytes(UTF_8), Optional.empty());
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("foo/bar/baz"));

    result = StripPrefixedPath.maybeDeprefix("/foo/bar/baz".getBytes(UTF_8), Optional.of("/foo"));
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("bar/baz"));
  }

  @Test
  public void testWindowsAbsolute() {
    if (OS.getCurrent() != OS.WINDOWS) {
      return;
    }
    StripPrefixedPath result =
        StripPrefixedPath.maybeDeprefix("c:/foo/bar".getBytes(UTF_8), Optional.empty());
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("foo/bar"));
  }

  @Test
  public void testNormalize() {
    StripPrefixedPath result =
        StripPrefixedPath.maybeDeprefix("../bar".getBytes(UTF_8), Optional.empty());
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("../bar"));

    result = StripPrefixedPath.maybeDeprefix("foo/../baz".getBytes(UTF_8), Optional.empty());
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("baz"));

    result = StripPrefixedPath.maybeDeprefix("foo/../baz".getBytes(UTF_8), Optional.of("foo"));
    assertThat(result.getPathFragment()).isEqualTo(PathFragment.create("baz"));
  }

  @Test
  public void testDeprefixSymlink() {
    InMemoryFileSystem fileSystem =
        new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);

    PathFragment relativeNoPrefix =
        StripPrefixedPath.maybeDeprefixSymlink(
            "a/b".getBytes(UTF_8), Optional.empty(), fileSystem.getPath("/usr"));
    // there is no attempt to get absolute path for the relative symlinks target path
    assertThat(relativeNoPrefix).isEqualTo(PathFragment.create("a/b"));

    PathFragment absoluteNoPrefix =
        StripPrefixedPath.maybeDeprefixSymlink(
            "/a/b".getBytes(UTF_8), Optional.empty(), fileSystem.getPath("/usr"));
    assertThat(absoluteNoPrefix).isEqualTo(PathFragment.create("/usr/a/b"));

    PathFragment absolutePrefix =
        StripPrefixedPath.maybeDeprefixSymlink(
            "/root/a/b".getBytes(UTF_8), Optional.of("root"), fileSystem.getPath("/usr"));
    assertThat(absolutePrefix).isEqualTo(PathFragment.create("/usr/a/b"));

    PathFragment relativePrefix =
        StripPrefixedPath.maybeDeprefixSymlink(
            "root/a/b".getBytes(UTF_8), Optional.of("root"), fileSystem.getPath("/usr"));
    // there is no attempt to get absolute path for the relative symlinks target path
    assertThat(relativePrefix).isEqualTo(PathFragment.create("a/b"));
  }
}
