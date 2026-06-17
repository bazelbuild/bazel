// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxBackendUtil#isAvailable}. */
@RunWith(JUnit4.class)
public final class SandboxBackendUtilTest {

  @Test
  public void isAvailable_emptyPath_false() {
    assertThat(SandboxBackendUtil.isAvailable(PathFragment.EMPTY_FRAGMENT, ImmutableMap.of())).isFalse();
  }

  @Test
  public void isAvailable_absolutePathMissing_false() {
    assertThat(
            SandboxBackendUtil.isAvailable(
                PathFragment.create("/definitely/not/a/real/binary/anywhere"), ImmutableMap.of()))
        .isFalse();
  }

  @Test
  public void isAvailable_absolutePathExecutable_true() throws Exception {
    java.nio.file.Path tmp = java.nio.file.Files.createTempDirectory("sandbox-backend-util-test-");
    java.nio.file.Path bin = tmp.resolve("dummy");
    java.nio.file.Files.writeString(bin, "#!/bin/sh\nexit 0\n");
    bin.toFile().setExecutable(true);

    assertThat(SandboxBackendUtil.isAvailable(PathFragment.create(bin.toString()), ImmutableMap.of()))
        .isTrue();
  }

  @Test
  public void isAvailable_bareNameNoPath_false() {
    assertThat(SandboxBackendUtil.isAvailable(PathFragment.create("anything"), ImmutableMap.of()))
        .isFalse();
  }

  @Test
  public void isAvailable_bareNameOnPath_true() throws Exception {
    java.nio.file.Path tmp = java.nio.file.Files.createTempDirectory("sandbox-backend-util-pathtest-");
    java.nio.file.Path bin = tmp.resolve("my-controller");
    java.nio.file.Files.writeString(bin, "#!/bin/sh\nexit 0\n");
    bin.toFile().setExecutable(true);

    assertThat(
            SandboxBackendUtil.isAvailable(
                PathFragment.create("my-controller"), ImmutableMap.of("PATH", tmp.toString())))
        .isTrue();
  }

  @Test
  public void isAvailable_bareNameNotOnAnyPathEntry_false() {
    assertThat(
            SandboxBackendUtil.isAvailable(
                PathFragment.create("definitely-not-here-xyz-9999"),
                ImmutableMap.of("PATH", "/usr/bin:/bin")))
        .isFalse();
  }
}
