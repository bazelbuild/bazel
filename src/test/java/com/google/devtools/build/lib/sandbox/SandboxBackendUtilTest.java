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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.sandbox.SandboxBackendUtil.BackendConfig;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxBackendUtil}. */
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

  @Test
  public void configuredBackends_groupsOptionsByNameInOrder() throws Exception {
    ImmutableMap<String, BackendConfig> backends =
        SandboxBackendUtil.configuredBackends(
            ImmutableList.of(Map.entry("fskit", "/opt/sb"), Map.entry("cfs", "/opt/sb2")),
            ImmutableList.of(
                Map.entry("fskit", "backend=fskit"),
                Map.entry("cfs", "verbose=true"),
                Map.entry("fskit", "cache_dir=/x")));

    assertThat(backends.keySet()).containsExactly("fskit", "cfs").inOrder();
    assertThat(backends.get("fskit"))
        .isEqualTo(
            new BackendConfig(
                PathFragment.create("/opt/sb"),
                ImmutableList.of("backend=fskit", "cache_dir=/x")));
    assertThat(backends.get("cfs"))
        .isEqualTo(new BackendConfig(PathFragment.create("/opt/sb2"), ImmutableList.of("verbose=true")));
  }

  @Test
  public void configuredBackends_lastBinaryWinsForRepeatedName() throws Exception {
    // rc file says one path, command line overrides it: a later --sandbox_backend wins.
    ImmutableMap<String, BackendConfig> backends =
        SandboxBackendUtil.configuredBackends(
            ImmutableList.of(Map.entry("fskit", "/opt/old"), Map.entry("fskit", "/opt/new")),
            ImmutableList.of());

    assertThat(backends.keySet()).containsExactly("fskit");
    assertThat(backends.get("fskit").binary()).isEqualTo(PathFragment.create("/opt/new"));
  }

  @Test
  public void configuredBackends_optionForUnknownBackend_fails() {
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                SandboxBackendUtil.configuredBackends(
                    ImmutableList.of(Map.entry("fskit", "/opt/sb")),
                    ImmutableList.of(Map.entry("fskti", "cache_dir=/x"))));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "--sandbox_backend_opt=fskti=cache_dir=/x refers to unknown sandbox backend 'fskti';"
                + " registered backends: fskit");
  }

  @Test
  public void configuredBackends_optionWithNoBackendsRegistered_fails() {
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                SandboxBackendUtil.configuredBackends(
                    ImmutableList.of(), ImmutableList.of(Map.entry("fskit", "verbose=true"))));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "--sandbox_backend_opt=fskit=verbose=true refers to unknown sandbox backend 'fskit'; no"
                + " backends are registered with --sandbox_backend");
  }
}
