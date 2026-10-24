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
package com.google.devtools.build.lib.sandbox.backend;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.util.OS;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BackendConfig}. */
@RunWith(JUnit4.class)
public final class BackendConfigTest {

  @Test
  public void fromFlags_lastRegistrationWinsAndOptionsAccumulate() throws Exception {
    ImmutableMap<String, BackendConfig> configs =
        BackendConfig.fromFlags(
            List.of(Map.entry("a", "/old"), Map.entry("b", "b"), Map.entry("a", "/new")),
            List.of(Map.entry("a", "--x"), Map.entry("b", "--y"), Map.entry("a", "--z")));

    assertThat(configs.keySet()).containsExactly("a", "b").inOrder();
    assertThat(configs.get("a")).isEqualTo(new BackendConfig("/new", ImmutableList.of("--x", "--z")));
    assertThat(configs.get("b")).isEqualTo(new BackendConfig("b", ImmutableList.of("--y")));
  }

  @Test
  public void fromFlags_optionForUnknownBackend_fails() {
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                BackendConfig.fromFlags(
                    List.of(Map.entry("a", "/a")), List.of(Map.entry("c", "--x"))));

    assertThat(e).hasMessageThat().contains("unknown sandbox backend 'c'");
    assertThat(e).hasMessageThat().contains("registered backends: a");
  }

  @Test
  public void fromFlags_emptyBinary_fails() {
    assertThrows(
        IOException.class,
        () -> BackendConfig.fromFlags(List.of(Map.entry("a", "")), List.of()));
  }

  @Test
  public void resolveBinary_barenameUsesPath() throws Exception {
    boolean windows = OS.getCurrent() == OS.WINDOWS;
    java.nio.file.Path dir = java.nio.file.Files.createTempDirectory("bin");
    // On Windows a bare name is also tried with the extensions in PATHEXT.
    java.nio.file.Path binary = dir.resolve(windows ? "mybackend.exe" : "mybackend");
    java.nio.file.Files.writeString(binary, "#!/bin/sh\n");
    binary.toFile().setExecutable(true);
    String nonexistent = windows ? "C:\\nonexistent" : "/nonexistent";

    assertThat(
            new BackendConfig("mybackend", ImmutableList.of())
                .resolveBinary(
                    ImmutableMap.of("PATH", nonexistent + File.pathSeparator + dir)))
        .isEqualTo(binary.toString());
    assertThat(
            new BackendConfig("mybackend", ImmutableList.of())
                .resolveBinary(ImmutableMap.of("PATH", nonexistent)))
        .isNull();
    assertThat(
            new BackendConfig(nonexistent + File.separator + "x", ImmutableList.of())
                .resolveBinary(ImmutableMap.of()))
        .isNull();
  }

  @Test
  public void resolveBinary_nameWithPathSeparatorIsNotSearchedOnPath() throws Exception {
    java.nio.file.Path dir = java.nio.file.Files.createTempDirectory("bin");
    java.nio.file.Path binary = dir.resolve(OS.getCurrent() == OS.WINDOWS ? "b.exe" : "b");
    java.nio.file.Files.writeString(binary, "x");
    binary.toFile().setExecutable(true);

    // An absolute path is used as is.
    assertThat(new BackendConfig(binary.toString(), ImmutableList.of()).resolveBinary(ImmutableMap.of()))
        .isEqualTo(binary.toString());
    // A relative name with a separator is not looked up on PATH.
    assertThat(
            new BackendConfig("sub" + File.separator + binary.getFileName(), ImmutableList.of())
                .resolveBinary(ImmutableMap.of("PATH", dir.toString())))
        .isNull();
  }

  @Test
  public void resolveBinary_directoryIsNotAnExecutable() throws Exception {
    java.nio.file.Path dir = java.nio.file.Files.createTempDirectory("bin");
    java.nio.file.Files.createDirectory(dir.resolve("mybackend"));

    assertThat(
            new BackendConfig("mybackend", ImmutableList.of())
                .resolveBinary(ImmutableMap.of("PATH", dir.toString())))
        .isNull();
  }
}
