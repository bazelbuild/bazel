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

package com.google.devtools.build.lib.bazel.bzlmod.modcommand;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.VersionsRenderer.ModuleVersionEntry;
import java.io.ByteArrayOutputStream;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link VersionsRenderer} used by the {@code mod upgrade} subcommand. */
@RunWith(JUnit4.class)
public class VersionsRendererTest {

  @Test
  public void findLatestStable_prefersNonPrerelease() throws ParseException {
    List<Version> versions =
        ImmutableList.of(
            Version.parse("1.0.0"), Version.parse("2.0.0"), Version.parse("3.0.0-rc1"));
    assertThat(VersionsRenderer.findLatestStable(versions)).isEqualTo(Version.parse("2.0.0"));
  }

  @Test
  public void findLatestStable_fallsBackToPrerelease() throws ParseException {
    List<Version> versions =
        ImmutableList.of(Version.parse("1.0.0-alpha"), Version.parse("2.0.0-beta"));
    assertThat(VersionsRenderer.findLatestStable(versions)).isEqualTo(Version.parse("2.0.0-beta"));
  }

  @Test
  public void findLatestStable_emptyList() {
    assertThat(VersionsRenderer.findLatestStable(ImmutableList.of())).isNull();
  }

  @Test
  public void findLatestStable_unsortedInput() throws ParseException {
    List<Version> versions =
        ImmutableList.of(Version.parse("3.0.0"), Version.parse("1.0.0"), Version.parse("2.0.0"));
    assertThat(VersionsRenderer.findLatestStable(versions)).isEqualTo(Version.parse("3.0.0"));
  }

  @Test
  public void render_directAndTransitiveDeps() throws ParseException {
    List<ModuleVersionEntry> directDeps =
        List.of(
            new ModuleVersionEntry(
                "rules_java", Version.parse("7.6.5"), Version.parse("8.0.0"), true),
            new ModuleVersionEntry(
                "rules_python", Version.parse("0.31.0"), Version.parse("0.31.0"), true));

    List<ModuleVersionEntry> transitiveDeps =
        List.of(
            new ModuleVersionEntry(
                "platforms", Version.parse("0.0.9"), Version.parse("0.0.9"), false),
            new ModuleVersionEntry(
                "abseil-cpp", Version.parse("20240116.2"), Version.parse("20240722.0"), false));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ false, /* useUtf8= */ false)
        .render(directDeps, transitiveDeps);
    String output = out.toString();

    // Header for direct deps.
    assertThat(output).contains("Module");
    assertThat(output).contains("Current");
    assertThat(output).contains("Latest");
    assertThat(output).contains("Status");

    // Header for transitive deps.
    assertThat(output).contains("Module (Indirect)");

    // Direct deps rows.
    assertThat(output).contains("rules_java");
    assertThat(output).contains("7.6.5");
    assertThat(output).contains("8.0.0");
    assertThat(output).contains("upgrade available");

    assertThat(output).contains("rules_python");
    assertThat(output).contains("0.31.0");
    assertThat(output).contains("up to date");

    // Transitive deps rows (sorted alphabetically, so abseil-cpp before platforms).
    assertThat(output).contains("abseil-cpp");
    assertThat(output).contains("20240116.2");
    assertThat(output).contains("20240722.0");

    assertThat(output).contains("platforms");
    assertThat(output).contains("0.0.9");

    // Summary.
    assertThat(output).contains("4 modules total, 2 with upgrades available");
  }

  @Test
  public void render_allUpToDate() throws ParseException {
    List<ModuleVersionEntry> directDeps =
        List.of(
            new ModuleVersionEntry(
                "rules_java", Version.parse("8.0.0"), Version.parse("8.0.0"), true));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ false, /* useUtf8= */ false)
        .render(directDeps, List.of());
    String output = out.toString();

    assertThat(output).contains("1 modules total, all up to date");
    assertThat(output).doesNotContain("upgrade available");
  }

  @Test
  public void render_unknownLatestVersion() throws ParseException {
    List<ModuleVersionEntry> directDeps =
        List.of(
            new ModuleVersionEntry("my_module", Version.parse("1.0.0"), /* latest= */ null, true));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ false, /* useUtf8= */ false)
        .render(directDeps, List.of());
    String output = out.toString();

    assertThat(output).contains("my_module");
    assertThat(output).contains("?");
    assertThat(output).contains("unknown");
    assertThat(output).contains("1 modules total, all up to date (1 with unknown versions)");
  }

  @Test
  public void render_emptyDeps() {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ false, /* useUtf8= */ false)
        .render(List.of(), List.of());
    String output = out.toString();

    assertThat(output).contains("No external module dependencies found.");
  }

  @Test
  public void render_onlyTransitiveDeps() throws ParseException {
    List<ModuleVersionEntry> transitiveDeps =
        List.of(
            new ModuleVersionEntry(
                "protobuf", Version.parse("25.0"), Version.parse("27.0"), false));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ false, /* useUtf8= */ false)
        .render(List.of(), transitiveDeps);
    String output = out.toString();

    // Should have indirect header but no direct header.
    assertThat(output).contains("Module (Indirect)");
    assertThat(output).contains("protobuf");
    assertThat(output).contains("1 modules total, 1 with upgrades available");
  }

  @Test
  public void render_utf8Separator() throws ParseException {
    List<ModuleVersionEntry> directDeps =
        List.of(
            new ModuleVersionEntry("foo", Version.parse("1.0.0"), Version.parse("1.0.0"), true));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ false, /* useUtf8= */ true)
        .render(directDeps, List.of());
    String output = out.toString();

    assertThat(output).contains("\u2500"); // Box-drawing horizontal line.
  }

  @Test
  public void render_asciiSeparator() throws ParseException {
    List<ModuleVersionEntry> directDeps =
        List.of(
            new ModuleVersionEntry("foo", Version.parse("1.0.0"), Version.parse("1.0.0"), true));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ false, /* useUtf8= */ false)
        .render(directDeps, List.of());
    String output = out.toString();

    assertThat(output).contains("---");
    assertThat(output).doesNotContain("\u2500");
  }

  @Test
  public void render_sortsByName() throws ParseException {
    List<ModuleVersionEntry> directDeps =
        List.of(
            new ModuleVersionEntry(
                "zzz_module", Version.parse("1.0.0"), Version.parse("1.0.0"), true),
            new ModuleVersionEntry(
                "aaa_module", Version.parse("2.0.0"), Version.parse("2.0.0"), true));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ false, /* useUtf8= */ false)
        .render(directDeps, List.of());
    String output = out.toString();

    // aaa_module should appear before zzz_module.
    assertThat(output.indexOf("aaa_module")).isLessThan(output.indexOf("zzz_module"));
  }

  @Test
  public void findLatestStable_singleElement() throws ParseException {
    List<Version> versions = ImmutableList.of(Version.parse("1.0.0"));
    assertThat(VersionsRenderer.findLatestStable(versions)).isEqualTo(Version.parse("1.0.0"));
  }

  @Test
  public void render_withColor() throws ParseException {
    List<ModuleVersionEntry> directDeps =
        List.of(
            new ModuleVersionEntry("foo", Version.parse("1.0.0"), Version.parse("2.0.0"), true));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ true, /* useUtf8= */ false)
        .render(directDeps, List.of());
    String output = out.toString();

    // ANSI escape sequences should be present.
    assertThat(output).contains("\u001b[");
  }

  @Test
  public void renderHint_containsUpgradeInstructions() {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    new VersionsRenderer(out, /* useColor= */ false, /* useUtf8= */ false).renderHint();
    String output = out.toString();

    assertThat(output).contains("bazel mod upgrade <module>");
    assertThat(output).contains("--all");
  }
}
