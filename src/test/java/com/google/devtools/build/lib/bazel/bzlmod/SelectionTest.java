// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.ModuleBuilder;
import com.google.devtools.build.lib.bazel.bzlmod.Selection.SelectionResult;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Selection}. */
@RunWith(JUnit4.class)
public class SelectionTest {

  @Test
  public void diamond_simple() throws Exception {
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("BfromA", createModuleKey("B", "1.0"))
                    .addDep("CfromA", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("DfromB", createModuleKey("D", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("C", "2.0")
                    .addDep("DfromC", createModuleKey("D", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "1.0", 1).buildEntry())
            .put(ModuleBuilder.create("D", "2.0", 1).buildEntry())
            .buildOrThrow();

    SelectionResult selectionResult = Selection.run(depGraph, /*overrides=*/ ImmutableMap.of());
    assertThat(selectionResult.getResolvedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("BfromA", createModuleKey("B", "1.0"))
                .addDep("CfromA", createModuleKey("C", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0")
                .addDep("DfromB", createModuleKey("D", "2.0"))
                .addOriginalDep("DfromB", createModuleKey("D", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("C", "2.0")
                .addDep("DfromC", createModuleKey("D", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("D", "2.0", 1).buildEntry())
        .inOrder();

    assertThat(selectionResult.getUnprunedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("BfromA", createModuleKey("B", "1.0"))
                .addDep("CfromA", createModuleKey("C", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0")
                .addDep("DfromB", createModuleKey("D", "2.0"))
                .addOriginalDep("DfromB", createModuleKey("D", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("C", "2.0")
                .addDep("DfromC", createModuleKey("D", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("D", "1.0", 1).buildEntry(),
            ModuleBuilder.create("D", "2.0", 1).buildEntry());
  }

  @Test
  public void diamond_withFurtherRemoval() throws Exception {
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B", createModuleKey("B", "1.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("D", createModuleKey("D", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("C", "2.0")
                    .addDep("D", createModuleKey("D", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("D", "1.0")
                    .addDep("E", createModuleKey("E", "1.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "2.0").buildEntry())
            // Only D@1.0 needs E. When D@1.0 is removed, E should be gone as well (even though
            // E@1.0 is selected for E).
            .put(ModuleBuilder.create("E", "1.0").buildEntry())
            .build();

    SelectionResult selectionResult = Selection.run(depGraph, /*overrides=*/ ImmutableMap.of());
    assertThat(selectionResult.getResolvedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("B", createModuleKey("B", "1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0")
                .addDep("D", createModuleKey("D", "2.0"))
                .addOriginalDep("D", createModuleKey("D", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("C", "2.0").addDep("D", createModuleKey("D", "2.0")).buildEntry(),
            ModuleBuilder.create("D", "2.0").buildEntry())
        .inOrder();

    assertThat(selectionResult.getUnprunedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("B", createModuleKey("B", "1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0")
                .addDep("D", createModuleKey("D", "2.0"))
                .addOriginalDep("D", createModuleKey("D", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("C", "2.0").addDep("D", createModuleKey("D", "2.0")).buildEntry(),
            ModuleBuilder.create("D", "2.0").buildEntry(),
            ModuleBuilder.create("D", "1.0").addDep("E", createModuleKey("E", "1.0")).buildEntry(),
            ModuleBuilder.create("E", "1.0").buildEntry());
  }

  @Test
  public void circularDependencyDueToSelection() throws Exception {
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B", createModuleKey("B", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("C", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("C", "2.0")
                    .addDep("B", createModuleKey("B", "1.0-pre"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0-pre")
                    .addDep("D", createModuleKey("D", "1.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "1.0").buildEntry())
            .buildOrThrow();

    SelectionResult selectionResult = Selection.run(depGraph, /*overrides=*/ ImmutableMap.of());
    assertThat(selectionResult.getResolvedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("B", createModuleKey("B", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0").addDep("C", createModuleKey("C", "2.0")).buildEntry(),
            ModuleBuilder.create("C", "2.0")
                .addDep("B", createModuleKey("B", "1.0"))
                .addOriginalDep("B", createModuleKey("B", "1.0-pre"))
                .buildEntry())
        .inOrder();
    // D is completely gone.

    assertThat(selectionResult.getUnprunedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("B", createModuleKey("B", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0").addDep("C", createModuleKey("C", "2.0")).buildEntry(),
            ModuleBuilder.create("C", "2.0")
                .addDep("B", createModuleKey("B", "1.0"))
                .addOriginalDep("B", createModuleKey("B", "1.0-pre"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0-pre")
                .addDep("D", createModuleKey("D", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("D", "1.0").buildEntry());
  }

  @Test
  public void differentCompatibilityLevelIsRejected() throws Exception {
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("BfromA", createModuleKey("B", "1.0"))
                    .addDep("CfromA", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("DfromB", createModuleKey("D", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("C", "2.0")
                    .addDep("DfromC", createModuleKey("D", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "1.0", 1).buildEntry())
            .put(ModuleBuilder.create("D", "2.0", 2).buildEntry())
            .buildOrThrow();

    ExternalDepsException e =
        assertThrows(
            ExternalDepsException.class,
            () -> Selection.run(depGraph, /*overrides=*/ ImmutableMap.of()));
    String error = e.getMessage();
    assertThat(error).contains("B@1.0 depends on D@1.0 with compatibility level 1");
    assertThat(error).contains("C@2.0 depends on D@2.0 with compatibility level 2");
    assertThat(error).contains("which is different");
  }

  @Test
  public void differentCompatibilityLevelIsOkIfUnreferenced() throws Exception {
    // A 1.0 -> B 1.0 -> C 2.0
    //       \-> C 1.0
    //        \-> D 1.0 -> B 1.1
    //         \-> E 1.0 -> C 1.1
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", "1.0")
                    .setKey(ModuleKey.ROOT)
                    .addDep("B", createModuleKey("B", "1.0"))
                    .addDep("C", createModuleKey("C", "1.0"))
                    .addDep("D", createModuleKey("D", "1.0"))
                    .addDep("E", createModuleKey("E", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("C", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("C", "2.0", 2).buildEntry())
            .put(ModuleBuilder.create("C", "1.0", 1).buildEntry())
            .put(
                ModuleBuilder.create("D", "1.0")
                    .addDep("B", createModuleKey("B", "1.1"))
                    .buildEntry())
            .put(ModuleBuilder.create("B", "1.1").buildEntry())
            .put(
                ModuleBuilder.create("E", "1.0")
                    .addDep("C", createModuleKey("C", "1.1"))
                    .buildEntry())
            .put(ModuleBuilder.create("C", "1.1", 1).buildEntry())
            .buildOrThrow();

    // After selection, C 2.0 is gone, so we're okay.
    // A 1.0 -> B 1.1
    //       \-> C 1.1
    //        \-> D 1.0 -> B 1.1
    //         \-> E 1.0 -> C 1.1
    SelectionResult selectionResult = Selection.run(depGraph, /*overrides=*/ ImmutableMap.of());
    assertThat(selectionResult.getResolvedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", "1.0")
                .setKey(ModuleKey.ROOT)
                .addDep("B", createModuleKey("B", "1.1"))
                .addOriginalDep("B", createModuleKey("B", "1.0"))
                .addDep("C", createModuleKey("C", "1.1"))
                .addOriginalDep("C", createModuleKey("C", "1.0"))
                .addDep("D", createModuleKey("D", "1.0"))
                .addDep("E", createModuleKey("E", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.1").buildEntry(),
            ModuleBuilder.create("C", "1.1", 1).buildEntry(),
            ModuleBuilder.create("D", "1.0").addDep("B", createModuleKey("B", "1.1")).buildEntry(),
            ModuleBuilder.create("E", "1.0").addDep("C", createModuleKey("C", "1.1")).buildEntry())
        .inOrder();

    assertThat(selectionResult.getUnprunedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", "1.0")
                .setKey(ModuleKey.ROOT)
                .addDep("B", createModuleKey("B", "1.1"))
                .addOriginalDep("B", createModuleKey("B", "1.0"))
                .addDep("C", createModuleKey("C", "1.1"))
                .addOriginalDep("C", createModuleKey("C", "1.0"))
                .addDep("D", createModuleKey("D", "1.0"))
                .addDep("E", createModuleKey("E", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0").addDep("C", createModuleKey("C", "2.0")).buildEntry(),
            ModuleBuilder.create("B", "1.1").buildEntry(),
            ModuleBuilder.create("C", "1.0", 1).buildEntry(),
            ModuleBuilder.create("C", "1.1", 1).buildEntry(),
            ModuleBuilder.create("C", "2.0", 2).buildEntry(),
            ModuleBuilder.create("D", "1.0").addDep("B", createModuleKey("B", "1.1")).buildEntry(),
            ModuleBuilder.create("E", "1.0").addDep("C", createModuleKey("C", "1.1")).buildEntry());
  }

  @Test
  public void multipleVersionOverride_fork_allowedVersionMissingInDepGraph() throws Exception {
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B1", createModuleKey("B", "1.0"))
                    .addDep("B2", createModuleKey("B", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("B", "1.0").buildEntry())
            .put(ModuleBuilder.create("B", "2.0").buildEntry())
            .buildOrThrow();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "B",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0"), Version.parse("3.0")),
                ""));

    ExternalDepsException e =
        assertThrows(ExternalDepsException.class, () -> Selection.run(depGraph, overrides));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "multiple_version_override for module B contains version 3.0, but it doesn't exist in"
                + " the dependency graph");
  }

  @Test
  public void multipleVersionOverride_fork_goodCase() throws Exception {
    // For more complex good cases, see the "diamond" test cases below.
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B1", createModuleKey("B", "1.0"))
                    .addDep("B2", createModuleKey("B", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("B", "1.0").buildEntry())
            .put(ModuleBuilder.create("B", "2.0").buildEntry())
            .buildOrThrow();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "B",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""));

    SelectionResult selectionResult = Selection.run(depGraph, overrides);
    assertThat(selectionResult.getResolvedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("B1", createModuleKey("B", "1.0"))
                .addDep("B2", createModuleKey("B", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0").buildEntry(),
            ModuleBuilder.create("B", "2.0").buildEntry())
        .inOrder();

    assertThat(selectionResult.getUnprunedDepGraph())
        .isEqualTo(selectionResult.getResolvedDepGraph());
  }

  @Test
  public void multipleVersionOverride_fork_sameVersionUsedTwice() throws Exception {
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B1", createModuleKey("B", "1.0"))
                    .addDep("B2", createModuleKey("B", "1.3"))
                    .addDep("B3", createModuleKey("B", "1.5"))
                    .buildEntry())
            .put(ModuleBuilder.create("B", "1.0").buildEntry())
            .put(ModuleBuilder.create("B", "1.3").buildEntry())
            .put(ModuleBuilder.create("B", "1.5").buildEntry())
            .buildOrThrow();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "B",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("1.5")), ""));

    ExternalDepsException e =
        assertThrows(ExternalDepsException.class, () -> Selection.run(depGraph, overrides));
    assertThat(e)
        .hasMessageThat()
        .containsMatch(
            "A@_ depends on B@1.5 at least twice \\(with repo names (B2 and B3)|(B3 and B2)\\)");
  }

  @Test
  public void multipleVersionOverride_diamond_differentCompatibilityLevels() throws Exception {
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("BfromA", createModuleKey("B", "1.0"))
                    .addDep("CfromA", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("DfromB", createModuleKey("D", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("C", "2.0")
                    .addDep("DfromC", createModuleKey("D", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "1.0", 1).buildEntry())
            .put(ModuleBuilder.create("D", "2.0", 2).buildEntry())
            .buildOrThrow();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "D",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""));

    SelectionResult selectionResult = Selection.run(depGraph, overrides);
    assertThat(selectionResult.getResolvedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("BfromA", createModuleKey("B", "1.0"))
                .addDep("CfromA", createModuleKey("C", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0")
                .addDep("DfromB", createModuleKey("D", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("C", "2.0")
                .addDep("DfromC", createModuleKey("D", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("D", "1.0", 1).buildEntry(),
            ModuleBuilder.create("D", "2.0", 2).buildEntry())
        .inOrder();

    assertThat(selectionResult.getUnprunedDepGraph())
        .isEqualTo(selectionResult.getResolvedDepGraph());
  }

  @Test
  public void multipleVersionOverride_diamond_sameCompatibilityLevel() throws Exception {
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("BfromA", createModuleKey("B", "1.0"))
                    .addDep("CfromA", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("DfromB", createModuleKey("D", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("C", "2.0")
                    .addDep("DfromC", createModuleKey("D", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "1.0").buildEntry())
            .put(ModuleBuilder.create("D", "2.0").buildEntry())
            .buildOrThrow();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "D",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""));

    SelectionResult selectionResult = Selection.run(depGraph, overrides);
    assertThat(selectionResult.getResolvedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("BfromA", createModuleKey("B", "1.0"))
                .addDep("CfromA", createModuleKey("C", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("B", "1.0")
                .addDep("DfromB", createModuleKey("D", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("C", "2.0")
                .addDep("DfromC", createModuleKey("D", "2.0"))
                .buildEntry(),
            ModuleBuilder.create("D", "1.0").buildEntry(),
            ModuleBuilder.create("D", "2.0").buildEntry())
        .inOrder();

    assertThat(selectionResult.getUnprunedDepGraph())
        .isEqualTo(selectionResult.getResolvedDepGraph());
  }

  @Test
  public void multipleVersionOverride_diamond_snappingToNextHighestVersion() throws Exception {
    // A --> B1@1.0 -> C@1.0
    //   \-> B2@1.0 -> C@1.3  [allowed]
    //   \-> B3@1.0 -> C@1.5
    //   \-> B4@1.0 -> C@1.7  [allowed]
    //   \-> B5@1.0 -> C@2.0  [allowed]
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B1", createModuleKey("B1", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.0"))
                    .addDep("B3", createModuleKey("B3", "1.0"))
                    .addDep("B4", createModuleKey("B4", "1.0"))
                    .addDep("B5", createModuleKey("B5", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B1", "1.0")
                    .addDep("C", createModuleKey("C", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B2", "1.0")
                    .addDep("C", createModuleKey("C", "1.3"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B3", "1.0")
                    .addDep("C", createModuleKey("C", "1.5"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B4", "1.0")
                    .addDep("C", createModuleKey("C", "1.7"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B5", "1.0")
                    .addDep("C", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("C", "1.0", 1).buildEntry())
            .put(ModuleBuilder.create("C", "1.3", 1).buildEntry())
            .put(ModuleBuilder.create("C", "1.5", 1).buildEntry())
            .put(ModuleBuilder.create("C", "1.7", 1).buildEntry())
            .put(ModuleBuilder.create("C", "2.0", 2).buildEntry())
            .buildOrThrow();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "C",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.3"), Version.parse("1.7"), Version.parse("2.0")),
                ""));

    // A --> B1@1.0 -> C@1.3  [originally C@1.0]
    //   \-> B2@1.0 -> C@1.3  [allowed]
    //   \-> B3@1.0 -> C@1.7  [originally C@1.5]
    //   \-> B4@1.0 -> C@1.7  [allowed]
    //   \-> B5@1.0 -> C@2.0  [allowed]
    SelectionResult selectionResult = Selection.run(depGraph, overrides);
    assertThat(selectionResult.getResolvedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("B1", createModuleKey("B1", "1.0"))
                .addDep("B2", createModuleKey("B2", "1.0"))
                .addDep("B3", createModuleKey("B3", "1.0"))
                .addDep("B4", createModuleKey("B4", "1.0"))
                .addDep("B5", createModuleKey("B5", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B1", "1.0")
                .addDep("C", createModuleKey("C", "1.3"))
                .addOriginalDep("C", createModuleKey("C", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B2", "1.0").addDep("C", createModuleKey("C", "1.3")).buildEntry(),
            ModuleBuilder.create("B3", "1.0")
                .addDep("C", createModuleKey("C", "1.7"))
                .addOriginalDep("C", createModuleKey("C", "1.5"))
                .buildEntry(),
            ModuleBuilder.create("B4", "1.0").addDep("C", createModuleKey("C", "1.7")).buildEntry(),
            ModuleBuilder.create("B5", "1.0").addDep("C", createModuleKey("C", "2.0")).buildEntry(),
            ModuleBuilder.create("C", "1.3", 1).buildEntry(),
            ModuleBuilder.create("C", "1.7", 1).buildEntry(),
            ModuleBuilder.create("C", "2.0", 2).buildEntry())
        .inOrder();

    assertThat(selectionResult.getUnprunedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("B1", createModuleKey("B1", "1.0"))
                .addDep("B2", createModuleKey("B2", "1.0"))
                .addDep("B3", createModuleKey("B3", "1.0"))
                .addDep("B4", createModuleKey("B4", "1.0"))
                .addDep("B5", createModuleKey("B5", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B1", "1.0")
                .addDep("C", createModuleKey("C", "1.3"))
                .addOriginalDep("C", createModuleKey("C", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B2", "1.0").addDep("C", createModuleKey("C", "1.3")).buildEntry(),
            ModuleBuilder.create("B3", "1.0")
                .addDep("C", createModuleKey("C", "1.7"))
                .addOriginalDep("C", createModuleKey("C", "1.5"))
                .buildEntry(),
            ModuleBuilder.create("B4", "1.0").addDep("C", createModuleKey("C", "1.7")).buildEntry(),
            ModuleBuilder.create("B5", "1.0").addDep("C", createModuleKey("C", "2.0")).buildEntry(),
            ModuleBuilder.create("C", "1.0", 1).buildEntry(),
            ModuleBuilder.create("C", "1.3", 1).buildEntry(),
            ModuleBuilder.create("C", "1.5", 1).buildEntry(),
            ModuleBuilder.create("C", "1.7", 1).buildEntry(),
            ModuleBuilder.create("C", "2.0", 2).buildEntry());
  }

  @Test
  public void multipleVersionOverride_diamond_dontSnapToDifferentCompatibility() throws Exception {
    // A --> B1@1.0 -> C@1.0  [allowed]
    //   \-> B2@1.0 -> C@1.7
    //   \-> B3@1.0 -> C@2.0  [allowed]
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B1", createModuleKey("B1", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.0"))
                    .addDep("B3", createModuleKey("B3", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B1", "1.0")
                    .addDep("C", createModuleKey("C", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B2", "1.0")
                    .addDep("C", createModuleKey("C", "1.7"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B3", "1.0")
                    .addDep("C", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("C", "1.0", 1).buildEntry())
            .put(ModuleBuilder.create("C", "1.7", 1).buildEntry())
            .put(ModuleBuilder.create("C", "2.0", 2).buildEntry())
            .buildOrThrow();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "C",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""));

    ExternalDepsException e =
        assertThrows(ExternalDepsException.class, () -> Selection.run(depGraph, overrides));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "B2@1.0 depends on C@1.7 which is not allowed by the multiple_version_override on C,"
                + " which allows only [1.0, 2.0]");
  }

  @Test
  public void multipleVersionOverride_diamond_unknownCompatibility() throws Exception {
    // A --> B1@1.0 -> C@1.0  [allowed]
    //   \-> B2@1.0 -> C@2.0  [allowed]
    //   \-> B3@1.0 -> C@3.0
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B1", createModuleKey("B1", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.0"))
                    .addDep("B3", createModuleKey("B3", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B1", "1.0")
                    .addDep("C", createModuleKey("C", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B2", "1.0")
                    .addDep("C", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B3", "1.0")
                    .addDep("C", createModuleKey("C", "3.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("C", "1.0", 1).buildEntry())
            .put(ModuleBuilder.create("C", "2.0", 2).buildEntry())
            .put(ModuleBuilder.create("C", "3.0", 3).buildEntry())
            .buildOrThrow();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "C",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""));

    ExternalDepsException e =
        assertThrows(ExternalDepsException.class, () -> Selection.run(depGraph, overrides));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "B3@1.0 depends on C@3.0 which is not allowed by the multiple_version_override on C,"
                + " which allows only [1.0, 2.0]");
  }

  @Test
  public void multipleVersionOverride_diamond_badVersionsAreOkayIfUnreferenced() throws Exception {
    // A --> B1@1.0 --> C@1.0  [allowed]
    //   \          \-> B2@1.1
    //   \-> B2@1.0 --> C@1.5
    //   \-> B3@1.0 --> C@2.0  [allowed]
    //   \          \-> B4@1.1
    //   \-> B4@1.0 --> C@3.0
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B1", createModuleKey("B1", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.0"))
                    .addDep("B3", createModuleKey("B3", "1.0"))
                    .addDep("B4", createModuleKey("B4", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B1", "1.0")
                    .addDep("C", createModuleKey("C", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.1"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B2", "1.0")
                    .addDep("C", createModuleKey("C", "1.5"))
                    .buildEntry())
            .put(ModuleBuilder.create("B2", "1.1").buildEntry())
            .put(
                ModuleBuilder.create("B3", "1.0")
                    .addDep("C", createModuleKey("C", "2.0"))
                    .addDep("B4", createModuleKey("B4", "1.1"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B4", "1.0")
                    .addDep("C", createModuleKey("C", "3.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("B4", "1.1").buildEntry())
            .put(ModuleBuilder.create("C", "1.0", 1).buildEntry())
            .put(ModuleBuilder.create("C", "1.5", 1).buildEntry())
            .put(ModuleBuilder.create("C", "2.0", 2).buildEntry())
            .put(ModuleBuilder.create("C", "3.0", 3).buildEntry())
            .buildOrThrow();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "C",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""));

    // A --> B1@1.0 --> C@1.0  [allowed]
    //   \          \-> B2@1.1
    //   \-> B2@1.1
    //   \-> B3@1.0 --> C@2.0  [allowed]
    //   \          \-> B4@1.1
    //   \-> B4@1.1
    // C@1.5 and C@3.0, the versions violating the allowlist, are gone.
    SelectionResult selectionResult = Selection.run(depGraph, overrides);
    assertThat(selectionResult.getResolvedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("B1", createModuleKey("B1", "1.0"))
                .addDep("B2", createModuleKey("B2", "1.1"))
                .addOriginalDep("B2", createModuleKey("B2", "1.0"))
                .addDep("B3", createModuleKey("B3", "1.0"))
                .addDep("B4", createModuleKey("B4", "1.1"))
                .addOriginalDep("B4", createModuleKey("B4", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B1", "1.0")
                .addDep("C", createModuleKey("C", "1.0"))
                .addDep("B2", createModuleKey("B2", "1.1"))
                .buildEntry(),
            ModuleBuilder.create("B2", "1.1").buildEntry(),
            ModuleBuilder.create("B3", "1.0")
                .addDep("C", createModuleKey("C", "2.0"))
                .addDep("B4", createModuleKey("B4", "1.1"))
                .buildEntry(),
            ModuleBuilder.create("B4", "1.1").buildEntry(),
            ModuleBuilder.create("C", "1.0", 1).buildEntry(),
            ModuleBuilder.create("C", "2.0", 2).buildEntry())
        .inOrder();

    assertThat(selectionResult.getUnprunedDepGraph().entrySet())
        .containsExactly(
            ModuleBuilder.create("A", Version.EMPTY)
                .setKey(ModuleKey.ROOT)
                .addDep("B1", createModuleKey("B1", "1.0"))
                .addDep("B2", createModuleKey("B2", "1.1"))
                .addOriginalDep("B2", createModuleKey("B2", "1.0"))
                .addDep("B3", createModuleKey("B3", "1.0"))
                .addDep("B4", createModuleKey("B4", "1.1"))
                .addOriginalDep("B4", createModuleKey("B4", "1.0"))
                .buildEntry(),
            ModuleBuilder.create("B1", "1.0")
                .addDep("C", createModuleKey("C", "1.0"))
                .addDep("B2", createModuleKey("B2", "1.1"))
                .buildEntry(),
            ModuleBuilder.create("B2", "1.0").addDep("C", createModuleKey("C", "1.5")).buildEntry(),
            ModuleBuilder.create("B2", "1.1").buildEntry(),
            ModuleBuilder.create("B3", "1.0")
                .addDep("C", createModuleKey("C", "2.0"))
                .addDep("B4", createModuleKey("B4", "1.1"))
                .buildEntry(),
            ModuleBuilder.create("B4", "1.0").addDep("C", createModuleKey("C", "3.0")).buildEntry(),
            ModuleBuilder.create("B4", "1.1").buildEntry(),
            ModuleBuilder.create("C", "1.0", 1).buildEntry(),
            ModuleBuilder.create("C", "1.5", 1).buildEntry(),
            ModuleBuilder.create("C", "2.0", 2).buildEntry(),
            ModuleBuilder.create("C", "3.0", 3).buildEntry());
  }
}
