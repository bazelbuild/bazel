// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule.ResolutionReason;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.ModuleBuilder;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BazelModuleInspectorFunction}. */
@RunWith(JUnit4.class)
public class BazelModuleInspectorFunctionTest {

  @Test
  public void testDiamond_simple() throws Exception {
    ImmutableMap<ModuleKey, Module> unprunedDepGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("BfromA", createModuleKey("B", "1.0"))
                    .addDep("CfromA", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("DfromB", createModuleKey("D", "2.0"))
                    .addOriginalDep("DfromB", createModuleKey("D", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("C", "2.0")
                    .addDep("DfromC", createModuleKey("D", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "1.0", 1).buildEntry())
            .put(ModuleBuilder.create("D", "2.0", 1).buildEntry())
            .buildOrThrow();

    ImmutableSet<ModuleKey> usedModules =
        ImmutableSet.of(
            ModuleKey.ROOT,
            createModuleKey("B", "1.0"),
            createModuleKey("C", "2.0"),
            createModuleKey("D", "2.0"));

    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        BazelModuleInspectorFunction.computeAugmentedGraph(
            unprunedDepGraph, usedModules, /*overrides*/ ImmutableMap.of());

    assertThat(depGraph.entrySet())
        .containsExactly(
            buildAugmentedModule(ModuleKey.ROOT, "A")
                .addDep("B", "1.0")
                .addDep("C", "2.0")
                .buildEntry(),
            buildAugmentedModule("B", "1.0")
                .addDep("D", "2.0", ResolutionReason.MINIMAL_VERSION_SELECTION)
                .addStillDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("C", "2.0")
                .addDep("D", "2.0")
                .addStillDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("D", "2.0")
                .addDependant("B", "1.0")
                .addStillDependant("C", "2.0")
                .buildEntry(),
            buildAugmentedModule("D", "1.0").addOriginalDependant("B", "1.0").buildEntry());
  }

  @Test
  public void testDiamond_withFurtherRemoval() throws Exception {
    ImmutableMap<ModuleKey, Module> unprunedDepGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B", createModuleKey("B", "1.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("D", createModuleKey("D", "2.0"))
                    .addOriginalDep("D", createModuleKey("D", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("C", "2.0")
                    .addDep("D", createModuleKey("D", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "2.0").buildEntry())
            .put(
                ModuleBuilder.create("D", "1.0")
                    .addDep("E", createModuleKey("E", "1.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("E", "1.0").buildEntry())
            .buildOrThrow();

    ImmutableSet<ModuleKey> usedModules =
        ImmutableSet.of(
            ModuleKey.ROOT,
            createModuleKey("B", "1.0"),
            createModuleKey("C", "2.0"),
            createModuleKey("D", "2.0"));

    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        BazelModuleInspectorFunction.computeAugmentedGraph(
            unprunedDepGraph, usedModules, /*overrides*/ ImmutableMap.of());

    assertThat(depGraph.entrySet())
        .containsExactly(
            buildAugmentedModule(ModuleKey.ROOT, "A")
                .addDep("B", "1.0")
                .addDep("C", "2.0")
                .buildEntry(),
            buildAugmentedModule("B", "1.0")
                .addDep("D", "2.0", ResolutionReason.MINIMAL_VERSION_SELECTION)
                .addStillDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("C", "2.0")
                .addDep("D", "2.0")
                .addStillDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("D", "2.0")
                .addDependant("B", "1.0")
                .addStillDependant("C", "2.0")
                .buildEntry(),
            buildAugmentedModule("D", "1.0")
                .addDep("E", "1.0")
                .addOriginalDependant("B", "1.0")
                .buildEntry(),
            buildAugmentedModule("E", "1.0").addOriginalDependant("D", "1.0").buildEntry());
  }

  @Test
  public void testCircularDependencyDueToSelection() throws Exception {
    ImmutableMap<ModuleKey, Module> unprunedDepGraph =
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
                    .addDep("B", createModuleKey("B", "1.0"))
                    .addOriginalDep("B", createModuleKey("B", "1.0-pre"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0-pre")
                    .addDep("D", createModuleKey("D", "1.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "1.0").buildEntry())
            .buildOrThrow();

    ImmutableSet<ModuleKey> usedModules =
        ImmutableSet.of(ModuleKey.ROOT, createModuleKey("B", "1.0"), createModuleKey("C", "2.0"));

    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        BazelModuleInspectorFunction.computeAugmentedGraph(
            unprunedDepGraph, usedModules, /*overrides*/ ImmutableMap.of());

    assertThat(depGraph.entrySet())
        .containsExactly(
            buildAugmentedModule(ModuleKey.ROOT, "A").addDep("B", "1.0").buildEntry(),
            buildAugmentedModule("B", "1.0")
                .addDep("C", "2.0")
                .addStillDependant(ModuleKey.ROOT)
                .addDependant("C", "2.0")
                .buildEntry(),
            buildAugmentedModule("C", "2.0")
                .addDep("B", "1.0", ResolutionReason.MINIMAL_VERSION_SELECTION)
                .addStillDependant("B", "1.0")
                .buildEntry(),
            buildAugmentedModule("B", "1.0-pre")
                .addDep("D", "1.0")
                .addOriginalDependant("C", "2.0")
                .buildEntry(),
            buildAugmentedModule("D", "1.0").addOriginalDependant("B", "1.0-pre").buildEntry());
  }

  @Test
  public void testSingleVersionOverride_withRemoval() throws Exception {
    // Original (non-resolved) dep graph
    // single_version_override (C, 2.0)
    // A -> B 1.0 -> C 1.0 -> D -> 1.0
    //               C 2.0 -> D -> 2.0
    ImmutableMap<ModuleKey, Module> unprunedDepGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B", createModuleKey("B", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("C", createModuleKey("C", "2.0"))
                    .addOriginalDep("C", createModuleKey("C", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("C", "2.0")
                    .addDep("D", createModuleKey("D", "2.0"))
                    .buildEntry())
            .put(ModuleBuilder.create("D", "2.0").buildEntry())
            .buildOrThrow();

    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "C", SingleVersionOverride.create(Version.parse("2.0"), "", ImmutableList.of(), 0));

    ImmutableSet<ModuleKey> usedModules =
        ImmutableSet.of(
            ModuleKey.ROOT,
            createModuleKey("B", "1.0"),
            createModuleKey("C", "1.0"),
            createModuleKey("C", "2.0"),
            createModuleKey("D", "1.0"),
            createModuleKey("D", "2.0"));

    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        BazelModuleInspectorFunction.computeAugmentedGraph(
            unprunedDepGraph, usedModules, overrides);

    assertThat(depGraph.entrySet())
        .containsExactly(
            buildAugmentedModule(ModuleKey.ROOT, "A").addDep("B", "1.0").buildEntry(),
            buildAugmentedModule("B", "1.0")
                .addDep("C", "2.0", ResolutionReason.SINGLE_VERSION_OVERRIDE)
                .addStillDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("C", "1.0", false).addOriginalDependant("B", "1.0").buildEntry(),
            buildAugmentedModule("C", "2.0")
                .addDependant("B", "1.0")
                .addDep("D", "2.0")
                .buildEntry(),
            buildAugmentedModule("D", "2.0").addStillDependant("C", "2.0").buildEntry());
  }

  @Test
  public void testNonRegistryOverride_withRemoval() throws Exception {
    // Original (non-resolved) dep graph
    // archive_override "file://users/user/B.zip"
    // A    -> B 1.0        -> C 1.0 (not loaded)
    // (local) B 1.0-hotfix -> C 1.1
    ImmutableMap<ModuleKey, Module> unprunedDepGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B", createModuleKey("B", ""))
                    .addOriginalDep("B", createModuleKey("B", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .setKey(createModuleKey("B", ""))
                    .addDep("C", createModuleKey("C", "1.1"))
                    .buildEntry())
            .put(ModuleBuilder.create("C", "1.1").buildEntry())
            .buildOrThrow();

    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "B",
            ArchiveOverride.create(
                ImmutableList.of("file://users/user/B.zip"), ImmutableList.of(), "", "", 0));

    ImmutableSet<ModuleKey> usedModules =
        ImmutableSet.of(ModuleKey.ROOT, createModuleKey("B", ""), createModuleKey("C", "1.1"));

    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        BazelModuleInspectorFunction.computeAugmentedGraph(
            unprunedDepGraph, usedModules, overrides);

    assertThat(depGraph.entrySet())
        .containsExactly(
            buildAugmentedModule(ModuleKey.ROOT, "A")
                .addDep("B", "", ResolutionReason.NON_REGISTRY_OVERRIDE)
                .buildEntry(),
            buildAugmentedModule("B", "1.0", false)
                .addOriginalDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule(createModuleKey("B", ""), "B", Version.parse("1.0"), true)
                .addDep("C", "1.1")
                .addDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("C", "1.1").addStillDependant("B", "").buildEntry());
  }

  @Test
  public void testMultipleVersionOverride_simpleSnapToHigher() throws Exception {
    // Initial dep graph
    // A  -> (B1)B 1.0 -> C 1.0
    //   \-> (B2)B 2.0 -> C 1.5
    //   \-> C 2.0
    // multiple_version_override C: [1.5, 2.0]
    // multiple_version_override B: [1.0, 2.0]
    ImmutableMap<ModuleKey, Module> unprunedDepGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B1", createModuleKey("B", "1.0"))
                    .addDep("B2", createModuleKey("B", "2.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "1.0")
                    .addDep("C", createModuleKey("C", "1.5"))
                    .addOriginalDep("C", createModuleKey("C", "1.0"))
                    .buildEntry())
            .put(
                ModuleBuilder.create("B", "2.0")
                    .addDep("C", createModuleKey("C", "1.5"))
                    .buildEntry())
            .put(ModuleBuilder.create("C", "1.0").buildEntry())
            .put(ModuleBuilder.create("C", "1.5").buildEntry())
            .put(ModuleBuilder.create("C", "2.0").buildEntry())
            .buildOrThrow();

    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "B",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""),
            "C",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.5"), Version.parse("2.0")), ""));

    ImmutableSet<ModuleKey> usedModules =
        ImmutableSet.of(
            ModuleKey.ROOT,
            createModuleKey("B", "1.0"),
            createModuleKey("B", "2.0"),
            createModuleKey("C", "1.5"),
            createModuleKey("C", "2.0"));
    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        BazelModuleInspectorFunction.computeAugmentedGraph(
            unprunedDepGraph, usedModules, overrides);

    assertThat(depGraph.entrySet())
        .containsExactly(
            buildAugmentedModule(ModuleKey.ROOT, "A")
                .addDep("B", "1.0")
                .addDep("B", "2.0")
                .addDep("C", "2.0")
                .buildEntry(),
            buildAugmentedModule("B", "1.0")
                .addStillDependant(ModuleKey.ROOT)
                .addDep("C", "1.5", ResolutionReason.MULTIPLE_VERSION_OVERRIDE)
                .buildEntry(),
            buildAugmentedModule("B", "2.0")
                .addStillDependant(ModuleKey.ROOT)
                .addDep("C", "1.5")
                .buildEntry(),
            buildAugmentedModule("C", "1.0").addOriginalDependant("B", "1.0").buildEntry(),
            buildAugmentedModule("C", "1.5")
                .addDependant("B", "1.0")
                .addStillDependant("B", "2.0")
                .buildEntry(),
            buildAugmentedModule("C", "2.0").addStillDependant(ModuleKey.ROOT).buildEntry());
  }

  @Test
  public void testMultipleVersionOverride_badDepsUnreferenced() throws Exception {
    // Initial dep graph
    // A --> B1@1.0 --> C@1.0  [allowed]
    //   \          \-> B2@1.1
    //   \-> B2@1.0 --> C@1.5
    //   \-> B3@1.0 --> C@2.0  [allowed]
    //   \          \-> B4@1.1
    //   \-> B4@1.0 --> C@3.0
    //
    // Resolved dep graph
    // A --> B1@1.0 --> C@1.0  [allowed]
    //   \          \-> B2@1.1
    //   \-> B2@1.1
    //   \-> B3@1.0 --> C@2.0  [allowed]
    //   \          \-> B4@1.1
    //   \-> B4@1.1
    // C@1.5 and C@3.0, the versions violating the allowlist, are gone.
    ImmutableMap<ModuleKey, Module> unprunedDepGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleBuilder.create("A", Version.EMPTY)
                    .setKey(ModuleKey.ROOT)
                    .addDep("B1", createModuleKey("B1", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.1"))
                    .addOriginalDep("B2", createModuleKey("B2", "1.0"))
                    .addDep("B3", createModuleKey("B3", "1.0"))
                    .addDep("B4", createModuleKey("B4", "1.1"))
                    .addOriginalDep("B4", createModuleKey("B4", "1.0"))
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

    ImmutableSet<ModuleKey> usedModules =
        ImmutableSet.of(
            ModuleKey.ROOT,
            createModuleKey("B1", "1.0"),
            createModuleKey("B2", "1.1"),
            createModuleKey("B3", "1.0"),
            createModuleKey("B4", "1.1"),
            createModuleKey("C", "1.0"),
            createModuleKey("C", "2.0"));
    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        BazelModuleInspectorFunction.computeAugmentedGraph(
            unprunedDepGraph, usedModules, overrides);

    assertThat(depGraph.entrySet())
        .containsExactly(
            buildAugmentedModule(ModuleKey.ROOT, "A")
                .addDep("B1", "1.0")
                .addDep("B2", "1.1", ResolutionReason.MINIMAL_VERSION_SELECTION)
                .addDep("B3", "1.0")
                .addDep("B4", "1.1", ResolutionReason.MINIMAL_VERSION_SELECTION)
                .buildEntry(),
            buildAugmentedModule("B1", "1.0")
                .addDep("C", "1.0")
                .addDep("B2", "1.1")
                .addStillDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("B2", "1.0")
                .addDep("C", "1.5")
                .addOriginalDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("B2", "1.1")
                .addDependant(ModuleKey.ROOT)
                .addStillDependant("B1", "1.0")
                .buildEntry(),
            buildAugmentedModule("B3", "1.0")
                .addDep("C", "2.0")
                .addDep("B4", "1.1")
                .addStillDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("B4", "1.0")
                .addDep("C", "3.0")
                .addOriginalDependant(ModuleKey.ROOT)
                .buildEntry(),
            buildAugmentedModule("B4", "1.1")
                .addDependant(ModuleKey.ROOT)
                .addStillDependant("B3", "1.0")
                .buildEntry(),
            buildAugmentedModule("C", "1.0").addStillDependant("B1", "1.0").buildEntry(),
            buildAugmentedModule("C", "1.5").addOriginalDependant("B2", "1.0").buildEntry(),
            buildAugmentedModule("C", "2.0").addStillDependant("B3", "1.0").buildEntry(),
            buildAugmentedModule("C", "3.0").addOriginalDependant("B4", "1.0").buildEntry());
  }

  static ModuleAugmentBuilder buildAugmentedModule(
      ModuleKey key, String name, Version version, boolean loaded) {
    ModuleAugmentBuilder myBuilder = new ModuleAugmentBuilder();
    myBuilder.key = key;
    myBuilder.builder =
        AugmentedModule.builder(key).setName(name).setVersion(version).setLoaded(loaded);
    return myBuilder;
  }

  static ModuleAugmentBuilder buildAugmentedModule(String name, String version, boolean loaded)
      throws ParseException {
    ModuleKey key = createModuleKey(name, version);
    return buildAugmentedModule(key, name, Version.parse(version), loaded);
  }

  static ModuleAugmentBuilder buildAugmentedModule(String name, String version)
      throws ParseException {
    ModuleKey key = createModuleKey(name, version);
    return buildAugmentedModule(key, name, Version.parse(version), true);
  }

  static ModuleAugmentBuilder buildAugmentedModule(ModuleKey key, String name) {
    return buildAugmentedModule(key, name, key.getVersion(), true);
  }

  private static final class ModuleAugmentBuilder {

    private AugmentedModule.Builder builder;
    private ModuleKey key;

    private ModuleAugmentBuilder() {}

    ModuleAugmentBuilder addDep(String name, String version, ResolutionReason reason) {
      this.builder.addDep(createModuleKey(name, version), reason);
      return this;
    }

    ModuleAugmentBuilder addDep(String name, String version) {
      this.builder.addDep(createModuleKey(name, version), ResolutionReason.ORIGINAL);
      return this;
    }

    ModuleAugmentBuilder addDependant(String name, String version) {
      this.builder.addDependant(createModuleKey(name, version));
      return this;
    }

    ModuleAugmentBuilder addDependant(ModuleKey key) {
      this.builder.addDependant(key);
      return this;
    }

    ModuleAugmentBuilder addOriginalDependant(String name, String version) {
      this.builder.addOriginalDependant(createModuleKey(name, version));
      return this;
    }

    ModuleAugmentBuilder addOriginalDependant(ModuleKey key) {
      this.builder.addOriginalDependant(key);
      return this;
    }

    ModuleAugmentBuilder addStillDependant(String name, String version) {
      this.builder.addOriginalDependant(createModuleKey(name, version));
      this.builder.addDependant(createModuleKey(name, version));
      return this;
    }

    ModuleAugmentBuilder addStillDependant(ModuleKey key) {
      this.builder.addOriginalDependant(key);
      this.builder.addDependant(key);
      return this;
    }

    Entry<ModuleKey, AugmentedModule> buildEntry() {
      return new SimpleEntry<>(this.key, this.builder.build());
    }
  }
}
