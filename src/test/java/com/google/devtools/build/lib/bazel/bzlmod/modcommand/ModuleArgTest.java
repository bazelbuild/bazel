// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.AugmentedModuleBuilder.buildAugmentedModule;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createRepositoryMapping;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule.ResolutionReason;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.AllVersionsOfModule;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.ApparentRepoName;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.CanonicalRepoName;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.ModuleArgConverter;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.SpecificVersionOfModule;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ModuleArgTest {
  @Test
  public void converter() throws Exception {
    assertThat(ModuleArgConverter.INSTANCE.convert("<root>"))
        .isEqualTo(SpecificVersionOfModule.create(ModuleKey.ROOT));
    assertThat(ModuleArgConverter.INSTANCE.convert("@@abc"))
        .isEqualTo(CanonicalRepoName.create(RepositoryName.createUnvalidated("abc")));
    assertThat(ModuleArgConverter.INSTANCE.convert("@abc"))
        .isEqualTo(ApparentRepoName.create("abc"));
    assertThat(ModuleArgConverter.INSTANCE.convert("abc"))
        .isEqualTo(AllVersionsOfModule.create("abc"));
    assertThat(ModuleArgConverter.INSTANCE.convert("a@b"))
        .isEqualTo(SpecificVersionOfModule.create(createModuleKey("a", "b")));
    assertThat(ModuleArgConverter.INSTANCE.convert("a@3.1.0-pre"))
        .isEqualTo(SpecificVersionOfModule.create(createModuleKey("a", "3.1.0-pre")));

    assertThrows(OptionsParsingException.class, () -> ModuleArgConverter.INSTANCE.convert("abc@"));
    assertThrows(OptionsParsingException.class, () -> ModuleArgConverter.INSTANCE.convert("@_abc"));
  }

  // For the resolveToX test cases, we build a very, very simple dep graph, where root originally
  // depends on foo@1.0, but it's magically upgraded to foo@2.0. The dependency has an affectionate
  // repo name of "fred".
  ModuleKey foo1 = createModuleKey("foo", "1.0");
  ModuleKey foo2 = createModuleKey("foo", "2.0");
  ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex =
      ImmutableMap.of("", ImmutableSet.of(ModuleKey.ROOT), "foo", ImmutableSet.of(foo1, foo2));
  ImmutableMap<ModuleKey, AugmentedModule> depGraph =
      ImmutableMap.<ModuleKey, AugmentedModule>builder()
          .put(
              buildAugmentedModule("", "")
                  .addChangedDep(
                      "fred", "foo", "2.0", "1.0", ResolutionReason.SINGLE_VERSION_OVERRIDE)
                  .buildEntry())
          .put(buildAugmentedModule("foo", "1.0").addOriginalDependant(ModuleKey.ROOT).buildEntry())
          .put(buildAugmentedModule("foo", "2.0").addStillDependant(ModuleKey.ROOT).buildEntry())
          .buildOrThrow();

  ImmutableMap<ModuleKey, RepositoryName> moduleKeyToCanonicalNames =
      depGraph.keySet().stream()
          .collect(toImmutableMap(k -> k, ModuleKey::getCanonicalRepoNameWithVersion));
  ImmutableBiMap<String, ModuleKey> baseModuleDeps = ImmutableBiMap.of("fred", foo2);
  ImmutableBiMap<String, ModuleKey> baseModuleUnusedDeps = ImmutableBiMap.of("fred", foo1);
  RepositoryMapping rootMapping = createRepositoryMapping(ModuleKey.ROOT, "fred", "foo~2.0");

  public ModuleArgTest() throws Exception {}

  @Test
  public void resolve_specificVersion_good() throws Exception {
    var arg = SpecificVersionOfModule.create(foo2);
    assertThat(
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ false,
                /* warnUnused= */ false))
        .containsExactly(foo2);

    assertThat(
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping))
        .containsExactly("foo@2.0", RepositoryName.create("foo~2.0"));
  }

  @Test
  public void resolve_specificVersion_notFound() throws Exception {
    var arg = SpecificVersionOfModule.create(createModuleKey("foo", "3.0"));
    assertThrows(
        InvalidArgumentException.class,
        () ->
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ true,
                /* warnUnused= */ true));
    assertThrows(
        InvalidArgumentException.class,
        () ->
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping));
  }

  @Test
  public void resolve_specificVersion_unused() throws Exception {
    var arg = SpecificVersionOfModule.create(foo1);
    // Without --include_unused, this doesn't resolve, as foo@1.0 has been replaced by foo@2.0.
    assertThat(
            assertThrows(
                InvalidArgumentException.class,
                () ->
                    arg.resolveToModuleKeys(
                        modulesIndex,
                        depGraph,
                        moduleKeyToCanonicalNames,
                        baseModuleDeps,
                        baseModuleUnusedDeps,
                        /* includeUnused= */ false,
                        /* warnUnused= */ true)))
        .hasMessageThat()
        .contains("--include_unused");
    // With --include_unused, this resolves to foo@1.0.
    assertThat(
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ true,
                /* warnUnused= */ true))
        .containsExactly(foo1);

    // resolving to repo names doesn't care about unused deps.
    assertThrows(
        InvalidArgumentException.class,
        () ->
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping));
  }

  @Test
  public void resolve_allVersions_good() throws Exception {
    var arg = AllVersionsOfModule.create("foo");

    assertThat(
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ false,
                /* warnUnused= */ false))
        .containsExactly(foo2);
    // foo1 is unused, so --include_unused would return that too
    assertThat(
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ true,
                /* warnUnused= */ false))
        .containsExactly(foo2, foo1);

    // resolving to repo names doesn't care about unused deps.
    assertThat(
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping))
        .containsExactly("foo@2.0", RepositoryName.create("foo~2.0"));
  }

  @Test
  public void resolve_allVersions_notFound() throws Exception {
    var arg = AllVersionsOfModule.create("bar");

    assertThrows(
        InvalidArgumentException.class,
        () ->
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ true,
                /* warnUnused= */ true));
    assertThrows(
        InvalidArgumentException.class,
        () ->
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping));
  }

  @Test
  public void resolve_apparentRepoName_good() throws Exception {
    var arg = ApparentRepoName.create("fred");

    assertThat(
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ false,
                /* warnUnused= */ false))
        .containsExactly(foo2);
    // foo1 is unused, so --include_unused would return that too
    assertThat(
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ true,
                /* warnUnused= */ false))
        .containsExactly(foo2, foo1);

    assertThat(
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping))
        .containsExactly("@fred", RepositoryName.create("foo~2.0"));
  }

  @Test
  public void resolve_apparentRepoName_notFound() throws Exception {
    var arg = ApparentRepoName.create("brad");

    assertThrows(
        InvalidArgumentException.class,
        () ->
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ true,
                /* warnUnused= */ true));
    assertThrows(
        InvalidArgumentException.class,
        () ->
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping));
  }

  @Test
  public void resolve_canonicalRepoName_good() throws Exception {
    var arg = CanonicalRepoName.create(foo2.getCanonicalRepoNameWithVersion());

    assertThat(
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ false,
                /* warnUnused= */ false))
        .containsExactly(foo2);

    assertThat(
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping))
        .containsExactly("@@foo~2.0", RepositoryName.create("foo~2.0"));
  }

  @Test
  public void resolve_canonicalRepoName_notFound() throws Exception {
    var arg = CanonicalRepoName.create(RepositoryName.create("bar~1.0"));

    assertThrows(
        InvalidArgumentException.class,
        () ->
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ true,
                /* warnUnused= */ true));
    // The repo need not exist in the "repo -> repo" case.
    assertThat(
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping))
        .containsExactly("@@bar~1.0", RepositoryName.create("bar~1.0"));
  }

  @Test
  public void resolve_canonicalRepoName_unused() throws Exception {
    var arg = CanonicalRepoName.create(foo1.getCanonicalRepoNameWithVersion());

    // Without --include_unused, this doesn't resolve, as foo@1.0 has been replaced by foo@2.0.
    assertThat(
            assertThrows(
                InvalidArgumentException.class,
                () ->
                    arg.resolveToModuleKeys(
                        modulesIndex,
                        depGraph,
                        moduleKeyToCanonicalNames,
                        baseModuleDeps,
                        baseModuleUnusedDeps,
                        /* includeUnused= */ false,
                        /* warnUnused= */ true)))
        .hasMessageThat()
        .contains("--include_unused");
    // With --include_unused, this resolves to foo@1.0.
    assertThat(
            arg.resolveToModuleKeys(
                modulesIndex,
                depGraph,
                moduleKeyToCanonicalNames,
                baseModuleDeps,
                baseModuleUnusedDeps,
                /* includeUnused= */ true,
                /* warnUnused= */ true))
        .containsExactly(foo1);

    // resolving to repo names doesn't care about unused deps.
    assertThat(
            arg.resolveToRepoNames(modulesIndex, depGraph, moduleKeyToCanonicalNames, rootMapping))
        .containsExactly("@@foo~1.0", RepositoryName.create("foo~1.0"));
  }
}
