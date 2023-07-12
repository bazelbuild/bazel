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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.AugmentedModuleBuilder.buildAugmentedModule;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ExtensionArg.ExtensionArgConverter;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.AllVersionsOfModule;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.ApparentRepoName;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.CanonicalRepoName;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.SpecificVersionOfModule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ExtensionArgTest {
  @Test
  public void converter() throws Exception {
    assertThat(ExtensionArgConverter.INSTANCE.convert("<root>//abc:haha.bzl%ext"))
        .isEqualTo(
            ExtensionArg.create(
                SpecificVersionOfModule.create(ModuleKey.ROOT), "//abc:haha.bzl", "ext"));
    assertThat(ExtensionArgConverter.INSTANCE.convert("@@abc//:def.bzl%ghi"))
        .isEqualTo(
            ExtensionArg.create(
                CanonicalRepoName.create(RepositoryName.createUnvalidated("abc")),
                "//:def.bzl",
                "ghi"));
    assertThat(ExtensionArgConverter.INSTANCE.convert("@abc//:def.bzl%ghi"))
        .isEqualTo(ExtensionArg.create(ApparentRepoName.create("abc"), "//:def.bzl", "ghi"));
    assertThat(ExtensionArgConverter.INSTANCE.convert("abc//:def.bzl%ghi"))
        .isEqualTo(ExtensionArg.create(AllVersionsOfModule.create("abc"), "//:def.bzl", "ghi"));
    assertThat(ExtensionArgConverter.INSTANCE.convert("a@b//:def.bzl%ghi"))
        .isEqualTo(
            ExtensionArg.create(
                SpecificVersionOfModule.create(createModuleKey("a", "b")), "//:def.bzl", "ghi"));

    assertThrows(
        OptionsParsingException.class,
        () -> ExtensionArgConverter.INSTANCE.convert("abc@//:def.bzl%ghi"));
    assertThrows(
        OptionsParsingException.class,
        () -> ExtensionArgConverter.INSTANCE.convert("@_abc//:def.bzl%ghi"));
    assertThrows(
        OptionsParsingException.class, () -> ExtensionArgConverter.INSTANCE.convert("abc"));
    assertThrows(
        OptionsParsingException.class, () -> ExtensionArgConverter.INSTANCE.convert("abc%def"));
    assertThrows(
        OptionsParsingException.class,
        () -> ExtensionArgConverter.INSTANCE.convert("@_abc%ghi//def"));
  }

  @Test
  public void resolve_good() throws Exception {
    ModuleKey key = createModuleKey("foo", "1.0");
    ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex =
        ImmutableMap.of("", ImmutableSet.of(ModuleKey.ROOT), "foo", ImmutableSet.of(key));
    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        ImmutableMap.<ModuleKey, AugmentedModule>builder()
            .put(buildAugmentedModule("", "").addDep("fred", "foo", "1.0").buildEntry())
            .put(buildAugmentedModule("foo", "1.0").addStillDependant(ModuleKey.ROOT).buildEntry())
            .buildOrThrow();
    ImmutableBiMap<String, ModuleKey> baseModuleDeps = ImmutableBiMap.of("fred", key);
    ImmutableBiMap<String, ModuleKey> baseModuleUnusedDeps = ImmutableBiMap.of();

    assertThat(
            ExtensionArg.create(SpecificVersionOfModule.create(key), "//:abc.bzl", "def")
                .resolveToExtensionId(modulesIndex, depGraph, baseModuleDeps, baseModuleUnusedDeps))
        .isEqualTo(
            ModuleExtensionId.create(
                Label.parseCanonical("@@foo~1.0//:abc.bzl"), "def", Optional.empty()));
  }

  @Test
  public void resolve_badLabel() throws Exception {
    ModuleKey key = createModuleKey("foo", "1.0");
    ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex =
        ImmutableMap.of("", ImmutableSet.of(ModuleKey.ROOT), "foo", ImmutableSet.of(key));
    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        ImmutableMap.<ModuleKey, AugmentedModule>builder()
            .put(buildAugmentedModule("", "").addDep("fred", "foo", "1.0").buildEntry())
            .put(buildAugmentedModule("foo", "1.0").addStillDependant(ModuleKey.ROOT).buildEntry())
            .buildOrThrow();
    ImmutableBiMap<String, ModuleKey> baseModuleDeps = ImmutableBiMap.of("fred", key);
    ImmutableBiMap<String, ModuleKey> baseModuleUnusedDeps = ImmutableBiMap.of();

    assertThrows(
        InvalidArgumentException.class,
        () ->
            ExtensionArg.create(SpecificVersionOfModule.create(key), "/:def.bzl", "ext")
                .resolveToExtensionId(
                    modulesIndex, depGraph, baseModuleDeps, baseModuleUnusedDeps));
    assertThrows(
        InvalidArgumentException.class,
        () ->
            ExtensionArg.create(SpecificVersionOfModule.create(key), "///////", "ext")
                .resolveToExtensionId(
                    modulesIndex, depGraph, baseModuleDeps, baseModuleUnusedDeps));
  }

  @Test
  public void resolve_noneOrtooManyModules() throws Exception {
    ModuleKey foo1 = createModuleKey("foo", "1.0");
    ModuleKey foo2 = createModuleKey("foo", "2.0");
    ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex =
        ImmutableMap.of("", ImmutableSet.of(ModuleKey.ROOT), "foo", ImmutableSet.of(foo1, foo2));
    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        ImmutableMap.<ModuleKey, AugmentedModule>builder()
            .put(
                buildAugmentedModule("", "")
                    .addDep("foo1", "foo", "1.0")
                    .addDep("foo2", "foo", "2.0")
                    .buildEntry())
            .put(buildAugmentedModule("foo", "1.0").addStillDependant(ModuleKey.ROOT).buildEntry())
            .put(buildAugmentedModule("foo", "2.0").addStillDependant(ModuleKey.ROOT).buildEntry())
            .buildOrThrow();
    ImmutableBiMap<String, ModuleKey> baseModuleDeps =
        ImmutableBiMap.of("foo1", foo1, "foo2", foo2);
    ImmutableBiMap<String, ModuleKey> baseModuleUnusedDeps = ImmutableBiMap.of();

    // Found too many, bad!
    assertThrows(
        InvalidArgumentException.class,
        () ->
            ExtensionArg.create(AllVersionsOfModule.create("foo"), "//:def.bzl", "ext")
                .resolveToExtensionId(
                    modulesIndex, depGraph, baseModuleDeps, baseModuleUnusedDeps));
    // Found none, bad!
    assertThrows(
        InvalidArgumentException.class,
        () ->
            ExtensionArg.create(AllVersionsOfModule.create("bar"), "//:def.bzl", "ext")
                .resolveToExtensionId(
                    modulesIndex, depGraph, baseModuleDeps, baseModuleUnusedDeps));
  }
}
