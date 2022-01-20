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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createRepositoryMapping;

import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BazelModuleResolutionFunction}. */
@RunWith(JUnit4.class)
public class BazelModuleResolutionFunctionTest {
  @Test
  public void createValue_basic() throws Exception {
    // Root depends on dep@1.0 and dep@2.0 at the same time with a multiple-version override.
    // Root also depends on rules_cc as a normal dep.
    // dep@1.0 depends on rules_java, which is overridden by a non-registry override (see below).
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleKey.ROOT,
                Module.builder()
                    .setName("my_root")
                    .setVersion(Version.parse("1.0"))
                    .setKey(ModuleKey.ROOT)
                    .addDep("my_dep_1", createModuleKey("dep", "1.0"))
                    .addDep("my_dep_2", createModuleKey("dep", "2.0"))
                    .addDep("rules_cc", createModuleKey("rules_cc", "1.0"))
                    .build())
            .put(
                createModuleKey("dep", "1.0"),
                Module.builder()
                    .setName("dep")
                    .setVersion(Version.parse("1.0"))
                    .setKey(createModuleKey("dep", "1.0"))
                    .addDep("rules_java", createModuleKey("rules_java", ""))
                    .build())
            .put(
                createModuleKey("dep", "2.0"),
                Module.builder()
                    .setName("dep")
                    .setVersion(Version.parse("2.0"))
                    .setKey(createModuleKey("dep", "2.0"))
                    .build())
            .put(
                createModuleKey("rules_cc", "1.0"),
                Module.builder()
                    .setName("rules_cc")
                    .setVersion(Version.parse("1.0"))
                    .setKey(createModuleKey("rules_cc", "1.0"))
                    .build())
            .put(
                createModuleKey("rules_java", ""),
                Module.builder()
                    .setName("rules_java")
                    .setVersion(Version.parse("1.0"))
                    .setKey(createModuleKey("rules_java", ""))
                    .build())
            .build();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "dep",
                MultipleVersionOverride.create(
                    ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""),
            "rules_java", LocalPathOverride.create("bleh"));

    BazelModuleResolutionValue value =
        BazelModuleResolutionFunction.createValue(depGraph, overrides);
    assertThat(value.getCanonicalRepoNameLookup())
        .containsExactly(
            "", ModuleKey.ROOT,
            "dep.1.0", createModuleKey("dep", "1.0"),
            "dep.2.0", createModuleKey("dep", "2.0"),
            "rules_cc.1.0", createModuleKey("rules_cc", "1.0"),
            "rules_java.override", createModuleKey("rules_java", ""));
    assertThat(value.getModuleNameLookup())
        .containsExactly(
            "rules_cc", createModuleKey("rules_cc", "1.0"),
            "rules_java", createModuleKey("rules_java", ""));
    assertThat(value.getAbridgedModules())
        .containsExactlyElementsIn(
            depGraph.values().stream().map(AbridgedModule::from).collect(toImmutableList()));
  }

  private static ModuleExtensionUsage createModuleExtensionUsage(
      String bzlFile, String name, String... imports) {
    ImmutableBiMap.Builder<String, String> importsBuilder = ImmutableBiMap.builder();
    for (int i = 0; i < imports.length; i += 2) {
      importsBuilder.put(imports[i], imports[i + 1]);
    }
    return ModuleExtensionUsage.builder()
        .setExtensionBzlFile(bzlFile)
        .setExtensionName(name)
        .setImports(importsBuilder.buildOrThrow())
        .setLocation(Location.BUILTIN)
        .build();
  }

  @Test
  public void createValue_moduleExtensions() throws Exception {
    Module root =
        Module.builder()
            .setName("root")
            .setVersion(Version.parse("1.0"))
            .setKey(ModuleKey.ROOT)
            .addDep("rje", createModuleKey("rules_jvm_external", "1.0"))
            .addDep("rpy", createModuleKey("rules_python", "2.0"))
            .addExtensionUsage(
                createModuleExtensionUsage("@rje//:defs.bzl", "maven", "av", "autovalue"))
            .addExtensionUsage(
                createModuleExtensionUsage("@rpy//:defs.bzl", "pip", "numpy", "numpy"))
            .build();
    ModuleKey depKey = createModuleKey("dep", "2.0");
    Module dep =
        Module.builder()
            .setName("dep")
            .setVersion(Version.parse("2.0"))
            .setKey(depKey)
            .addDep("rules_python", createModuleKey("rules_python", "2.0"))
            .addExtensionUsage(
                createModuleExtensionUsage("@rules_python//:defs.bzl", "pip", "np", "numpy"))
            .addExtensionUsage(
                createModuleExtensionUsage("//:defs.bzl", "myext", "oneext", "myext"))
            .addExtensionUsage(
                createModuleExtensionUsage("//incredible:conflict.bzl", "myext", "twoext", "myext"))
            .build();
    ImmutableMap<ModuleKey, Module> depGraph = ImmutableMap.of(ModuleKey.ROOT, root, depKey, dep);

    ModuleExtensionId maven =
        ModuleExtensionId.create(
            Label.parseAbsoluteUnchecked("@rules_jvm_external.1.0//:defs.bzl"), "maven");
    ModuleExtensionId pip =
        ModuleExtensionId.create(
            Label.parseAbsoluteUnchecked("@rules_python.2.0//:defs.bzl"), "pip");
    ModuleExtensionId myext =
        ModuleExtensionId.create(Label.parseAbsoluteUnchecked("@dep.2.0//:defs.bzl"), "myext");
    ModuleExtensionId myext2 =
        ModuleExtensionId.create(
            Label.parseAbsoluteUnchecked("@dep.2.0//incredible:conflict.bzl"), "myext");

    BazelModuleResolutionValue value =
        BazelModuleResolutionFunction.createValue(depGraph, ImmutableMap.of());
    assertThat(value.getExtensionUsagesTable()).hasSize(5);
    assertThat(value.getExtensionUsagesTable())
        .containsCell(maven, ModuleKey.ROOT, root.getExtensionUsages().get(0));
    assertThat(value.getExtensionUsagesTable())
        .containsCell(pip, ModuleKey.ROOT, root.getExtensionUsages().get(1));
    assertThat(value.getExtensionUsagesTable())
        .containsCell(pip, depKey, dep.getExtensionUsages().get(0));
    assertThat(value.getExtensionUsagesTable())
        .containsCell(myext, depKey, dep.getExtensionUsages().get(1));
    assertThat(value.getExtensionUsagesTable())
        .containsCell(myext2, depKey, dep.getExtensionUsages().get(2));

    assertThat(value.getExtensionUniqueNames())
        .containsExactly(
            maven, "rules_jvm_external.1.0.maven",
            pip, "rules_python.2.0.pip",
            myext, "dep.2.0.myext",
            myext2, "dep.2.0.myext2");

    assertThat(value.getFullRepoMapping(ModuleKey.ROOT))
        .isEqualTo(
            createRepositoryMapping(
                ModuleKey.ROOT,
                "",
                "",
                "root",
                "",
                "rje",
                "rules_jvm_external.1.0",
                "rpy",
                "rules_python.2.0",
                "av",
                "rules_jvm_external.1.0.maven.autovalue",
                "numpy",
                "rules_python.2.0.pip.numpy"));
    assertThat(value.getFullRepoMapping(depKey))
        .isEqualTo(
            createRepositoryMapping(
                depKey,
                "dep",
                "dep.2.0",
                "rules_python",
                "rules_python.2.0",
                "np",
                "rules_python.2.0.pip.numpy",
                "oneext",
                "dep.2.0.myext.myext",
                "twoext",
                "dep.2.0.myext2.myext"));
  }
}
