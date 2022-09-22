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

package com.google.devtools.build.lib.bazel.commands;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.AugmentedModuleBuilder.buildAugmentedModule;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.commands.ModqueryExecutor.ResultNode;
import com.google.devtools.build.lib.bazel.commands.ModqueryExecutor.ResultNode.IsExpanded;
import java.io.StringWriter;
import java.io.Writer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ModqueryExecutor}. */
@RunWith(JUnit4.class)
public class ModqueryExecutorTest {

  private final Writer writer = new StringWriter();

  // Tests for the ModqueryExecutor::expandAndPrune core function.
  //
  // (* In the ASCII graph hints "__>" or "-->" mean a direct edge, while "..>" means an indirect
  // edge. "aaa ..." means module "aaa" is unexpanded.)

  @Test
  public void testExpandFromTargetsFirst() throws ParseException {
    // aaa -> bbb -> ccc -> ddd
    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        new ImmutableMap.Builder<ModuleKey, AugmentedModule>()
            .put(
                buildAugmentedModule(ModuleKey.ROOT, "aaa", Version.parse("1.0"), true)
                    .addDep("bbb", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("bbb", "1.0")
                    .addStillDependant(ModuleKey.ROOT)
                    .addDep("ccc", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("ccc", "1.0")
                    .addStillDependant("bbb", "1.0")
                    .addDep("ddd", "1.0")
                    .buildEntry())
            .put(buildAugmentedModule("ddd", "1.0").addStillDependant("ccc", "1.0").buildEntry())
            .buildOrThrow();

    ModqueryOptions options = ModqueryOptions.getDefaultOptions();
    ModqueryExecutor executor = new ModqueryExecutor(depGraph, options, writer);

    // RESULT:
    // <root> ...> ccc -> ddd
    //       \___> bbb -> ccc ...
    assertThat(
            executor.expandAndPrune(
                ImmutableSet.of(ModuleKey.ROOT, createModuleKey("ccc", "1.0")),
                ImmutableSet.of(),
                false))
        .containsExactly(
            ModuleKey.ROOT,
            ResultNode.builder()
                .addChild(createModuleKey("bbb", "1.0"), IsExpanded.TRUE)
                .addIndirectChild(createModuleKey("ccc", "1.0"), IsExpanded.TRUE)
                .build(),
            createModuleKey("bbb", "1.0"),
            ResultNode.builder().addChild(createModuleKey("ccc", "1.0"), IsExpanded.FALSE).build(),
            createModuleKey("ccc", "1.0"),
            ResultNode.builder().addChild(createModuleKey("ddd", "1.0"), IsExpanded.TRUE).build(),
            createModuleKey("ddd", "1.0"),
            ResultNode.builder().build());
  }

  @Test
  public void testPathsDepth1_containsAllTargetsWithNestedIndirect() throws ParseException {
    // <root> -> bbb -> ccc -> ddd -> eee -> fff -> ggg -> hhh
    //                          ^     /
    //                           \___/
    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        new ImmutableMap.Builder<ModuleKey, AugmentedModule>()
            .put(
                buildAugmentedModule(ModuleKey.ROOT, "aaa", Version.parse("1.0"), true)
                    .addDep("bbb", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("bbb", "1.0")
                    .addStillDependant(ModuleKey.ROOT)
                    .addDep("ccc", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("ccc", "1.0")
                    .addStillDependant("bbb", "1.0")
                    .addDep("ddd", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("ddd", "1.0")
                    .addStillDependant("ccc", "1.0")
                    .addStillDependant("eee", "1.0")
                    .addDep("eee", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("eee", "1.0")
                    .addStillDependant("ddd", "1.0")
                    .addDep("fff", "1.0")
                    .addDep("ddd", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("fff", "1.0")
                    .addStillDependant("eee", "1.0")
                    .addDep("ggg", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("ggg", "1.0")
                    .addStillDependant("fff", "1.0")
                    .addDep("hhh", "1.0")
                    .buildEntry())
            .put(buildAugmentedModule("hhh", "1.0").addStillDependant("ggg", "1.0").buildEntry())
            .buildOrThrow();

    ModqueryOptions options = ModqueryOptions.getDefaultOptions();
    options.cycles = true;
    options.depth = 1;
    ModqueryExecutor executor = new ModqueryExecutor(depGraph, options, writer);
    ImmutableSet<ModuleKey> targets =
        ImmutableSet.of(createModuleKey("eee", "1.0"), createModuleKey("hhh", "1.0"));

    // RESULT:
    // <root> --> bbb ..> ddd --> eee --> ddd (cycle)
    //                               \..> ggg --> hhh
    assertThat(executor.expandAndPrune(ImmutableSet.of(ModuleKey.ROOT), targets, false))
        .containsExactly(
            ModuleKey.ROOT,
            ResultNode.builder().addChild(createModuleKey("bbb", "1.0"), IsExpanded.TRUE).build(),
            createModuleKey("bbb", "1.0"),
            ResultNode.builder()
                .addIndirectChild(createModuleKey("ddd", "1.0"), IsExpanded.TRUE)
                .build(),
            createModuleKey("ddd", "1.0"),
            ResultNode.builder().addChild(createModuleKey("eee", "1.0"), IsExpanded.TRUE).build(),
            createModuleKey("eee", "1.0"),
            ResultNode.builder()
                .setTarget(true)
                .addIndirectChild(createModuleKey("ggg", "1.0"), IsExpanded.TRUE)
                .addChild(createModuleKey("ddd", "1.0"), IsExpanded.FALSE)
                .build(),
            createModuleKey("ggg", "1.0"),
            ResultNode.builder().addChild(createModuleKey("hhh", "1.0"), IsExpanded.TRUE).build(),
            createModuleKey("hhh", "1.0"),
            ResultNode.builder().setTarget(true).build());
  }

  @Test
  public void testPathsDepth1_targetParentIsDirectAndIndirectChild() throws ParseException {
    // <root> --> bbb --> ccc
    //             \       |________
    //              \      V       |
    //               \__> ddd --> eee
    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        new ImmutableMap.Builder<ModuleKey, AugmentedModule>()
            .put(
                buildAugmentedModule(ModuleKey.ROOT, "aaa", Version.parse("1.0"), true)
                    .addDep("bbb", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("bbb", "1.0")
                    .addStillDependant(ModuleKey.ROOT)
                    .addDep("ccc", "1.0")
                    .addDep("ddd", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("ccc", "1.0")
                    .addStillDependant("bbb", "1.0")
                    .addDep("ddd", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("ddd", "1.0")
                    .addStillDependant("bbb", "1.0")
                    .addStillDependant("ccc", "1.0")
                    .addStillDependant("eee", "1.0")
                    .addDep("eee", "1.0")
                    .buildEntry())
            .put(
                buildAugmentedModule("eee", "1.0")
                    .addStillDependant("ddd", "1.0")
                    .addStillDependant("eee", "1.0")
                    .addDep("ddd", "1.0")
                    .buildEntry())
            .buildOrThrow();

    ModqueryOptions options = ModqueryOptions.getDefaultOptions();
    options.cycles = true;
    options.depth = 1;
    ModqueryExecutor executor = new ModqueryExecutor(depGraph, options, writer);
    ImmutableSet<ModuleKey> targets = ImmutableSet.of(createModuleKey("eee", "1.0"));

    // RESULT:
    // <root> --> bbb --- ddd --> eee --> ddd (c)
    //             \
    //              \..> ddd ...
    assertThat(executor.expandAndPrune(ImmutableSet.of(ModuleKey.ROOT), targets, false))
        .containsExactly(
            ModuleKey.ROOT,
            ResultNode.builder().addChild(createModuleKey("bbb", "1.0"), IsExpanded.TRUE).build(),
            createModuleKey("bbb", "1.0"),
            ResultNode.builder()
                .addChild(createModuleKey("ddd", "1.0"), IsExpanded.TRUE)
                .addIndirectChild(createModuleKey("ddd", "1.0"), IsExpanded.FALSE)
                .build(),
            createModuleKey("ddd", "1.0"),
            ResultNode.builder().addChild(createModuleKey("eee", "1.0"), IsExpanded.TRUE).build(),
            createModuleKey("eee", "1.0"),
            ResultNode.builder()
                .setTarget(true)
                .addChild(createModuleKey("ddd", "1.0"), IsExpanded.FALSE)
                .build());
  }

  // TODO(andreisolo): Add more eventual edge-case tests for the #expandAndPrune core method
}
