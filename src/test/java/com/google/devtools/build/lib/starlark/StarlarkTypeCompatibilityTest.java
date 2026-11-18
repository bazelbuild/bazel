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
package com.google.devtools.build.lib.starlark;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.EvaluationResult;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that changes to a dependency's .bzl files that don't break .bzl files loading them without
 * static type checking also don't break them with static type checking.
 */
@RunWith(JUnit4.class)
public final class StarlarkTypeCompatibilityTest extends BuildViewTestCase {

  @Before
  public void setUpMiddleModule() throws Exception {
    registry.addModule(
        createModuleKey("middle", "1.0"),
        "module(name = 'middle', version = '1.0')",
        "bazel_dep(name = 'lib', version = '1.0')");
    Path middleDir = moduleRoot.getRelative("middle+1.0");
    scratch.file(middleDir.getRelative("REPO.bazel").getPathString());
    scratch.file(middleDir.getRelative("BUILD").getPathString());
    scratch.file(middleDir.getRelative("middle.bzl").getPathString());
  }

  /**
   * Asserts that {@code //consumer:consumer.bzl} loads with versions 1.0 and 2.0 of the {@code lib}
   * module, both with and without static type checking.
   */
  private void assertUpdateKeepsConsumerValid(String libV1, String libV2, String consumer)
      throws Exception {
    for (String version : ImmutableList.of("1.0", "2.0")) {
      registry.addModule(
          createModuleKey("lib", version), "module(name = 'lib', version = '" + version + "')");
      Path libDir = moduleRoot.getRelative("lib+" + version);
      scratch.file(libDir.getRelative("REPO.bazel").getPathString());
      scratch.file(libDir.getRelative("BUILD").getPathString());
      scratch.file(
          libDir.getRelative("lib.bzl").getPathString(), version.equals("1.0") ? libV1 : libV2);
    }
    scratch.file("consumer/BUILD");
    scratch.file("consumer/consumer.bzl", consumer);
    reporter.removeHandler(failFastHandler);

    for (boolean staticTypeChecking : new boolean[] {false, true}) {
      setBuildLanguageOptions(
          "--experimental_starlark_type_syntax",
          staticTypeChecking
              ? "--experimental_starlark_static_type_checking"
              : "--noexperimental_starlark_static_type_checking");
      for (String version : ImmutableList.of("1.0", "2.0")) {
        scratch.overwriteFile(
            "MODULE.bazel",
            "bazel_dep(name = 'lib', version = '" + version + "')",
            "bazel_dep(name = 'middle', version = '1.0')");
        invalidatePackages();
        eventCollector.clear();

        BzlLoadValue.Key key =
            BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked("//consumer:consumer.bzl"));
        EvaluationResult<BzlLoadValue> result =
            SkyframeExecutorTestUtils.evaluate(
                getSkyframeExecutor(), key, /* keepGoing= */ false, reporter);

        assertWithMessage(
                "consumer.bzl failed to load with lib %s and static type checking %s, errors: %s",
                version,
                staticTypeChecking ? "enabled" : "disabled",
                Streams.stream(eventCollector)
                    .filter(e -> e.getKind() == EventKind.ERROR)
                    .map(Event::getMessage)
                    .collect(toImmutableList()))
            .that(result.hasError())
            .isFalse();
      }
    }
  }

  @Test
  public void stringBecomesLabel() throws Exception {
    assertUpdateKeepsConsumerValid(
        """
        TOOLCHAIN_TYPE = "@lib//:toolchain_type"
        """,
        """
        TOOLCHAIN_TYPE = Label("//:toolchain_type")
        """,
        """
        load("@lib//:lib.bzl", "TOOLCHAIN_TYPE")

        TOOLCHAIN_TYPES: list[str] = [TOOLCHAIN_TYPE, "@bazel_tools//tools/sh:toolchain_type"]
        """);
  }

  @Test
  public void listElementsBecomeLabels() throws Exception {
    assertUpdateKeepsConsumerValid(
        """
        TOOLCHAINS = ["@lib//:toolchain_type"]
        """,
        """
        TOOLCHAINS = [Label("//:toolchain_type")]
        """,
        """
        load("@lib//:lib.bzl", "TOOLCHAINS")

        MY_TOOLCHAINS: Sequence[str] = TOOLCHAINS
        """);
  }

  @Test
  public void listGainsElementOfNewType() throws Exception {
    assertUpdateKeepsConsumerValid(
        """
        TOOLCHAINS = ["@lib//:optional_toolchain_type", Label("//:toolchain_type")]
        """,
        """
        TOOLCHAINS = [
            config_common.toolchain_type("@lib//:optional_toolchain_type", mandatory = False),
            Label("//:toolchain_type"),
        ]
        """,
        """
        load("@lib//:lib.bzl", "TOOLCHAINS")

        MY_TOOLCHAINS: Sequence[str | Label] = TOOLCHAINS
        """);
  }

  @Test
  public void listBecomesEmpty() throws Exception {
    assertUpdateKeepsConsumerValid(
        """
        RUNTIME_DEPS = ["@lib//:runtime"]
        """,
        """
        RUNTIME_DEPS = []
        """,
        """
        load("@lib//:lib.bzl", "RUNTIME_DEPS")

        MY_RUNTIME_DEPS: list[str] = RUNTIME_DEPS
        """);
  }

  @Test
  public void reexportedListChangesWithTransitiveDependency() throws Exception {
    scratch.overwriteFile(
        moduleRoot.getRelative("middle+1.0/middle.bzl").getPathString(),
        """
        load("@lib//:lib.bzl", "TOOLCHAINS")

        middle_lib = struct(toolchains = ["@middle//:toolchain_type"] + TOOLCHAINS)
        """);
    assertUpdateKeepsConsumerValid(
        """
        TOOLCHAINS = ["@lib//:toolchain_type"]
        """,
        """
        TOOLCHAINS = [Label("//:toolchain_type")]
        """,
        """
        load("@middle//:middle.bzl", "middle_lib")

        MY_TOOLCHAINS: Sequence[str] = middle_lib.toolchains
        """);
  }

  @Test
  public void structFieldRemoved_readGuardedByHasattr() throws Exception {
    assertUpdateKeepsConsumerValid(
        """
        ACTION_NAMES = struct(compile = "compile", archive = "archive")
        """,
        """
        ACTION_NAMES = struct(compile = "compile")
        """,
        """
        load("@lib//:lib.bzl", "ACTION_NAMES")

        COMPILE: str = ACTION_NAMES.compile
        ARCHIVE = ACTION_NAMES.archive if hasattr(ACTION_NAMES, "archive") else None
        """);
  }

  @Test
  public void structFieldAdded_readGuardedByHasattr() throws Exception {
    assertUpdateKeepsConsumerValid(
        """
        ACTION_NAMES = struct(compile = "compile")
        """,
        """
        ACTION_NAMES = struct(compile = "compile", link = "link")
        """,
        """
        load("@lib//:lib.bzl", "ACTION_NAMES")

        COMPILE: str = ACTION_NAMES.compile
        LINK = ACTION_NAMES.link if hasattr(ACTION_NAMES, "link") else None
        """);
  }

  @Test
  public void providerFieldRemoved_readGuardedByHasattr() throws Exception {
    assertUpdateKeepsConsumerValid(
        """
        LibInfo = provider(fields = ["files", "legacy_files"])
        """,
        """
        LibInfo = provider(fields = ["files"])
        """,
        """
        load("@lib//:lib.bzl", "LibInfo")

        def get_legacy_files(info: LibInfo):
            if hasattr(info, "legacy_files"):
                return info.legacy_files
            return None
        """);
  }

  @Test
  public void localVariableReassignedInFunction() throws Exception {
    assertUpdateKeepsConsumerValid(
        """
        DEPS = ["@lib//:dep"]
        """,
        """
        DEPS = [Label("//:dep")]
        """,
        """
        load("@lib//:lib.bzl", "DEPS")

        def get_deps(extra: bool):
            deps = DEPS
            if extra:
                deps = ["@extra//:dep"]
            return deps
        """);
  }
}
