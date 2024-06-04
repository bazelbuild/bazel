// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class Cpp20ModulesConfiguredTargetTest extends BuildViewTestCase {
  void useFeatures(String... args) throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, Crosstool.CcToolchainConfig.builder().withFeatures(args));
  }

  @Test
  public void testCpp20ModulesCcBinaryConfigurationNoFlags() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_library(
            name = 'lib',
            module_interfaces = ["foo.cppm"],
        )
        """);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//foo:lib");
    assertContainsEvent("requires --experimental_cpp20_modules");
  }

  @Test
  public void testCpp20ModulesCcLibraryConfigurationNoFlags() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_binary(
            name = 'bin',
            module_interfaces = ["foo.cppm"],
        )
        """);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//foo:bin");
    assertContainsEvent("requires --experimental_cpp20_modules");
  }

  @Test
  public void testCpp20ModulesCcTestConfigurationNoFlags() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_test(
            name = 'test',
            module_interfaces = ["foo.cppm"],
        )
        """);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//foo:test");
    assertContainsEvent("requires --experimental_cpp20_modules");
  }

  @Test
  public void testCpp20ModulesCcLibraryConfigurationNoFeatures() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_library(
            name = 'lib',
            module_interfaces = ["foo.cppm"],
        )
        """);
    useConfiguration("--experimental_cpp20_modules");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//foo:lib");
    assertDoesNotContainEvent("requires --experimental_cpp20_modules");
    assertContainsEvent("the feature cpp20_modules must be enabled");
  }

  @Test
  public void testCpp20ModulesCcBinaryConfigurationNoFeatures() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_binary(
            name = 'bin',
            module_interfaces = ["foo.cppm"],
        )
        """);
    useConfiguration("--experimental_cpp20_modules");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//foo:bin");
    assertDoesNotContainEvent("requires --experimental_cpp20_modules");
    assertContainsEvent("the feature cpp20_modules must be enabled");
  }

  @Test
  public void testCpp20ModulesCcTestConfigurationNoFeatures() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_test(
            name = 'test',
            module_interfaces = ["foo.cppm"],
        )
        """);
    useConfiguration("--experimental_cpp20_modules");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//foo:test");
    assertDoesNotContainEvent("requires --experimental_cpp20_modules");
    assertContainsEvent("the feature cpp20_modules must be enabled");
  }

  @Test
  public void testCpp20ModulesCcLibraryConfigurationWithFeatures() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_library(
            name = 'lib',
            module_interfaces = ["foo.cppm"],
        )
        """);
    useFeatures(CppRuleClasses.CPP20_MODULES);
    useConfiguration("--experimental_cpp20_modules", "--features=cpp20_modules");

    ImmutableSet<String> features = getRuleContext(getConfiguredTarget("//foo:lib")).getFeatures();
    assertThat(features).contains("cpp20_modules");
    assertDoesNotContainEvent("requires --experimental_cpp20_modules");
    assertDoesNotContainEvent("the feature cpp20_modules must be enabled");
  }

  @Test
  public void testCpp20ModulesCcBinaryConfigurationWithFeatures() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_binary(
            name = 'bin',
            module_interfaces = ["foo.cppm"],
        )
        """);
    useFeatures(CppRuleClasses.CPP20_MODULES);
    useConfiguration("--experimental_cpp20_modules", "--features=cpp20_modules");

    ImmutableSet<String> features = getRuleContext(getConfiguredTarget("//foo:bin")).getFeatures();
    assertThat(features).contains("cpp20_modules");
    assertDoesNotContainEvent("requires --experimental_cpp20_modules");
    assertDoesNotContainEvent("the feature cpp20_modules must be enabled");
  }

  @Test
  public void testCpp20ModulesCcTestConfigurationWithFeatures() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_test(
            name = 'test',
            module_interfaces = ["foo.cppm"],
        )
        """);
    useFeatures(CppRuleClasses.CPP20_MODULES);
    useConfiguration("--experimental_cpp20_modules", "--features=cpp20_modules");

    ImmutableSet<String> features = getRuleContext(getConfiguredTarget("//foo:test")).getFeatures();
    assertThat(features).contains("cpp20_modules");
    assertDoesNotContainEvent("requires --experimental_cpp20_modules");
    assertDoesNotContainEvent("the feature cpp20_modules must be enabled");
  }

  @Test
  public void testSameModuleInterfacesFileInCcLibraryTwice() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        filegroup(
          name = "a1",
          srcs = ["a.cppm"],
        )
        filegroup(
          name = "a2",
          srcs = ["a.cppm"],
        )
        cc_library(
          name = "lib",
          module_interfaces = ["a1", "a2"],
        )
        """);

    useFeatures(CppRuleClasses.CPP20_MODULES);
    useConfiguration("--experimental_cpp20_modules", "--features=cpp20_modules");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:lib");
    assertContainsEvent("Artifact '<source file a/a.cppm>' is duplicated");
  }

  @Test
  public void testSameModuleInterfacesFileInCcBinaryTwice() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        filegroup(
          name = "a1",
          srcs = ["a.cppm"],
        )
        filegroup(
          name = "a2",
          srcs = ["a.cppm"],
        )
        cc_binary(
          name = "bin",
          module_interfaces = ["a1", "a2"],
        )
        """);

    useFeatures(CppRuleClasses.CPP20_MODULES);
    useConfiguration("--experimental_cpp20_modules", "--features=cpp20_modules");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:bin");
    assertContainsEvent("Artifact '<source file a/a.cppm>' is duplicated");
  }

  @Test
  public void testSameModuleInterfacesFileInCcTestTwice() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        filegroup(
          name = "a1",
          srcs = ["a.cppm"],
        )
        filegroup(
          name = "a2",
          srcs = ["a.cppm"],
        )
        cc_test(
          name = "test",
          module_interfaces = ["a1", "a2"],
        )
        """);

    useFeatures(CppRuleClasses.CPP20_MODULES);
    useConfiguration("--experimental_cpp20_modules", "--features=cpp20_modules");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:test");
    assertContainsEvent("Artifact '<source file a/a.cppm>' is duplicated");
  }
}
