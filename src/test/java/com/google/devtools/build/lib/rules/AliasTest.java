// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext;

import org.junit.Test;

/**
 * Unit tests for the <code>alias</code> rule.
 */
public class AliasTest extends BuildViewTestCase {
  @Test
  public void smoke() throws Exception {
    scratch.file("a/BUILD",
        "cc_library(name='a', srcs=['a.cc'])",
        "alias(name='b', actual='a')");

    ConfiguredTarget b = getConfiguredTarget("//a:b");
    assertThat(b.getProvider(CppCompilationContext.class)).isNotNull();
  }

  @Test
  public void visibilityIsOverriddenAndIsOkay() throws Exception {
    scratch.file("a/BUILD",
        "filegroup(name='a', visibility=['//b:__pkg__'])");
    scratch.file("b/BUILD",
        "alias(name='b', actual='//a:a', visibility=['//visibility:public'])");
    scratch.file("c/BUILD",
        "filegroup(name='c', srcs=['//b:b'])");

    getConfiguredTarget("//c:c");
  }

  @Test
  public void visibilityIsOverriddenAndIsError() throws Exception {
    scratch.file("a/BUILD",
        "filegroup(name='a', visibility=['//visibility:public'])");
    scratch.file("b/BUILD",
        "alias(name='b', actual='//a:a', visibility=['//visibility:private'])");
    scratch.file("c/BUILD",
        "filegroup(name='c', srcs=['//b:b'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//c:c");
    assertContainsEvent(
        "Target '//a:a' is not visible from target '//c:c' (aliased through '//b:b')");
  }

  @Test
  public void visibilityIsOverriddenAndIsErrorAfterMultipleAliases() throws Exception {
    scratch.file("a/BUILD",
        "filegroup(name='a', visibility=['//visibility:public'])");
    scratch.file("b/BUILD",
        "alias(name='b', actual='//a:a', visibility=['//visibility:public'])");
    scratch.file("c/BUILD",
        "alias(name='c', actual='//b:b', visibility=['//visibility:private'])");
    scratch.file("d/BUILD",
        "filegroup(name='d', srcs=['//c:c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//d:d");
    assertContainsEvent(
        "Target '//a:a' is not visible from target '//d:d' (aliased through '//c:c' -> '//b:b')");
  }

  @Test
  public void testAliasCycle() throws Exception {
    scratch.file("a/BUILD",
        "alias(name='a', actual=':b')",
        "alias(name='b', actual=':c')",
        "alias(name='c', actual=':a')",
        "filegroup(name='d', srcs=[':c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:d");
    assertContainsEvent("cycle in dependency graph");
  }

  @Test
  public void testAliasedInvalidDependency() throws Exception {
    scratch.file("a/BUILD",
        "cc_library(name='a', deps=[':b'])",
        "alias(name='b', actual=':c')",
        "filegroup(name='c')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("filegroup rule '//a:c' (aliased through '//a:b') is misplaced here");
  }

  @Test
  public void licensesAreCollected() throws Exception {
    scratch.file("a/BUILD",
        "filegroup(name='a', licenses=['unencumbered'])",
        "alias(name='b', actual=':a', licenses=['restricted'])",
        "filegroup(name='c', srcs=[':b'])");
    useConfiguration("--check_licenses");
    assertThat(
        getConfiguredTarget("//a:c").getProvider(LicensesProvider.class).getTransitiveLicenses())
        .hasSize(2);
  }

  @Test
  public void passesTargetTypeCheck() throws Exception {
    scratch.file("a/BUILD",
        "cc_library(name='a', srcs=['a.cc'], deps=[':b'])",
        "alias(name='b', actual=':c')",
        "cc_library(name='c', srcs=['c.cc'])");

    getConfiguredTarget("//a:a");
  }

  @Test
  public void packageGroupInAlias() throws Exception {
    scratch.file("a/BUILD",
        "package_group(name='a', packages=['//a'])",
        "alias(name='b', actual=':a')",
        "filegroup(name='c', srcs=[':b'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:c");
    assertContainsEvent(
        "in actual attribute of alias rule //a:b: package group '//a:a' is misplaced here");
  }

  @Test
  public void aliasedFile() throws Exception {
    scratch.file("a/BUILD",
        "exports_files(['a'])",
        "alias(name='b', actual='a')",
        "filegroup(name='c', srcs=[':b'])");

    ConfiguredTarget c = getConfiguredTarget("//a:c");
    assertThat(ActionsTestUtil.baseArtifactNames(
        c.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("a");
  }

  @Test
  public void aliasedConfigSetting() throws Exception {
    scratch.file("a/BUILD",
        "filegroup(name='a', srcs=select({':b': ['f1'], '//conditions:default': ['f2']}))",
        "alias(name='b', actual=':c')",
        "config_setting(name='c', values={'define': 'foo=bar'})");

    useConfiguration("--define=foo=bar");
    getConfiguredTarget("//a");
  }

  @Test
  public void aliasedTestSuiteDep() throws Exception {
    scratch.file("a/BUILD",
        "sh_test(name='a', srcs=['a.sh'])");
    scratch.file("b/BUILD",
        "alias(name='b', actual='//a:a', testonly=1)",
        "test_suite(name='c', tests=[':b'])");

    ConfiguredTarget c = getConfiguredTarget("//b:c");
    NestedSet<Artifact> runfiles =
        c.getProvider(RunfilesProvider.class).getDataRunfiles().getAllArtifacts();
    assertThat(ActionsTestUtil.baseArtifactNames(runfiles)).contains("a.sh");
  }

  @Test
  public void testRedirectChasing() throws Exception {
    String toolsRepository = ruleClassProvider.getToolsRepository();
    scratch.file("a/BUILD",
        "alias(name='cc', actual='" + toolsRepository + "//tools/cpp:toolchain')",
        "cc_library(name='a', srcs=['a.cc'])");

    useConfiguration("--crosstool_top=//a:cc");
    getConfiguredTarget("//a:a");
  }
}
