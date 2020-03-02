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
import com.google.devtools.build.lib.analysis.LicensesProvider.TargetLicense;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.License.LicenseType;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the <code>alias</code> rule. */
@RunWith(JUnit4.class)
public class AliasTest extends BuildViewTestCase {
  @Test
  public void smoke() throws Exception {
    scratch.file("a/BUILD",
        "cc_library(name='a', srcs=['a.cc'])",
        "alias(name='b', actual='a')");

    ConfiguredTarget b = getConfiguredTarget("//a:b");
    assertThat(b.get(CcInfo.PROVIDER).getCcCompilationContext()).isNotNull();
  }

  @Test
  public void aliasToInputFile() throws Exception {
    scratch.file("a/BUILD",
        "exports_files(['a'])",
        "alias(name='b', actual='a')");

    ConfiguredTarget b = getConfiguredTarget("//a:b");
    assertThat(ActionsTestUtil.baseArtifactNames(getFilesToBuild(b))).containsExactly("a");
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
        "alias '//b:b' referring to target '//a:a' is not visible from target '//c:c'");
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
    assertContainsEvent("alias '//c:c' referring to target '//a:a' through '//b:b' "
        + "is not visible from target '//d:d'");
  }

  @Test
  public void testAliasWithPrivateVisibilityAccessibleFromSamePackage() throws Exception {
    scratch.file("a/BUILD", "exports_files(['af'])");
    scratch.file("b/BUILD",
        "package(default_visibility=['//visibility:private'])",
        "alias(name='al', actual='//a:af')",
        "filegroup(name='ta', srcs=[':al'])");

    getConfiguredTarget("//b:ta");
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
    assertContainsEvent("alias '//a:b' referring to filegroup rule '//a:c' is misplaced here");
  }

  @Test
  public void licensesAreCollected() throws Exception {
    scratch.file("a/BUILD",
        "filegroup(name='a', licenses=['restricted'], output_licenses=['unencumbered'])",
        "alias(name='b', actual=':a')",
        "filegroup(name='c', srcs=[':b'])",
        "genrule(name='d', outs=['do'], tools=[':b'], cmd='cmd')",
        "genrule(name='e', outs=['eo'], srcs=[':b'], cmd='cmd')");
    useConfiguration("--check_licenses");
    assertThat(getLicenses("//a:d", "//a:a")).containsExactly(LicenseType.UNENCUMBERED);
    assertThat(getLicenses("//a:e", "//a:a")).containsExactly(LicenseType.RESTRICTED);
    assertThat(getLicenses("//a:b", "//a:a")).containsExactly(LicenseType.RESTRICTED);
    assertThat(
            getConfiguredTarget("//a:b")
                .getProvider(LicensesProvider.class)
                .getTransitiveLicenses()
                .toList())
        .hasSize(1);
  }

  @Test
  public void assertNoLicensesAttribute() throws Exception {
    scratch.file("a/BUILD",
        "filegroup(name='a')",
        "alias(name='b', actual=':a', licenses=['unencumbered'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:b");
    assertContainsEvent("no such attribute 'licenses' in 'alias' rule");
  }

  private Set<LicenseType> getLicenses(String topLevelTarget, String licenseTarget)
      throws Exception {
    LicensesProvider licenses =
        getConfiguredTarget(topLevelTarget).getProvider(LicensesProvider.class);
    for (TargetLicense license : licenses.getTransitiveLicenses().toList()) {
      if (license.getLabel().toString().equals(licenseTarget)) {
        return license.getLicense().getLicenseTypes();
      }
    }

    throw new IllegalStateException("License for '" + licenseTarget
        + "' not found in the transitive closure of '" + topLevelTarget + "'");
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

  @Test
  public void testNoActual() throws Exception {
    checkError("a", "a", "missing value for mandatory attribute 'actual'", "alias(name='a')");
  }
}
