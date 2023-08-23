// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for objc_import. */
@RunWith(JUnit4.class)
public class ObjcImportTest extends ObjcRuleTestCase {
  protected static final RuleType RULE_TYPE =
      new RuleType("objc_import") {
        @Override
        Iterable<String> requiredAttributes(
            Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException {
          List<String> attributes = new ArrayList<>();
          if (!alreadyAdded.contains("archives")) {
            scratch.file(packageDir + "/precomp_library.a");
            attributes.add("archives = ['precomp_library.a']");
          }
          return attributes;
        }
      };

  private void addTrivialImportLibrary() throws IOException {
    scratch.file("imp/precomp_lib.a");
    scratch.file("imp/BUILD",
        "objc_import(",
        "    name = 'imp',",
        "    archives = ['precomp_lib.a'],",
        ")");
  }

  @Test
  public void testImportLibrariesProvidedTransitively() throws Exception {
    scratch.file("imp/this_library.a");
    addTrivialImportLibrary();
    scratch.file("lib/BUILD",
        "objc_library(",
        "    name = 'lib',",
        "    deps = ['//imp:imp'],",
        ")");

    Artifact library =
        ccInfoForTarget("//lib:lib")
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getStaticLibrary();
    assertThat(library.getRunfilesPath().toString()).isEqualTo("imp/precomp_lib.a");
  }

  @Test
  public void testImportLibrariesLinkedToFinalBinary() throws Exception {
    addTrivialImportLibrary();
    addAppleBinaryStarlarkRule(scratch);
    scratch.file(
        "bin/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'ios',",
        "    deps = ['//imp:imp'],",
        ")");
    CommandAction linkBinAction = linkAction("//bin:bin");
    verifyObjlist(linkBinAction, "imp/precomp_lib.a");
    assertThat(Artifact.asExecPaths(linkBinAction.getInputs())).contains("imp/precomp_lib.a");
  }

  @Test
  public void testAlwaysLinkDefaultFalse() throws Exception {
    useConfiguration("--incompatible_objc_alwayslink_by_default=false");
    addTrivialImportLibrary();
    addAppleBinaryStarlarkRule(scratch);
    scratch.file(
        "bin/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'ios',",
        "    deps = ['//imp:imp'],",
        ")");
    CommandAction linkBinAction = linkAction("//bin:bin");
    assertThat(Joiner.on("").join(linkBinAction.getArguments())).doesNotContain("-force_load");
  }

  @Test
  public void testAlwaysLinkDefaultTrue() throws Exception {
    useConfiguration("--incompatible_objc_alwayslink_by_default");
    addTrivialImportLibrary();
    addAppleBinaryStarlarkRule(scratch);
    scratch.file(
        "bin/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'ios',",
        "    deps = ['//imp:imp'],",
        ")");
    CommandAction linkBinAction = linkAction("//bin:bin");
    assertThat(Joiner.on("").join(linkBinAction.getArguments()))
        .contains("-force_load imp/precomp_lib.a");
  }

  @Test
  public void testAlwaysLinkTrueDefaultFalse() throws Exception {
    useConfiguration("--incompatible_objc_alwayslink_by_default=false");
    addAppleBinaryStarlarkRule(scratch);

    scratch.file("imp/precomp_lib.a");
    scratch.file(
        "imp/BUILD",
        "objc_import(",
        "    name = 'imp',",
        "    archives = ['precomp_lib.a'],",
        "    alwayslink = True,",
        ")");
    scratch.file(
        "bin/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'ios',",
        "    deps = ['//imp:imp'],",
        ")");
    CommandAction linkBinAction = linkAction("//bin:bin");
    assertThat(Joiner.on("").join(linkBinAction.getArguments()))
        .contains("-force_load imp/precomp_lib.a");
  }

  @Test
  public void testAlwaysLinkFalseDefaultTrue() throws Exception {
    useConfiguration("--incompatible_objc_alwayslink_by_default");
    addAppleBinaryStarlarkRule(scratch);

    scratch.file("imp/precomp_lib.a");
    scratch.file(
        "imp/BUILD",
        "objc_import(",
        "    name = 'imp',",
        "    archives = ['precomp_lib.a'],",
        "    alwayslink = False,",
        ")");
    scratch.file(
        "bin/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'ios',",
        "    deps = ['//imp:imp'],",
        ")");
    CommandAction linkBinAction = linkAction("//bin:bin");
    assertThat(Joiner.on("").join(linkBinAction.getArguments())).doesNotContain("-force_load");
  }

  @Test
  public void testArchiveRequiresDotInName() throws Exception {
    checkError("x", "x", "'//x:fooa' does not produce any objc_import archives files (expected .a)",
        "objc_import(",
        "    name = 'x',",
        "    archives = ['fooa'],",
        ")");
  }

  @Test
  public void testDylibsProvided() throws Exception {
    scratch.file("imp/imp.a");
    scratch.file("imp/BUILD",
        "objc_import(",
        "    name = 'imp',",
        "    archives = ['imp.a'],",
        "    sdk_dylibs = ['libdy1', 'libdy2'],",
        ")");

    CcLinkingContext ccLinkingContext = ccInfoForTarget("//imp:imp").getCcLinkingContext();
    assertThat(ccLinkingContext.getFlattenedUserLinkFlags()).containsExactly("-ldy1", "-ldy2");
  }

  @Test
  public void testProvidesHdrsAndIncludes() throws Exception {
    checkProvidesHdrsAndIncludes(RULE_TYPE, Optional.absent());
  }

  @Test
  public void testSdkIncludesUsedInCompileActionsOfDependers() throws Exception {
    checkSdkIncludesUsedInCompileActionsOfDependers(RULE_TYPE);
  }

  @Test
  public void testObjcImportLoadedThroughMacro() throws Exception {
    setupTestObjcImportLoadedThroughMacro(/* loadMacro= */ true);
    assertThat(getConfiguredTarget("//a:a")).isNotNull();
    assertNoEvents();
  }

  private void setupTestObjcImportLoadedThroughMacro(boolean loadMacro) throws Exception {
    scratch.file(
        "a/BUILD",
        getAnalysisMock().ccSupport().getMacroLoadStatement(loadMacro, "objc_import"),
        "objc_import(name='a', archives=['a.a'])");
  }

  @Test
  public void testDependency() throws Exception {
    scratch.file("imp/precomp_dep.a");
    scratch.file("imp/precomp_dep.h");
    scratch.file("imp/precomp_lib.a");
    scratch.file(
        "imp/BUILD",
        "objc_import(",
        "    name = 'imp_dep',",
        "    archives = ['precomp_dep.a'],",
        "    hdrs = ['precomp_dep.h'],",
        ")",
        "objc_import(",
        "    name = 'imp',",
        "    archives = ['precomp_lib.a'],",
        "    deps = [':imp_dep'],",
        ")");

    assertThat(getArifactPathsOfLibraries(getConfiguredTarget("//imp:imp")))
        .containsExactly("imp/precomp_lib.a", "imp/precomp_dep.a");
    assertThat(getArifactPathsOfHeaders(getConfiguredTarget("//imp:imp")))
        .containsExactly("imp/precomp_dep.h");
  }
}
