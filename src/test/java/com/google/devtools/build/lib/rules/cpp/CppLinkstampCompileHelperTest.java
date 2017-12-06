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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that {@link CppLinkstampCompileHelper} creates sane compile actions for linkstamps */
@RunWith(JUnit4.class)
public class CppLinkstampCompileHelperTest extends BuildViewTestCase {

  /** Tests that linkstamp compilation applies expected command line options. */
  @Test
  public void testLinkstampCompileOptionsForExecutable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "builtin_sysroot: '/usr/local/custom-sysroot'");
    useConfiguration();
    scratch.file(
        "x/BUILD",
        "cc_binary(",
        "  name = 'foo',",
        "  deps = ['a'],",
        ")",
        "cc_library(",
        "  name = 'a',",
        "  srcs = [ 'a.cc' ],",
        "  linkstamp = 'ls.cc',",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Artifact executable = getExecutable(target);
    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);

    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);

    List<String> arguments = linkstampCompileAction.getArguments();
    assertThatArgumentsAreValid(
        arguments,
        target.getConfiguration().getFragment(CppConfiguration.class).toString(),
        target.getLabel().getCanonicalForm(),
        executable.getFilename());
  }

  private void assertThatArgumentsAreValid(
      List<String> arguments, String platform, String targetName, String buildTargetNameSuffix) {
    assertThat(arguments).contains("--sysroot=/usr/local/custom-sysroot");
    assertThat(arguments).contains("-include");
    assertThat(arguments).contains("-DG3_VERSION_INFO=\"" + targetName + "\"");
    assertThat(arguments).contains("-DG3_TARGET_NAME=\"" + targetName + "\"");
    assertThat(arguments).contains("-DGPLATFORM=\"" + platform + "\"");
    assertThat(arguments).contains("-I.");
    String correctG3BuildTargetPattern = "-DG3_BUILD_TARGET=\".*" + buildTargetNameSuffix + "\"";
    assertThat(Iterables.tryFind(arguments, (arg) -> arg.matches(correctG3BuildTargetPattern)))
        .named("in " + arguments + " flag matching " + correctG3BuildTargetPattern)
        .isPresent();
  }

  /** Tests that linkstamp compilation applies expected command line options. */
  @Test
  public void testLinkstampCompileOptionsForSharedLibrary() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "builtin_sysroot: '/usr/local/custom-sysroot'");
    useConfiguration();
    scratch.file(
        "x/BUILD",
        "cc_binary(",
        "  name = 'libfoo.so',",
        "  deps = ['a'],",
        "  linkshared = 1,",
        ")",
        "cc_library(",
        "  name = 'a',",
        "  srcs = [ 'a.cc' ],",
        "  linkstamp = 'ls.cc',",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//x:libfoo.so");
    Artifact executable = getExecutable(target);
    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);
    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    assertThat(generatingAction.getInputs()).contains(compiledLinkstamp);

    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);

    List<String> arguments = linkstampCompileAction.getArguments();
    assertThatArgumentsAreValid(
        arguments,
        target.getConfiguration().getFragment(CppConfiguration.class).toString(),
        target.getLabel().getCanonicalForm(),
        executable.getFilename());
  }
}
