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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.truth.Correspondence;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.rules.genrule.GenRuleAction;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for cc_binary with FDO. */
@RunWith(JUnit4.class)
public class CcBinaryFdoTest extends BuildViewTestCase {
  private static final Correspondence<String, String> MATCHES_REGEX =
      Correspondence.from((a, b) -> Pattern.matches(b, a), "matches");

  @Test
  public void testActionGraph() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(mockToolsConfig, CcToolchainConfig.builder());
    useConfiguration("--fdo_profile=//:mock_profile", "--compilation_mode=opt");

    scratch.file("binary.cc", "int main() { return 0; }");
    scratch.file(
        "BUILD",
        "genrule(name = 'generate-mock-profraw',",
        "    outs = ['mock.profraw'],",
        "    cmd = 'touch $@',",
        ")",
        "",
        "fdo_profile(name = 'mock_profile',",
        "     profile = 'mock.profraw',",
        ")",
        "",
        "cc_binary(name = 'binary',",
        "    srcs = ['binary.cc'],",
        ")");

    // Check the compile action uses a profdata file
    CppCompileAction compileAction =
        (CppCompileAction)
            getGeneratingAction(
                getBinArtifact("_objs/binary/binary.o", getConfiguredTarget("//:binary")));
    assertThat(compileAction).isNotNull();
    assertThat(compileAction.getArguments())
        .comparingElementsUsing(MATCHES_REGEX)
        .contains("-fprofile-use=bl?azel?-out/k8-opt/bin/.*/mock.profdata");
    Artifact profData =
        ActionsTestUtil.getFirstArtifactEndingWith(compileAction.getInputs(), ".profdata");
    assertThat(profData).isNotNull();

    // Get the action which generates the profdata file from the profraw file
    SpawnAction profDataAction = (SpawnAction) getGeneratingAction(profData);
    assertThat(profDataAction).isNotNull();
    Artifact profRawSymlink =
        ActionsTestUtil.getFirstArtifactEndingWith(profDataAction.getInputs(), ".profraw");
    assertThat(profRawSymlink).isNotNull();

    // Make sure the profData action is from the genrule
    SymlinkAction profRawSymlinkAction = (SymlinkAction) getGeneratingAction(profRawSymlink);
    assertThat(profRawSymlinkAction).isNotNull();
    Artifact profRaw =
        ActionsTestUtil.getFirstArtifactEndingWith(profRawSymlinkAction.getInputs(), ".profraw");
    assertThat(profRaw).isNotNull();

    // Make sure the symlink input is the genrule defined in the BUILD file
    GenRuleAction profRawAction = (GenRuleAction) getGeneratingAction(profRaw);
    assertThat(profRawAction).isNotNull();
    assertThat(profRawAction.getOwner().getLabel().getCanonicalForm())
        .isEqualTo("//:generate-mock-profraw");
  }
}
