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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Legacy test: These tests test --experimental_objc_crosstool=off. See README.
 */
@RunWith(JUnit4.class)
public class LegacyObjcBinaryTest extends ObjcBinaryTest {
  @Override
  protected ObjcCrosstoolMode getObjcCrosstoolMode() {
    return ObjcCrosstoolMode.OFF;
  }

  @Test
  public void testLinkActionWithTransitiveCppDependency() throws Exception {
    checkLinkActionWithTransitiveCppDependency(RULE_TYPE, new ExtraLinkArgs());
  }

  @Test
  public void testCompilesSourcesWithModuleMapsEnabled() throws Exception {
    checkCompilesSourcesWithModuleMapsEnabled(RULE_TYPE);
  }

  @Test
  public void testLinkWithFrameworkImportsIncludesFlagsAndInputArtifacts() throws Exception {
    checkLinkWithFrameworkImportsIncludesFlagsAndInputArtifacts(RULE_TYPE);
  }

  @Test
  public void testForceLoadsAlwayslinkTargets() throws Exception {
    checkForceLoadsAlwayslinkTargets(RULE_TYPE, new ExtraLinkArgs());
  }

  @Test
  public void testReceivesTransitivelyPropagatedDefines() throws Exception {
    checkReceivesTransitivelyPropagatedDefines(RULE_TYPE);
  }

  @Test
  public void testSdkIncludesUsedInCompileAction() throws Exception {
    checkSdkIncludesUsedInCompileAction(RULE_TYPE);
  }

  @Override
  @Test
  public void testCreate_debugSymbolActionWithAppleFlag() throws Exception {
    useConfiguration("--apple_generate_dsym");
    RULE_TYPE.scratchTarget(scratch, "srcs", "['a.m']");
    ConfiguredTarget target = getConfiguredTarget("//x:x");

    Artifact artifact = getBinArtifact("x.app.dSYM.temp.zip", target);
    String execPath = artifact.getExecPath().getParentDirectory().toString();
    CommandAction linkAction = (CommandAction) getGeneratingAction(artifact);
    assertThat(Joiner.on(" ").join(linkAction.getArguments()))
        .contains(
            Joiner.on(" ")
                .join(
                    "&&",
                    MOCK_XCRUNWRAPPER_PATH,
                    ObjcRuleClasses.DSYMUTIL,
                    execPath + "/x_bin",
                    "-o",
                    execPath + "/x.app.dSYM.temp",
                    "&&",
                    "zipped_bundle=${PWD}/" + artifact.getExecPathString(),
                    "&&",
                    "cd " + artifact.getExecPathString().replace(".zip", ""),
                    "&&",
                    "/usr/bin/zip -q -r \"${zipped_bundle}\" ."));

    Artifact plistArtifact = getBinArtifact("x.app.dSYM/Contents/Info.plist", target);
    Artifact debugSymbolArtifact =
        getBinArtifact("x.app.dSYM/Contents/Resources/DWARF/x_bin", target);
    SpawnAction plistAction = (SpawnAction) getGeneratingAction(plistArtifact);
    SpawnAction debugSymbolAction = (SpawnAction) getGeneratingAction(debugSymbolArtifact);
    assertThat(debugSymbolAction).isEqualTo(plistAction);

    String dsymUnzipActionArg =
        "unzip -p "
            + execPath
            + "/x.app.dSYM.temp.zip"
            + " Contents/Info.plist > "
            + plistArtifact.getExecPathString()
            + " && unzip -p "
            + execPath
            + "/x.app.dSYM.temp.zip"
            + " Contents/Resources/DWARF/x_bin > "
            + debugSymbolArtifact.getExecPathString();
    assertThat(plistAction.getArguments()).contains(dsymUnzipActionArg);
  }

}
