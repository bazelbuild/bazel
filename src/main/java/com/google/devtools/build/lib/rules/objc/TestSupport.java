// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.FileType;

import java.util.List;

/**
 * Support for running XcTests.
 */
class TestSupport {
  private final RuleContext ruleContext;

  TestSupport(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /**
   * Registers actions to create all files needed in order to actually run the test.
   */
  TestSupport registerTestRunnerActionsForSimulator() {
    registerTestScriptSubstitutionAction();
    return this;
  }

  /**
   * Returns the script which should be run in order to actually run the tests.
   */
  Artifact generatedTestScript() {
    return ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, "_test_script");
  }

  private void registerTestScriptSubstitutionAction() {
    // testIpa is the app actually containing the XcTests
    Artifact testIpa = testIpa();
    // xctestIpa is the app bundle being tested
    Artifact xctestIpa = xctestIpa();

    IosDeviceProvider targetDevice = targetDevice();

    List<Substitution> substitutions = ImmutableList.of(
        Substitution.of("%(test_app_ipa)s", testIpa.getRootRelativePathString()),
        Substitution.of("%(test_app_name)s", baseNameWithoutIpa(testIpa)),

        Substitution.of("%(xctest_app_ipa)s", xctestIpa.getRootRelativePathString()),
        Substitution.of("%(xctest_app_name)s", baseNameWithoutIpa(xctestIpa)),

        Substitution.of("%(iossim_path)s", iossim().getRootRelativePath().getPathString()),
        Substitution.of("%(device_type)s", targetDevice.getType()),
        Substitution.of("%(simulator_sdk)s", targetDevice.getIosVersion())
    );

    Artifact template = ruleContext.getPrerequisiteArtifact("$test_template", Mode.TARGET);

    ruleContext.registerAction(new TemplateExpansionAction(ruleContext.getActionOwner(),
        template, generatedTestScript(), substitutions, /*executable=*/true));
  }

  private IosDeviceProvider targetDevice() {
    IosDeviceProvider targetDevice =
        ruleContext.getPrerequisite("target_device", Mode.TARGET, IosDeviceProvider.class);
    if (targetDevice == null) {
      ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
      targetDevice = new IosDeviceProvider.Builder()
          // iPhone 6 should be the default, but 32-bit (i386) simulators don't support the
          // iPhone 6.
          .setType(objcConfiguration.getIosCpu().equals("x86_64") ? "iPhone 6" : "iPhone 5")
          .setIosVersion(objcConfiguration.getIosSimulatorVersion())
          .setLocale("en")
          .build();
    }
    return targetDevice;
  }

  private Artifact testIpa() {
    return ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA);
  }

  private Artifact xctestIpa() {
    FileProvider fileProvider =
        ruleContext.getPrerequisite("xctest_app", Mode.TARGET, FileProvider.class);
    return Iterables.getOnlyElement(
        Artifact.filterFiles(fileProvider.getFilesToBuild(), FileType.of(".ipa")));
  }

  private Artifact iossim() {
    return ruleContext.getPrerequisiteArtifact("$iossim", Mode.HOST);
  }

  /**
   * Adds all files needed to run this test to the passed Runfiles builder.
   */
  TestSupport addRunfiles(Runfiles.Builder runfilesBuilder) {
    runfilesBuilder
        .addArtifact(testIpa())
        .addArtifact(xctestIpa())
        .addArtifact(generatedTestScript())
        .addArtifact(iossim());
    return this;
  }

  /**
   * Adds files which must be built in order to run this test to builder.
   */
  TestSupport addFilesToBuild(NestedSetBuilder<Artifact> builder) {
    builder.add(testIpa()).add(xctestIpa());
    return this;
  }

  /**
   * Returns the base name of the artifact, with the .ipa stuffix stripped.
   */
  private static String baseNameWithoutIpa(Artifact artifact) {
    String baseName = artifact.getExecPath().getBaseName();
    Preconditions.checkState(baseName.endsWith(".ipa"),
        "%s should end in .ipa but doesn't", baseName);
    return baseName.substring(0, baseName.length() - 4);
  }
}
