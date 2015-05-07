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

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.FileType;

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

    ImmutableList.Builder<Substitution> substitutions = new ImmutableList.Builder<Substitution>()
        .add(Substitution.of("%(test_app_ipa)s", testIpa.getRootRelativePathString()))
        .add(Substitution.of("%(test_app_name)s", baseNameWithoutIpa(testIpa)))

        .add(Substitution.of("%(xctest_app_ipa)s", xctestIpa.getRootRelativePathString()))
        .add(Substitution.of("%(xctest_app_name)s", baseNameWithoutIpa(xctestIpa)))

        .add(Substitution.of("%(iossim_path)s", iossim().getRootRelativePath().getPathString()))
        .add(Substitution.of("%(plugin_jars)s", Artifact.joinRootRelativePaths(":", plugins())))

        .addAll(deviceSubstitutions().getSubstitutionsForTestRunnerScript());

    Optional<Artifact> testRunner = testRunner();
    if (testRunner.isPresent()) {
      substitutions.add(
          Substitution.of("%(testrunner_binary)s", testRunner.get().getRootRelativePathString()));
    }

    Artifact template = ruleContext.getPrerequisiteArtifact("$test_template", Mode.TARGET);

    ruleContext.registerAction(new TemplateExpansionAction(ruleContext.getActionOwner(),
        template, generatedTestScript(), substitutions.build(), /*executable=*/true));
  }

  private IosTestSubstitutionProvider deviceSubstitutions() {
    return ruleContext.getPrerequisite(
        "target_device", Mode.TARGET, IosTestSubstitutionProvider.class);
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
   * Gets the binary of the testrunner attribute, if there is one.
   */
  private Optional<Artifact> testRunner() {
    return Optional.fromNullable(ruleContext.getPrerequisiteArtifact("$test_runner", Mode.TARGET));
  }

  /**
   * Adds all files needed to run this test to the passed Runfiles builder.
   */
  TestSupport addRunfiles(Runfiles.Builder runfilesBuilder) {
    runfilesBuilder
        .addArtifact(testIpa())
        .addArtifact(xctestIpa())
        .addArtifact(generatedTestScript())
        .addArtifact(iossim())
        .addTransitiveArtifacts(deviceRunfiles())
        .addTransitiveArtifacts(plugins())
        .addArtifacts(testRunner().asSet());
    return this;
  }

  /**
   * Jar files for plugins to the test runner. May be empty.
   */
  private NestedSet<Artifact> plugins() {
    return PrerequisiteArtifacts.nestedSet(ruleContext, "plugins", Mode.TARGET);
  }

  /**
   * Runfiles required in order to use the specified target device.
   */
  private NestedSet<Artifact> deviceRunfiles() {
    return ruleContext.getPrerequisite("target_device", Mode.TARGET, RunfilesProvider.class)
        .getDefaultRunfiles().getAllArtifacts();
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
