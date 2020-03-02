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
package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.Template;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.ExecutionInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/** An implementation of the {@code android_instrumentation_test} rule. */
public class AndroidInstrumentationTestBase implements RuleConfiguredTargetFactory {

  private final AndroidSemantics androidSemantics;

  protected AndroidInstrumentationTestBase(AndroidSemantics androidSemantics) {
    this.androidSemantics = androidSemantics;
  }

  private static final Template ANDROID_INSTRUMENTATION_TEST_STUB_SCRIPT =
      Template.forResource(
          AndroidInstrumentationTestBase.class, "android_instrumentation_test_template.txt");
  private static final String TEST_SUITE_PROPERTY_NAME_FILE = "test_suite_property_name.txt";

  /** Checks expected rule invariants, throws rule errors if anything is set wrong. */
  private static void validateRuleContext(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    if (getInstrumentationProvider(ruleContext) == null) {
      ruleContext.throwWithAttributeError(
          "test_app",
          String.format(
              "The android_binary target %s is missing an 'instruments' attribute. Please set "
                  + "it to the label of the android_binary under test.",
              ruleContext.attributes().get("test_app", BuildType.LABEL)));
    }
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    androidSemantics.checkForMigrationTag(ruleContext);
    validateRuleContext(ruleContext);

    // The wrapper script that invokes the test entry point.
    Artifact testExecutable = createTestExecutable(ruleContext);

    ImmutableList<TransitiveInfoCollection> runfilesDeps =
        ImmutableList.<TransitiveInfoCollection>builder()
            .addAll(ruleContext.getPrerequisites("fixtures", Mode.TARGET))
            .add(ruleContext.getPrerequisite("target_device", Mode.HOST))
            .add(ruleContext.getPrerequisite("$test_entry_point", Mode.HOST))
            .build();

    Runfiles runfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .addArtifact(testExecutable)
            .addArtifact(getInstrumentationApk(ruleContext))
            .addArtifact(getTargetApk(ruleContext))
            .addTargets(runfilesDeps, RunfilesProvider.DEFAULT_RUNFILES)
            .addTransitiveArtifacts(AndroidCommon.getSupportApks(ruleContext))
            .addTransitiveArtifacts(getAdb(ruleContext).getFilesToRun())
            .merge(getAapt(ruleContext).getRunfilesSupport())
            .addArtifacts(getDataDeps(ruleContext))
            .build();

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.<Artifact>stableOrder().add(testExecutable).build())
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .setRunfilesSupport(
            RunfilesSupport.withExecutable(ruleContext, runfiles, testExecutable), testExecutable)
        .addNativeDeclaredProvider(getExecutionInfoProvider(ruleContext))
        .build();
  }

  /** Registers a {@link TemplateExpansionAction} to write the test executable. */
  private Artifact createTestExecutable(RuleContext ruleContext) throws RuleErrorException {
    Artifact testExecutable = ruleContext.createOutputArtifact();
    ruleContext.registerAction(
        new TemplateExpansionAction(
            ruleContext.getActionOwner(),
            testExecutable,
            ANDROID_INSTRUMENTATION_TEST_STUB_SCRIPT,
            getTemplateSubstitutions(ruleContext),
            /* makeExecutable = */ true));
    return testExecutable;
  }

  /**
   * This method defines all substitutions need to fill in {@link
   * #ANDROID_INSTRUMENTATION_TEST_STUB_SCRIPT}.
   */
  private ImmutableList<Substitution> getTemplateSubstitutions(RuleContext ruleContext)
      throws RuleErrorException {
    return ImmutableList.<Substitution>builder()
        .add(Substitution.of("%workspace%", ruleContext.getWorkspaceName()))
        .add(Substitution.of("%test_label%", ruleContext.getLabel().getCanonicalForm()))
        .add(executableSubstitution("%adb%", getAdb(ruleContext)))
        .add(executableSubstitution("%aapt%", getAapt(ruleContext)))
        .add(executableSubstitution("%device_script%", getTargetDevice(ruleContext)))
        .add(executableSubstitution("%test_entry_point%", getTestEntryPoint(ruleContext)))
        .add(artifactSubstitution("%target_apk%", getTargetApk(ruleContext)))
        .add(artifactSubstitution("%instrumentation_apk%", getInstrumentationApk(ruleContext)))
        .add(artifactListSubstitution("%support_apks%", getAllSupportApks(ruleContext).toList()))
        .add(deviceScriptFixturesSubstitution(ruleContext))
        .addAll(hostServiceFixturesSubstitutions(ruleContext))
        .add(artifactListSubstitution("%data_deps%", getDataDeps(ruleContext)))
        .add(Substitution.of("%device_broker_type%", getDeviceBrokerType(ruleContext)))
        .add(Substitution.of("%test_suite_property_name%", getTestSuitePropertyName(ruleContext)))
        .build();
  }

  /**
   * An ad-hoc substitution to put the information from the {@code android_device_script_fixture}s
   * into the bash stub script.
   *
   * <p>TODO(ajmichael): Determine an actual protocol to pass this information to the test suite.
   */
  private static Substitution deviceScriptFixturesSubstitution(RuleContext ruleContext) {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (AndroidDeviceScriptFixtureInfoProvider deviceScriptFixture :
        getDeviceScriptFixtures(ruleContext)) {
      builder.add(
          String.format(
              "[%s]=%b,%b",
              deviceScriptFixture.getFixtureScript().getRunfilesPathString(),
              deviceScriptFixture.getDaemon(),
              deviceScriptFixture.getStrictExit()));
    }
    return Substitution.ofSpaceSeparatedList("%device_script_fixtures%", builder.build());
  }

  /**
   * An ad-hoc substitution to put the information from the {@code android_host_service_fixture}s
   * into the bash stub script.
   *
   * <p>TODO(ajmichael): Determine an actual protocol to pass this information to the test suite.
   */
  private static ImmutableList<Substitution> hostServiceFixturesSubstitutions(
      RuleContext ruleContext) {
    AndroidHostServiceFixtureInfoProvider hostServiceFixture = getHostServiceFixture(ruleContext);
    return ImmutableList.of(
        Substitution.of(
            "%host_service_fixture%",
            hostServiceFixture != null
                ? hostServiceFixture.getExecutable().getRunfilesPathString()
                : ""),
        Substitution.of(
            "%host_service_fixture_services%",
            hostServiceFixture != null
                ? Joiner.on(",").join(hostServiceFixture.getServiceNames())
                : ""));
  }

  private static Substitution executableSubstitution(
      String key, FilesToRunProvider filesToRunProvider) {
    return Substitution.of(key, filesToRunProvider.getExecutable().getRunfilesPathString());
  }

  private static Substitution artifactSubstitution(String key, Artifact artifact) {
    return Substitution.of(key, artifact.getRunfilesPathString());
  }

  private static Substitution artifactListSubstitution(String key, List<Artifact> artifacts) {
    return Substitution.ofSpaceSeparatedList(
        key,
        artifacts.stream()
            .map(Artifact::getRunfilesPathString)
            .collect(ImmutableList.toImmutableList()));
  }

  @Nullable
  private static AndroidInstrumentationInfo getInstrumentationProvider(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(
        "test_app", Mode.TARGET, AndroidInstrumentationInfo.PROVIDER);
  }

  @Nullable
  private static ApkInfo getApkProvider(RuleContext ruleContext) {
    return ruleContext.getPrerequisite("test_app", Mode.TARGET, ApkInfo.PROVIDER);
  }

  /** The target APK from the {@code android_binary} in the {@code instrumentation} attribute. */
  @Nullable
  private static Artifact getTargetApk(RuleContext ruleContext) {
    return getInstrumentationProvider(ruleContext).getTarget().getApk();
  }

  /**
   * The instrumentation APK from the {@code android_binary} in the {@code instrumentation}
   * attribute.
   */
  @Nullable
  private static Artifact getInstrumentationApk(RuleContext ruleContext) {
    return getApkProvider(ruleContext).getApk();
  }

  /** The support APKs from the {@code support_apks} and {@code fixtures} attributes. */
  private static NestedSet<Artifact> getAllSupportApks(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> allSupportApks =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(AndroidCommon.getSupportApks(ruleContext));
    for (AndroidDeviceScriptFixtureInfoProvider fixture :
        ruleContext.getPrerequisites(
            "fixtures", Mode.TARGET, AndroidDeviceScriptFixtureInfoProvider.SKYLARK_CONSTRUCTOR)) {
      allSupportApks.addTransitive(fixture.getSupportApks());
    }
    for (AndroidHostServiceFixtureInfoProvider fixture :
        ruleContext.getPrerequisites(
            "fixtures",
            Mode.TARGET,
            AndroidHostServiceFixtureInfoProvider.ANDROID_HOST_SERVICE_FIXTURE_INFO)) {
      allSupportApks.addTransitive(fixture.getSupportApks());
    }
    return allSupportApks.build();
  }

  /** The deploy jar that interacts with the device. */
  private static FilesToRunProvider getTestEntryPoint(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite("$test_entry_point", Mode.HOST);
  }

  /** The {@code android_device} script to launch an emulator for the test. */
  private static FilesToRunProvider getTargetDevice(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite("target_device", Mode.HOST);
  }

  /** ADB binary from the Android SDK. */
  private static FilesToRunProvider getAdb(RuleContext ruleContext) {
    return AndroidSdkProvider.fromRuleContext(ruleContext).getAdb();
  }

  /** AAPT binary from the Android SDK. */
  private static FilesToRunProvider getAapt(RuleContext ruleContext) {
    return AndroidSdkProvider.fromRuleContext(ruleContext).getAapt();
  }

  private static ImmutableList<Artifact> getDataDeps(RuleContext ruleContext) {
    return ruleContext.getPrerequisiteArtifacts("data", Mode.DONT_CHECK).list();
  }

  /**
   * Checks for a {@code android_host_service_fixture} in the {@code fixtures} attribute. Returns
   * null if there is none, a {@link AndroidHostServiceFixtureInfoProvider} if there is one or
   * throws an error if there is more than one.
   */
  @Nullable
  private static AndroidHostServiceFixtureInfoProvider getHostServiceFixture(
      RuleContext ruleContext) {
    ImmutableList<AndroidHostServiceFixtureInfoProvider> hostServiceFixtures =
        ImmutableList.copyOf(
            ruleContext.getPrerequisites(
                "fixtures",
                Mode.TARGET,
                AndroidHostServiceFixtureInfoProvider.ANDROID_HOST_SERVICE_FIXTURE_INFO));
    if (hostServiceFixtures.size() > 1) {
      ruleContext.ruleError(
          "android_instrumentation_test accepts at most one android_host_service_fixture");
    }
    return Iterables.getFirst(hostServiceFixtures, null);
  }

  private static Iterable<AndroidDeviceScriptFixtureInfoProvider> getDeviceScriptFixtures(
      RuleContext ruleContext) {
    return ruleContext.getPrerequisites(
        "fixtures", Mode.TARGET, AndroidDeviceScriptFixtureInfoProvider.SKYLARK_CONSTRUCTOR);
  }

  private static String getDeviceBrokerType(RuleContext ruleContext) {
    return ruleContext
        .getPrerequisite("target_device", Mode.HOST, AndroidDeviceBrokerInfo.PROVIDER)
        .getDeviceBrokerType();
  }

  /**
   * Returns the name of the test suite property that the test runner uses to determine which test
   * suite to run.
   *
   * <p>This is stored in a separate resource file to facilitate different runners for internal and
   * external Bazel.
   */
  private static String getTestSuitePropertyName(RuleContext ruleContext)
      throws RuleErrorException {
    try {
      return ResourceFileLoader.loadResource(
              AndroidInstrumentationTestBase.class, TEST_SUITE_PROPERTY_NAME_FILE)
          .trim();
    } catch (IOException e) {
      throw ruleContext.throwWithRuleError(
          "Cannot load test suite property name: " + e.getMessage(), e);
    }
  }

  /**
   * Propagates the {@link ExecutionInfo} from the {@code android_device} rule in the {@code
   * target_device} attribute.
   *
   * <p>This allows the dependent {@code android_device} rule to specify some requirements on the
   * machine that the {@code android_instrumentation_test} runs on.
   */
  private static ExecutionInfo getExecutionInfoProvider(RuleContext ruleContext) {
    ExecutionInfo executionInfo =
        ruleContext.getPrerequisite("target_device", Mode.HOST, ExecutionInfo.PROVIDER);
    ImmutableMap<String, String> executionRequirements =
        (executionInfo != null) ? executionInfo.getExecutionInfo() : ImmutableMap.of();
    return new ExecutionInfo(executionRequirements);
  }
}
