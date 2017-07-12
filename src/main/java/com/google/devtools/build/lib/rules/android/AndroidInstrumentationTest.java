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
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.test.ExecutionInfoProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;
import javax.annotation.Nullable;

/** An implementation of the {@code android_instrumentation} rule. */
public class AndroidInstrumentationTest implements RuleConfiguredTargetFactory {

  private static final Template ANDROID_INSTRUMENTATION_TEST_STUB_SCRIPT =
      Template.forResource(
          AndroidInstrumentationTest.class, "android_instrumentation_test_template.txt");
  private static final String TEST_SUITE_PROPERTY_NAME_FILE = "test_suite_property_name.txt";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    // The wrapper script that invokes the test entry point.
    Artifact testExecutable = createTestExecutable(ruleContext);

    ImmutableList<TransitiveInfoCollection> runfilesDeps =
        ImmutableList.<TransitiveInfoCollection>builder()
            .addAll(ruleContext.getPrerequisites("instrumentations", Mode.TARGET))
            .addAll(ruleContext.getPrerequisites("fixtures", Mode.TARGET))
            .add(ruleContext.getPrerequisite("target_device", Mode.HOST))
            .add(ruleContext.getPrerequisite("$test_entry_point", Mode.HOST))
            .build();

    Runfiles runfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .addArtifact(testExecutable)
            .addTargets(runfilesDeps, RunfilesProvider.DEFAULT_RUNFILES)
            .addTransitiveArtifacts(AndroidCommon.getSupportApks(ruleContext))
            .addTransitiveArtifacts(getAdb(ruleContext).getFilesToRun())
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
        .add(executableSubstitution("%device_script%", getTargetDevice(ruleContext)))
        .add(executableSubstitution("%test_entry_point%", getTestEntryPoint(ruleContext)))
        .add(artifactListSubstitution("%target_apks%", getTargetApks(ruleContext)))
        .add(
            artifactListSubstitution("%instrumentation_apks%", getInstrumentationApks(ruleContext)))
        .add(artifactListSubstitution("%support_apks%", getAllSupportApks(ruleContext)))
        .add(Substitution.ofSpaceSeparatedMap("%test_args%", getTestArgs(ruleContext)))
        .add(Substitution.ofSpaceSeparatedMap("%fixture_args%", getFixtureArgs(ruleContext)))
        .add(Substitution.ofSpaceSeparatedMap("%log_levels%", getLogLevels(ruleContext)))
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

  private static Substitution artifactListSubstitution(String key, Iterable<Artifact> artifacts) {
    return Substitution.ofSpaceSeparatedList(
        key,
        Streams.stream(artifacts)
            .map(Artifact::getRunfilesPathString)
            .collect(ImmutableList.toImmutableList()));
  }

  /**
   * The target APKs from each {@code android_instrumentation} in the {@code instrumentations}
   * attribute.
   */
  private static Iterable<Artifact> getTargetApks(RuleContext ruleContext) {
    return Iterables.transform(
        ruleContext.getPrerequisites(
            "instrumentations",
            Mode.TARGET,
            AndroidInstrumentationInfoProvider.ANDROID_INSTRUMENTATION_INFO),
        AndroidInstrumentationInfoProvider::getTargetApk);
  }

  /**
   * The instrumentation APKs from each {@code android_instrumentation} in the {@code
   * instrumentations} attribute.
   */
  private static Iterable<Artifact> getInstrumentationApks(RuleContext ruleContext) {
    return Iterables.transform(
        ruleContext.getPrerequisites(
            "instrumentations",
            Mode.TARGET,
            AndroidInstrumentationInfoProvider.ANDROID_INSTRUMENTATION_INFO),
        AndroidInstrumentationInfoProvider::getInstrumentationApk);
  }

  /** The support APKs from the {@code support_apks} and {@code fixtures} attributes. */
  private static NestedSet<Artifact> getAllSupportApks(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> allSupportApks =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(AndroidCommon.getSupportApks(ruleContext));
    for (AndroidDeviceScriptFixtureInfoProvider fixture :
        ruleContext.getPrerequisites(
            "fixtures", Mode.TARGET, AndroidDeviceScriptFixtureInfoProvider.class)) {
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

  /** Map of {@code test_args} for the test runner to make available to test test code. */
  private static ImmutableMap<String, String> getTestArgs(RuleContext ruleContext) {
    return ImmutableMap.copyOf(ruleContext.attributes().get("test_args", Type.STRING_DICT));
  }

  /** Map of {@code fixture_args} for the test runner to pass to the {@code fixtures}. */
  private static ImmutableMap<String, String> getFixtureArgs(RuleContext ruleContext) {
    return ImmutableMap.copyOf(ruleContext.attributes().get("fixture_args", Type.STRING_DICT));
  }

  /** Map of {@code log_levels} to enable before the test run. */
  private static ImmutableMap<String, String> getLogLevels(RuleContext ruleContext) {
    return ImmutableMap.copyOf(ruleContext.attributes().get("log_levels", Type.STRING_DICT));
  }

  private static ImmutableList<Artifact> getDataDeps(RuleContext ruleContext) {
    return ruleContext.getPrerequisiteArtifacts("data", Mode.DATA).list();
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
        "fixtures",
        Mode.TARGET,
        AndroidDeviceScriptFixtureInfoProvider.ANDROID_DEVICE_SCRIPT_FIXTURE_INFO);
  }

  private static String getDeviceBrokerType(RuleContext ruleContext) {
    return ruleContext
        .getPrerequisite("target_device", Mode.HOST, DeviceBrokerTypeProvider.class)
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
          AndroidInstrumentationTest.class, TEST_SUITE_PROPERTY_NAME_FILE);
    } catch (IOException e) {
      ruleContext.throwWithRuleError("Cannot load test suite property name: " + e.getMessage());
      return null;
    }
  }

  /**
   * Propagates the {@link ExecutionInfoProvider} from the {@code android_device} rule in the {@code
   * target_device} attribute.
   *
   * <p>This allows the dependent {@code android_device} rule to specify some requirements on the
   * machine that the {@code android_instrumentation_test} runs on.
   */
  private static ExecutionInfoProvider getExecutionInfoProvider(RuleContext ruleContext) {
    ExecutionInfoProvider executionInfoProvider =
            ruleContext.getPrerequisite(
                "target_device", Mode.HOST, ExecutionInfoProvider.SKYLARK_CONSTRUCTOR);
    ImmutableMap<String, String> executionRequirements =
        (executionInfoProvider != null)
            ? executionInfoProvider.getExecutionInfo()
            : ImmutableMap.of();
    return new ExecutionInfoProvider(executionRequirements);
  }
}
