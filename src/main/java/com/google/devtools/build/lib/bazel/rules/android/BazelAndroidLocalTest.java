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
package com.google.devtools.build.lib.bazel.rules.android;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaSemantics;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestBase;
import com.google.devtools.build.lib.rules.android.AndroidSemantics;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaRuntimeInfo;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.util.ShellEscaper;
import javax.annotation.Nullable;

/** An implementation for the "android_local_test" rule. */
public class BazelAndroidLocalTest extends AndroidLocalTestBase {

  private static final String JACOCO_COVERAGE_RUNNER_MAIN_CLASS =
      "com.google.testing.coverage.JacocoCoverageRunner";

  public BazelAndroidLocalTest() {
    super(BazelAndroidSemantics.INSTANCE);
  }

  @Override
  protected AndroidSemantics createAndroidSemantics() {
    return BazelAndroidSemantics.INSTANCE;
  }

  @Override
  protected JavaSemantics createJavaSemantics() {
    return BazelJavaSemantics.INSTANCE;
  }

  @Override
  protected ImmutableList<String> getJvmFlags(RuleContext ruleContext, String testClass)
      throws RuleErrorException, InterruptedException {
    Artifact androidAllJarsPropertiesFile = getAndroidAllJarsPropertiesFile(ruleContext);

    ImmutableList.Builder<String> builder =
        ImmutableList.<String>builder()
            .addAll(JavaCommon.getJvmFlags(ruleContext))
            .add("-ea")
            .add("-Dbazel.test_suite=" + ShellEscaper.escapeString(testClass))
            .add("-Drobolectric.offline=true")
            .add(
                "-Drobolectric-deps.properties="
                    + androidAllJarsPropertiesFile.getRunfilesPathString())
            .add("-Duse_framework_manifest_parser=true")
            .add("-Dorg.robolectric.packagesToNotAcquire=com.google.testing.junit.runner.util");

    if (JavaRuntimeInfo.from(ruleContext, createJavaSemantics().getJavaRuntimeToolchainType())
            .version()
        >= 17) {
      builder.add("-Djava.security.manager=allow");
    }

    return builder.build();
  }

  @Override
  protected String addCoverageSupport(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      JavaCompilationHelper helper,
      Artifact executable,
      Artifact instrumentationMetadata,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      JavaTargetAttributes.Builder attributesBuilder,
      String mainClass)
      throws RuleErrorException {
    // This method can be called only for *_binary/*_test targets.
    Preconditions.checkNotNull(executable);

    helper.addCoverageSupport();

    // We do not add the instrumented jar to the runtime classpath, but provide it in the shell
    // script via an environment variable.
    return JACOCO_COVERAGE_RUNNER_MAIN_CLASS;
  }

  @Override
  protected TransitiveInfoCollection getAndCheckTestSupport(RuleContext ruleContext) {
    // Add the unit test support to the list of dependencies.
    return Iterables.getOnlyElement(ruleContext.getPrerequisites("$testsupport"));
  }

  @Override
  @Nullable
  // Bazel needs the android-all jars properties file in order for robolectric to
  // run. If it does not find it in the deps of the android_local_test rule, it will
  // throw an error.
  protected Artifact getAndroidAllJarsPropertiesFile(RuleContext ruleContext)
      throws RuleErrorException {
    Iterable<RunfilesProvider> runfilesProviders =
        ruleContext.getPrerequisites("deps", RunfilesProvider.class);
    for (RunfilesProvider runfilesProvider : runfilesProviders) {
      Runfiles dataRunfiles = runfilesProvider.getDataRunfiles();
      for (Artifact artifact : dataRunfiles.getAllArtifacts().toList()) {
        if (artifact.getFilename().equals("robolectric-deps.properties")) {
          return artifact;
        }
      }
    }
    ruleContext.throwWithRuleError(
        "'robolectric-deps.properties' not found in" + " the deps of the rule.");
    return null;
  }
}
