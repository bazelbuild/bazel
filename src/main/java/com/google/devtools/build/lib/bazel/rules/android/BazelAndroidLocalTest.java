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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaSemantics;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestBase;
import com.google.devtools.build.lib.rules.android.AndroidSdkProvider;
import com.google.devtools.build.lib.rules.android.AndroidSemantics;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts.Builder;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.util.ShellEscaper;

/** An implementation for the "android_local_test" rule. */
public class BazelAndroidLocalTest extends AndroidLocalTestBase {

  Artifact androidAllJarsPropFile;

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
      throws RuleErrorException {
    Artifact androidAllJarsPropertiesFile = getAndroidAllJarsPropertiesFile(ruleContext);

    return ImmutableList.<String>builder()
        .addAll(JavaCommon.getJvmFlags(ruleContext))
        .add("-ea")
        .add("-Dbazel.test_suite=" + ShellEscaper.escapeString(testClass))
        .add("-Drobolectric.offline=true")
        .add(
            "-Drobolectric-deps.properties=" + androidAllJarsPropertiesFile.getRunfilesPathString())
        .add("-Duse_framework_manifest_parser=true")
        .build();
  }

  @Override
  protected String getMainClass(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      JavaCompilationHelper helper,
      Artifact executable,
      Artifact instrumentationMetadata,
      Builder javaArtifactsBuilder,
      JavaTargetAttributes.Builder attributesBuilder)
      throws InterruptedException, RuleErrorException {
    // coverage does not yet work with android_local_test
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      ruleContext.throwWithRuleError("android_local_test does not yet support coverage");
    }
    return "com.google.testing.junit.runner.BazelTestRunner";
  }

  @Override
  protected JavaCompilationHelper getJavaCompilationHelperWithDependencies(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      JavaCommon javaCommon,
      JavaTargetAttributes.Builder javaTargetAttributesBuilder) {

    JavaCompilationHelper javaCompilationHelper =
        new JavaCompilationHelper(
            ruleContext, javaSemantics, javaCommon.getJavacOpts(), javaTargetAttributesBuilder);
    javaCompilationHelper.addLibrariesToAttributes(
        ImmutableList.copyOf(javaCommon.targetsTreatedAsDeps(ClasspathType.COMPILE_ONLY)));

    javaCompilationHelper.addLibrariesToAttributes(
        ImmutableList.of(getAndCheckTestSupport(ruleContext)));

    javaTargetAttributesBuilder.setBootClassPath(
        ImmutableList.of(AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar()));
    javaTargetAttributesBuilder.addRuntimeClassPathEntry(
        AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar());

    return javaCompilationHelper;
  }

  @Override
  protected TransitiveInfoCollection getAndCheckTestSupport(RuleContext ruleContext) {
    // Add the unit test support to the list of dependencies.
    return Iterables.getOnlyElement(ruleContext.getPrerequisites("$testsupport", Mode.TARGET));
  }

  @Override
  // Bazel needs the android-all jars properties file in order for robolectric to
  // run. If it does not find it in the deps of the android_local_test rule, it will
  // throw an error.
  protected Artifact getAndroidAllJarsPropertiesFile(RuleContext ruleContext)
      throws RuleErrorException {
    if (androidAllJarsPropFile == null) {
      androidAllJarsPropFile = getAndroidAllJarsPropertiesFileHelper(ruleContext);
    }
    return androidAllJarsPropFile;
  }

  private Artifact getAndroidAllJarsPropertiesFileHelper(RuleContext ruleContext)
      throws RuleErrorException {
    Iterable<RunfilesProvider> runfilesProviders =
        ruleContext.getPrerequisites("deps", Mode.TARGET, RunfilesProvider.class);
    for (RunfilesProvider runfilesProvider : runfilesProviders) {
      Runfiles dataRunfiles = runfilesProvider.getDataRunfiles();
      for (Artifact artifact : dataRunfiles.getAllArtifacts()) {
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
