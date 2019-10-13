// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidBinary;
import com.google.devtools.build.lib.rules.android.AndroidCommon;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidDataContext;
import com.google.devtools.build.lib.rules.android.AndroidSemantics;
import com.google.devtools.build.lib.rules.android.ProguardHelper.ProguardOutput;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;

/**
 * Implementation of Bazel-specific behavior in Android rules.
 */
public class BazelAndroidSemantics implements AndroidSemantics {
  public static final BazelAndroidSemantics INSTANCE = new BazelAndroidSemantics();

  private BazelAndroidSemantics() {
  }

  @Override
  public String getNativeDepsFileName() {
    return "nativedeps";
  }

  @Override
  public ImmutableList<String> getCompatibleJavacOptions(RuleContext ruleContext) {
    ImmutableList.Builder<String> javacArgs = new ImmutableList.Builder<>();
    if (!ruleContext.getFragment(AndroidConfiguration.class).desugarJava8()) {
      javacArgs.add("-source", "7", "-target", "7");
    }
    return javacArgs.build();
  }

  @Override
  public void addMainDexListActionArguments(
      RuleContext ruleContext,
      SpawnAction.Builder builder,
      CustomCommandLine.Builder commandLine,
      Artifact proguardMap) {}

  @Override
  public ImmutableList<Artifact> getProguardSpecsForManifest(
      AndroidDataContext context, Artifact manifest) {
    return ImmutableList.of();
  }

  @Override
  public void addCoverageSupport(
      RuleContext ruleContext,
      AndroidCommon common,
      JavaSemantics javaSemantics,
      boolean forAndroidTest,
      JavaTargetAttributes.Builder attributes,
      JavaCompilationArtifacts.Builder artifactsBuilder) {}

  @Override
  public ImmutableList<String> getAttributesWithJavaRuntimeDeps(RuleContext ruleContext) {
    switch (ruleContext.getRule().getRuleClass()) {
      case "android_binary":
        return ImmutableList.of("deps");
      default:
        throw new UnsupportedOperationException("Only supported for top-level binaries");
    }
  }

  @Override
  public Artifact getProguardOutputMap(RuleContext ruleContext) throws InterruptedException {
    return ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_MAP);
  }

  /** Bazel does not currently support any dex postprocessing. */
  @Override
  public AndroidBinary.DexPostprocessingOutput postprocessClassesDexZip(
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> filesBuilder,
      Artifact classesDexZip,
      ProguardOutput proguardOutput)
      throws InterruptedException {
    return AndroidBinary.DexPostprocessingOutput.create(classesDexZip, proguardOutput.getMapping());
  }

  @Override
  public void registerMigrationRuleError(RuleContext ruleContext) throws RuleErrorException {
    ruleContext.attributeError(
        "tags",
        "The native Android rules are deprecated. Please use the Starlark Android rules by adding "
            + "the following load statement to the BUILD file: "
            + "load(\"@build_bazel_rules_android//android:rules.bzl\", \""
            + ruleContext.getRule().getRuleClass()
            + "\"). See http://github.com/bazelbuild/rules_android.");
  }
}
