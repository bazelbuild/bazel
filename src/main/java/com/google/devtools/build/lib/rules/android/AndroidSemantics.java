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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.ProguardHelper.ProguardOutput;

/**
 * Pluggable semantics for Android rules.
 *
 * <p>A new instance of this class is created for each configured target, therefore, it is allowed
 * to keep state.
 */
public interface AndroidSemantics {
  /**
   * Returns the manifest to be used when compiling a given rule.
   *
   * @throws InterruptedException
   */
  default ApplicationManifest getManifestForRule(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    Artifact rawManifest = ApplicationManifest.getManifestFromAttributes(ruleContext);
    return ApplicationManifest.fromExplicitManifest(
        ruleContext, renameManifest(ruleContext, rawManifest));
  }

  default Artifact renameManifest(RuleContext ruleContext, Artifact rawManifest)
      throws InterruptedException {
    return ApplicationManifest.renameManifestIfNeeded(ruleContext, rawManifest);
  }

  /** Returns the name of the file in which the file names of native dependencies are listed. */
  String getNativeDepsFileName();

  /**
   * Returns the command line options to be used when compiling Java code for {@code android_*}
   * rules.
   *
   * <p>These will come after the default options specified by the toolchain, and before the ones in
   * the {@code javacopts} attribute.
   */
  ImmutableList<String> getCompatibleJavacOptions(RuleContext ruleContext);

  /**
   * Configures the builder for generating the output jar used to configure the main dex file.
   *
   * @throws InterruptedException
   */
  void addMainDexListActionArguments(
      RuleContext ruleContext,
      SpawnAction.Builder builder,
      CustomCommandLine.Builder commandLine,
      Artifact proguardMap)
      throws InterruptedException;

  /** Given an Android {@code manifest}, returns a list of relevant Proguard specs. */
  ImmutableList<Artifact> getProguardSpecsForManifest(RuleContext ruleContext, Artifact manifest);

  /**
   * Add coverage instrumentation to the Java compilation of an Android binary.
   *
   * @throws InterruptedException
   */
  void addCoverageSupport(
      RuleContext ruleContext,
      AndroidCommon common,
      JavaSemantics javaSemantics,
      boolean forAndroidTest,
      JavaTargetAttributes.Builder attributes,
      JavaCompilationArtifacts.Builder artifactsBuilder)
      throws InterruptedException;

  /** Returns the list of attributes that may contribute Java runtime dependencies. */
  ImmutableList<String> getAttributesWithJavaRuntimeDeps(RuleContext ruleContext);

  /** A hook for checks of internal-only or external-only attributes of {@code android_binary}. */
  default void validateAndroidBinaryRuleContext(RuleContext ruleContext) throws RuleErrorException {
  }

  /** A hook for checks of internal-only or external-only attributes of {@code android_library}. */
  default void validateAndroidLibraryRuleContext(RuleContext ruleContext)
      throws RuleErrorException {
  }

  /** The artifact for the map that proguard will output. */
  Artifact getProguardOutputMap(RuleContext ruleContext) throws InterruptedException;

  /** Maybe post process the dex files and proguard output map. */
  AndroidBinary.DexPostprocessingOutput postprocessClassesDexZip(
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> filesBuilder,
      Artifact classesDexZip,
      ProguardOutput proguardOutput)
      throws InterruptedException;
}
