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
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;

import javax.annotation.Nullable;

/**
 * Pluggable semantics for Android rules.
 *
 * <p>A new instance of this class is created for each configured target, therefore, it is allowed
 * to keep state.
 */
public interface AndroidSemantics {
  /**
   * Name of the output group used for idl jars (the jars containing the class files for sources
   * generated from annotation processors).
   */
  String IDL_JARS_OUTPUT_GROUP =
      OutputGroupProvider.HIDDEN_OUTPUT_GROUP_PREFIX + "idl_jars";

  /**
   * Adds transitive info providers for {@code android_binary} and {@code android_library} rules.
   * @throws InterruptedException
   */
  void addTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder,
      RuleContext ruleContext,
      JavaCommon javaCommon,
      AndroidCommon androidCommon,
      Artifact jarWithAllClasses)
      throws InterruptedException;

  /**
   * Add additional resources to IDE info for {@code android_binary} and {@code android_library}
   *
   * @param ruleContext rule context for target rule
   * @param resourceApk resource apk directly provided by the rule
   * @param ideInfoProviderBuilder
   */
  void addNonLocalResources(
      RuleContext ruleContext,
      @Nullable ResourceApk resourceApk,
      AndroidIdeInfoProvider.Builder ideInfoProviderBuilder);

  /**
   * Returns the manifest to be used when compiling a given rule.
   * @throws InterruptedException 
   */
  ApplicationManifest getManifestForRule(RuleContext ruleContext) throws InterruptedException;

  /**
   * Returns the name of the file in which the file names of native dependencies are listed.
   */
  String getNativeDepsFileName();

  /**
   * Returns the command line options to be used when compiling Java code for {@code android_*}
   * rules.
   *
   * <p>These will come after the default options specified by the toolchain and the ones in the
   * {@code javacopts} attribute.
   */
  ImmutableList<String> getJavacArguments();

  /**
   * JVM arguments to be passed to the command line of dx.
   */
  ImmutableList<String> getDxJvmArguments();

  /**
   * Returns the artifact for the debug key for signing the APK.
   */
  Artifact getApkDebugSigningKey(RuleContext ruleContext);
  
  /**
   * Add coverage instrumentation to the Java compilation of an Android binary.
   * @throws InterruptedException 
   */
  void addCoverageSupport(RuleContext ruleContext, AndroidCommon common,
      JavaSemantics javaSemantics, boolean forAndroidTest, JavaTargetAttributes.Builder attributes,
      JavaCompilationArtifacts.Builder artifactsBuilder) throws InterruptedException;
}
