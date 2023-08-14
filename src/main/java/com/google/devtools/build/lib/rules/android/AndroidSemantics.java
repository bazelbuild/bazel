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
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.android.ProguardHelper.ProguardOutput;
import com.google.devtools.build.lib.rules.java.BootClassPathInfo;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import java.util.Map;
import java.util.Optional;

/**
 * Pluggable semantics for Android rules.
 *
 * <p>A new instance of this class is created for each configured target, therefore, it is allowed
 * to keep state.
 */
public interface AndroidSemantics {

  default AndroidManifest renameManifest(
      AndroidDataContext dataContext, AndroidManifest rawManifest) throws InterruptedException {
    return rawManifest.renameManifestIfNeeded(dataContext);
  }

  default Optional<Artifact> maybeDoLegacyManifestMerging(
      Map<Artifact, Label> mergeeManifests,
      AndroidDataContext dataContext,
      Artifact primaryManifest) {
    if (mergeeManifests.isEmpty()) {
      return Optional.empty();
    }

    throw new UnsupportedOperationException();
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
  ImmutableList<Artifact> getProguardSpecsForManifest(
      AndroidDataContext dataContext, Artifact manifest);

  /**
   * Add coverage instrumentation to the Java compilation of an Android binary.
   *
   * @throws InterruptedException
   */
  void addCoverageSupport(
      RuleContext ruleContext, boolean forAndroidTest, JavaTargetAttributes.Builder attributes)
      throws InterruptedException;

  /** Returns the list of attributes that may contribute Java runtime dependencies. */
  ImmutableList<String> getAttributesWithJavaRuntimeDeps(RuleContext ruleContext);

  /** A hook for checks of internal-only or external-only attributes of {@code android_binary}. */
  default void validateAndroidBinaryRuleContext(RuleContext ruleContext)
      throws RuleErrorException {}

  /** A hook for checks of internal-only or external-only attributes of {@code android_library}. */
  default void validateAndroidLibraryRuleContext(RuleContext ruleContext)
      throws RuleErrorException {}

  /** The artifact for the map that proguard will output. */
  Artifact getProguardOutputMap(RuleContext ruleContext) throws InterruptedException;

  /** The artifact for ART profile information. */
  Artifact getArtProfileForApk(
      RuleContext ruleContext,
      Artifact finalClassesDex,
      Artifact proguardOutputMap,
      String baselineProfileDir);

  /** The merged baseline profiles from the {@code baseline_profiles} attribute. */
  Artifact mergeBaselineProfiles(
      RuleContext ruleContext, String baselineProfileDir, boolean includeStartupProfiles);

  /** The merged startup profiles from the {@code startup_profiles} attribute. */
  Artifact mergeStartupProfiles(RuleContext ruleContext, String baselineProfileDir);

  /** Expands any wildcards present in a baseline profile, and returns the new expanded artifact. */
  public Artifact expandBaselineProfileWildcards(
      RuleContext ruleContext,
      Artifact deployJar,
      Artifact mergedStaticProfile,
      String baselineProfileDir);

  /** The artifact for ART profile information, given a particular merged profile. */
  Artifact compileBaselineProfile(
      RuleContext ruleContext,
      Artifact finalClassesDex,
      Artifact proguardOutputMap,
      Artifact mergedStaticProfile,
      String baselineProfileDir);

  boolean postprocessClassesRewritesMap(RuleContext ruleContext);

  /** Maybe post process the dex files and proguard output map. */
  AndroidBinary.DexPostprocessingOutput postprocessClassesDexZip(
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> filesBuilder,
      Artifact classesDexZip,
      ProguardOutput proguardOutput,
      Artifact proguardMapOutput,
      Artifact mainDexList)
      throws InterruptedException;

  default AndroidDataContext makeContextForNative(RuleContext ruleContext) {
    return AndroidDataContext.forNative(ruleContext);
  }

  /**
   * Checks if the migration tag has been added to the rules list of tags. If the tag is missing,
   * the user is accessing the rule directly in a BUILD file or through a macro that is accessing it
   * directly.
   */
  default void checkForMigrationTag(RuleContext ruleContext) throws RuleErrorException {
    if (!AndroidCommon.getAndroidConfig(ruleContext).checkForMigrationTag()) {
      return;
    }
    boolean hasMigrationTag =
        ruleContext
            .attributes()
            .get("tags", Type.STRING_LIST)
            .contains("__ANDROID_RULES_MIGRATION__");
    if (!hasMigrationTag) {
      registerMigrationRuleError(ruleContext);
    }
  }

  /** Executes a ruleContext.attributeError when the check for the migration tag fails. */
  void registerMigrationRuleError(RuleContext ruleContext) throws RuleErrorException;

  /**
   * Whether invoking apksigner, whether or not to pass it flags to make DSA signing be
   * deterministic.
   */
  default boolean deterministicSigning() {
    return false;
  }

  default BootClassPathInfo getBootClassPathInfo(RuleContext ruleContext)
      throws RuleErrorException {
    BootClassPathInfo bootClassPathInfo;
    AndroidSdkProvider androidSdkProvider = AndroidSdkProvider.fromRuleContext(ruleContext);
    if (androidSdkProvider.getSystem() != null) {
      bootClassPathInfo = androidSdkProvider.getSystem();
    } else {
      NestedSetBuilder<Artifact> bootclasspath = NestedSetBuilder.<Artifact>stableOrder();
      if (ruleContext.getConfiguration().getFragment(AndroidConfiguration.class).desugarJava8()) {
        bootclasspath.addTransitive(
            PrerequisiteArtifacts.nestedSet(ruleContext, "$desugar_java8_extra_bootclasspath"));
      }
      bootclasspath.add(androidSdkProvider.getAndroidJar());
      bootClassPathInfo = BootClassPathInfo.create(bootclasspath.build());
    }
    return bootClassPathInfo;
  }
}
