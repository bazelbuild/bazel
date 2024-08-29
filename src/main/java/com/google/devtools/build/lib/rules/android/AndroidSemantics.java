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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import java.util.Map;
import java.util.Optional;

/**
 * Pluggable semantics for Android rules.
 *
 * <p>A new instance of this class is created for each configured target, therefore, it is allowed
 * to keep state.
 */
public interface AndroidSemantics {
  /** Implementation for the :proguard attribute. */
  @SerializationConstant
  LabelLateBoundDefault<JavaConfiguration> PROGUARD =
      LabelLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class,
          null,
          (rule, attributes, javaConfig) -> javaConfig.getProguardBinary());

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
  ImmutableList<String> getCompatibleJavacOptions(RuleContext ruleContext)
      throws RuleErrorException;

  default AndroidDataContext makeContextForNative(RuleContext ruleContext)
      throws RuleErrorException {
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
            .get("tags", Types.STRING_LIST)
            .contains("__ANDROID_RULES_MIGRATION__");
    if (!hasMigrationTag) {
      registerMigrationRuleError(ruleContext);
    }
  }

  /** Executes a ruleContext.attributeError when the check for the migration tag fails. */
  void registerMigrationRuleError(RuleContext ruleContext) throws RuleErrorException;
}
