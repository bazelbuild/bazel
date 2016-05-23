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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

import java.util.regex.Pattern;

/**
 * Implementation for the {@code android_tools_defaults_jar} rule.
 *
 * <p>This rule is a sad, sad way to let people depend on {@code android.jar} when an
 * {@code android_sdk} rule is used. In an ideal world, people would say "depend on
 * the android_jar output group of $config.android_sdk", but, alas, neither depending on labels in
 * the configuration nor depending on a specified output group works.
 *
 * <p>So all this needs to be implemented manually. This rule is injected into the defaults package
 * from {@link AndroidConfiguration.Options#getDefaultsRules()}.
 */
public class AndroidToolsDefaultsJar implements RuleConfiguredTargetFactory {
  private static final Pattern ANDROID_JAR_BASENAME_RX =
      Pattern.compile("android[a-zA-Z0-9_]*\\.jar$");

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    if (!ruleContext.getLabel().getPackageName().equals("tools/defaults")) {
      // Guard against extraordinarily inquisitive individuals.
      ruleContext.ruleError("The android_tools_defaults_jar rule should not be used in BUILD files."
          + " It is a rule internal to the build tool.");
      return null;
    }

    TransitiveInfoCollection androidSdk = ruleContext.getPrerequisite(":android_sdk", Mode.TARGET);
    AndroidSdkProvider sdkProvider =  androidSdk.getProvider(AndroidSdkProvider.class);
    Artifact androidJar = sdkProvider != null
        ? sdkProvider.getAndroidJar()
        : findAndroidJar(androidSdk.getProvider(FileProvider.class).getFilesToBuild());

    NestedSet<Artifact> filesToBuild = androidJar == null
        ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
        : NestedSetBuilder.create(Order.STABLE_ORDER, androidJar);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .setFilesToBuild(filesToBuild)
        .build();
  }

  private static Artifact findAndroidJar(Iterable<Artifact> fullSdk) {
    // We need to do this by sifting through all the files in the Android SDK because we need to
    // handle the case when --android_sdk points to a plain filegroup.
    //
    // We can't avoid adding an android_tools_defaults_jar rule to the defaults package when this is
    // the case, because the defaults package is constructed very early and it's not possible to get
    // information about the rule class of the target pointed to by --android_sdk earlier, and it's
    // doubly impossible to do redirect chasing then.
    for (Artifact artifact : fullSdk) {

      if (ANDROID_JAR_BASENAME_RX.matcher(artifact.getExecPath().getBaseName()).matches()) {
        return artifact;
      }
    }

    return null;
  }
}
