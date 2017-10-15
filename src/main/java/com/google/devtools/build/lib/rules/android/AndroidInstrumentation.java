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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.util.FileType;

/**
 * An implementation of the {@code android_instrumentation} rule.
 */
public class AndroidInstrumentation implements RuleConfiguredTargetFactory {

  private static final SafeImplicitOutputsFunction TARGET_APK = ImplicitOutputsFunction
      .fromTemplates("%{name}-target.apk");
  private static final SafeImplicitOutputsFunction INSTRUMENTATION_APK =
      ImplicitOutputsFunction.fromTemplates("%{name}-instrumentation.apk");
  static final SafeImplicitOutputsFunction IMPLICIT_OUTPUTS_FUNCTION =
      ImplicitOutputsFunction.fromFunctions(TARGET_APK, INSTRUMENTATION_APK);

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    Artifact targetApk = getTargetApk(ruleContext);
    Artifact instrumentationApk = createInstrumentationApk(ruleContext);
    NestedSet<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().add(targetApk).add(instrumentationApk).build();

    RuleConfiguredTargetBuilder ruleBuilder = new RuleConfiguredTargetBuilder(ruleContext);
    return ruleBuilder
        .setFilesToBuild(filesToBuild)
        .addProvider(
            RunfilesProvider.class,
            RunfilesProvider.simple(
                new Runfiles.Builder(ruleContext.getWorkspaceName())
                    .addTransitiveArtifacts(filesToBuild)
                    .build()))
        .addNativeDeclaredProvider(new AndroidInstrumentationInfo(targetApk, instrumentationApk))
        .build();
  }

  private static boolean exactlyOneOf(boolean expression1, boolean expression2) {
    return (expression1 && !expression2) || (!expression1 && expression2);
  }

  /**
   * Returns the APK from the {@code target} attribute or creates one from the {@code
   * target_library} attribute.
   */
  private static Artifact getTargetApk(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    Artifact apk = ruleContext.getImplicitOutputArtifact(TARGET_APK);
    TransitiveInfoCollection target = ruleContext.getPrerequisite("target", Mode.TARGET);
    TransitiveInfoCollection targetLibrary =
        ruleContext.getPrerequisite("target_library", Mode.TARGET);

    if (!exactlyOneOf(target == null, targetLibrary == null)) {
      ruleContext.throwWithRuleError(
          "android_instrumentation requires that exactly one of the target and target_library "
              + "attributes be specified.");
    }

    if (target != null) {
      // target attribute is specified
      symlinkApkFromApkProviderOrFile(ruleContext, target, apk, "Symlinking target APK");
    } else {
      // target_library attribute is specified
      createApkFromLibrary(ruleContext, targetLibrary, apk);
    }

    return apk;
  }

  /**
   * Returns the APK from the {@code instrumentation} attribute or creates one from the {@code
   * instrumentation_library} attribute.
   */
  private static Artifact createInstrumentationApk(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    Artifact apk = ruleContext.getImplicitOutputArtifact(INSTRUMENTATION_APK);
    TransitiveInfoCollection instrumentation =
        ruleContext.getPrerequisite("instrumentation", Mode.TARGET);
    TransitiveInfoCollection instrumentationLibrary =
        ruleContext.getPrerequisite("instrumentation_library", Mode.TARGET);

    if (!exactlyOneOf(instrumentation == null, instrumentationLibrary == null)) {
      ruleContext.throwWithRuleError(
          "android_instrumentation requires that exactly one of the instrumentation and "
              + "instrumentation_library attributes be specified.");
    }

    if (instrumentation != null) {
      // instrumentation attribute is specified
      symlinkApkFromApkProviderOrFile(
          ruleContext, instrumentation, apk, "Symlinking instrumentation APK");
    } else {
      // instrumentation_library attribute is specified
      createApkFromLibrary(ruleContext, instrumentationLibrary, apk);
    }

    return apk;
  }

  // We symlink instead of simply providing the artifact as is to satisfy the implicit outputs
  // function. This allows user to refer to the APK outputs of the android_instrumentation rule by
  // the same name, whether they were built from libraries or simply symlinked from the output of
  // an android_binary rule.
  private static void symlinkApkFromApkProviderOrFile(
      RuleContext ruleContext,
      TransitiveInfoCollection transitiveInfoCollection,
      Artifact apk,
      String message) {
    Artifact existingApk;
    ApkProvider apkProvider = transitiveInfoCollection.getProvider(ApkProvider.class);
    if (apkProvider != null) {
      existingApk = apkProvider.getApk();
    } else {
      existingApk =
          Iterables.getOnlyElement(
              FileType.filter(
                  transitiveInfoCollection.getProvider(FileProvider.class).getFilesToBuild(),
                  AndroidRuleClasses.APK));
    }

    ruleContext.registerAction(
        new SymlinkAction(ruleContext.getActionOwner(), existingApk, apk, message));
  }

  @SuppressWarnings("unused") // TODO(b/37856762): Implement APK building from libraries.
  private static Artifact createApkFromLibrary(
      RuleContext ruleContext, TransitiveInfoCollection library, Artifact apk)
      throws RuleErrorException {
    // TODO(b/37856762): Cleanup AndroidBinary#createAndroidBinary and use it here.
    ruleContext.throwWithRuleError(
        "android_instrumentation dependencies on android_library rules are not yet supported");
    return null;
  }
}
