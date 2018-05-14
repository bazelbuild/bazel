// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/** Wraps validated and packaged Android resource information */
public class ValidatedAndroidResources extends MergedAndroidResources
    implements ValidatedAndroidData {
  private final Artifact rTxt;
  private final Artifact sourceJar;
  private final Artifact apk;

  // aapt2 outputs. Will be null if and only if aapt2 is not used for validation.
  @Nullable private final Artifact aapt2RTxt;
  @Nullable private final Artifact aapt2SourceJar;
  @Nullable private final Artifact staticLibrary;

  /**
   * Validates and packages merged resources.
   *
   * <p>Specifically, validates that:
   *
   * <ul>
   *   <li>there are no conflicts between resources (though currently we just warn)
   *   <li>each reference to a resource in resources and manifest are satisfied
   * </ul>
   *
   * <p>And packs resources into:
   *
   * <ul>
   *   <li>R.java and R.txt files
   *   <li>A resource-only APK (deprecated)
   *   <li>When building with aapt2, aapt2 equivalents of the above
   *   <li>When building with aapt2, a compiled symbols zip
   * </ul>
   */
  public static ValidatedAndroidResources validateFrom(
      RuleContext ruleContext, MergedAndroidResources merged, AndroidAaptVersion aaptVersion)
      throws InterruptedException {
    AndroidResourceValidatorActionBuilder builder =
        new AndroidResourceValidatorActionBuilder(ruleContext)
            .setJavaPackage(merged.getJavaPackage())
            .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
            .setMergedResources(merged.getMergedResources())
            .setRTxtOut(ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT))
            .setSourceJarOut(
                ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR))
            // Request an APK so it can be inherited when a library is used in a binary's
            // resources attr.
            // TODO(b/30307842): Remove this once it is no longer needed for resources migration.
            .setApkOut(
                ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_APK))
            .withDependencies(merged.getResourceDependencies());

    if (aaptVersion == AndroidAaptVersion.AAPT2) {
      builder
          .setCompiledSymbols(merged.getCompiledSymbols())
          .setAapt2RTxtOut(
              ruleContext.getImplicitOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_R_TXT))
          .setAapt2SourceJarOut(
              ruleContext.getImplicitOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_SOURCE_JAR))
          .setStaticLibraryOut(
              ruleContext.getImplicitOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_LIBRARY_APK));
    }

    return builder.build(ruleContext, merged);
  }

  static ValidatedAndroidResources of(
      MergedAndroidResources merged,
      Artifact rTxt,
      Artifact sourceJar,
      Artifact apk,
      Artifact aapt2RTxt,
      Artifact aapt2SourceJar,
      Artifact staticLibrary) {
    return new ValidatedAndroidResources(
        merged, rTxt, sourceJar, apk, aapt2RTxt, aapt2SourceJar, staticLibrary);
  }

  private ValidatedAndroidResources(
      MergedAndroidResources merged,
      Artifact rTxt,
      Artifact sourceJar,
      Artifact apk,
      @Nullable Artifact aapt2RTxt,
      @Nullable Artifact aapt2SourceJar,
      @Nullable Artifact staticLibrary) {
    super(merged);
    this.rTxt = rTxt;
    this.sourceJar = sourceJar;
    this.apk = apk;
    this.aapt2RTxt = aapt2RTxt;
    this.aapt2SourceJar = aapt2SourceJar;
    this.staticLibrary = staticLibrary;
  }

  public AndroidResourcesInfo toProvider() {
    return getResourceDependencies().toInfo(this);
  }

  @Override
  public Artifact getRTxt() {
    return rTxt;
  }

  @Override
  public Artifact getJavaSourceJar() {
    return sourceJar;
  }

  @Override
  public Artifact getApk() {
    return apk;
  }

  @Override
  @Nullable
  public Artifact getAapt2RTxt() {
    return aapt2RTxt;
  }

  @Nullable
  public Artifact getAapt2SourceJar() {
    return aapt2SourceJar;
  }

  @Override
  @Nullable
  public Artifact getStaticLibrary() {
    return staticLibrary;
  }

  @Override
  public ValidatedAndroidResources filter(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter, boolean isDependency)
      throws RuleErrorException {
    return maybeFilter(errorConsumer, resourceFilter, isDependency).orElse(this);
  }

  @Override
  public Optional<ValidatedAndroidResources> maybeFilter(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter, boolean isDependency)
      throws RuleErrorException {
    return super.maybeFilter(errorConsumer, resourceFilter, isDependency)
        .map(
            merged ->
                ValidatedAndroidResources.of(
                    merged, rTxt, sourceJar, apk, aapt2RTxt, aapt2SourceJar, staticLibrary));
  }

  @Override
  public boolean equals(Object object) {
    if (!super.equals(object)) {
      return false;
    }

    ValidatedAndroidResources other = (ValidatedAndroidResources) object;
    return rTxt.equals(other.rTxt)
        && sourceJar.equals(other.sourceJar)
        && apk.equals(other.apk)
        && Objects.equals(aapt2RTxt, other.aapt2RTxt)
        && Objects.equals(aapt2SourceJar, other.aapt2SourceJar)
        && Objects.equals(staticLibrary, other.staticLibrary);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        super.hashCode(), rTxt, sourceJar, apk, aapt2RTxt, aapt2SourceJar, staticLibrary);
  }
}
