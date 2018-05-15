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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/** Wraps merged Android resources. */
public class MergedAndroidResources extends ParsedAndroidResources {
  private final Artifact mergedResources;
  private final Artifact classJar;
  @Nullable private final Artifact dataBindingInfoZip;
  private final ResourceDependencies resourceDependencies;
  /**
   * The processed manifest.
   *
   * <p>TODO(b/30817309): Just use the manifest inherited from {@link ParsedAndroidResources} once
   * the legacy manifest merger is removed.
   */
  private final ProcessedAndroidManifest manifest;

  public static MergedAndroidResources mergeFrom(
      RuleContext ruleContext,
      ParsedAndroidResources parsed,
      ResourceDependencies resourceDeps,
      boolean enableDataBinding,
      AndroidAaptVersion aaptVersion)
      throws InterruptedException {

    AndroidConfiguration androidConfiguration = AndroidCommon.getAndroidConfig(ruleContext);

    boolean useCompiledMerge =
        aaptVersion == AndroidAaptVersion.AAPT2 && androidConfiguration.skipParsingAction();

    Preconditions.checkState(
        !useCompiledMerge || parsed.getCompiledSymbols() != null,
        "Should not use compiled merge if no compiled symbols are available!");

    AndroidResourceMergingActionBuilder builder =
        new AndroidResourceMergingActionBuilder(ruleContext)
            .setJavaPackage(parsed.getJavaPackage())
            .withDependencies(resourceDeps)
            .setThrowOnResourceConflict(androidConfiguration.throwOnResourceConflict())
            .setUseCompiledMerge(useCompiledMerge);

    if (enableDataBinding) {
      builder.setDataBindingInfoZip(DataBinding.getLayoutInfoFile(ruleContext));
    }

    return builder
        .setManifestOut(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST))
        .setMergedResourcesOut(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP))
        .setClassJarOut(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR))
        .build(ruleContext, parsed);
  }

  public static MergedAndroidResources of(
      ParsedAndroidResources parsed,
      Artifact mergedResources,
      Artifact classJar,
      @Nullable Artifact dataBindingInfoZip,
      ResourceDependencies resourceDependencies,
      ProcessedAndroidManifest manifest) {
    return new MergedAndroidResources(
        parsed, mergedResources, classJar, dataBindingInfoZip, resourceDependencies, manifest);
  }

  MergedAndroidResources(MergedAndroidResources other) {
    this(
        other,
        other.mergedResources,
        other.classJar,
        other.dataBindingInfoZip,
        other.resourceDependencies,
        other.manifest);
  }

  private MergedAndroidResources(
      ParsedAndroidResources other,
      Artifact mergedResources,
      Artifact classJar,
      @Nullable Artifact dataBindingInfoZip,
      ResourceDependencies resourceDependencies,
      ProcessedAndroidManifest manifest) {
    super(other, manifest);
    this.mergedResources = mergedResources;
    this.classJar = classJar;
    this.dataBindingInfoZip = dataBindingInfoZip;
    this.resourceDependencies = resourceDependencies;
    this.manifest = manifest;
  }

  @Deprecated
  public Artifact getMergedResources() {
    return mergedResources;
  }

  public Artifact getClassJar() {
    return classJar;
  }

  @Nullable
  public Artifact getDataBindingInfoZip() {
    return dataBindingInfoZip;
  }

  public Artifact getJavaClassJar() {
    return classJar;
  }

  @Override
  public ProcessedAndroidManifest getStampedManifest() {
    return manifest;
  }

  public ProcessedAndroidManifest getProcessedManifest() {
    return manifest;
  }

  public ResourceDependencies getResourceDependencies() {
    return resourceDependencies;
  }

  /**
   * Validates and packages this rule's resources.
   *
   * <p>See {@link ValidatedAndroidResources#validateFrom(RuleContext, MergedAndroidResources,
   * AndroidAaptVersion)}. This method is a convenience method for calling that one.
   */
  public ValidatedAndroidResources validate(RuleContext ruleContext, AndroidAaptVersion aaptVersion)
      throws InterruptedException {
    return ValidatedAndroidResources.validateFrom(ruleContext, this, aaptVersion);
  }

  @Override
  public Optional<? extends MergedAndroidResources> maybeFilter(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter, boolean isDependency)
      throws RuleErrorException {
    return super.maybeFilter(errorConsumer, resourceFilter, isDependency)
        .map(
            parsed ->
                MergedAndroidResources.of(
                    parsed,
                    mergedResources,
                    classJar,
                    dataBindingInfoZip,
                    resourceDependencies,
                    manifest));
  }

  @Override
  public boolean equals(Object object) {
    if (!super.equals(object)) {
      return false;
    }

    MergedAndroidResources other = (MergedAndroidResources) object;
    return mergedResources.equals(other.mergedResources)
        && classJar.equals(other.classJar)
        && Objects.equals(dataBindingInfoZip, other.dataBindingInfoZip)
        && resourceDependencies.equals(other.resourceDependencies);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        super.hashCode(), mergedResources, classJar, dataBindingInfoZip, resourceDependencies);
  }
}
