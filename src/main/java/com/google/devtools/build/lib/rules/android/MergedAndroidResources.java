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
  @Nullable private final Artifact aapt2RTxt;
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
      AndroidDataContext dataContext,
      ParsedAndroidResources parsed,
      ResourceDependencies resourceDeps,
      AndroidAaptVersion aaptVersion)
      throws InterruptedException {

    Preconditions.checkArgument(aaptVersion == AndroidAaptVersion.AAPT2);
    Preconditions.checkNotNull(
        parsed.getCompiledSymbols(),
        "Should not use compiled merge if no compiled symbols are available!");

    AndroidResourceMergingActionBuilder builder =
        new AndroidResourceMergingActionBuilder()
            .setJavaPackage(parsed.getJavaPackage())
            .withDependencies(resourceDeps)
            .setThrowOnResourceConflict(dataContext.throwOnResourceConflict())
            .setAnnotateRFieldsFromTransitiveDeps(dataContext.annotateRFieldsFromTransitiveDeps())
            .setOmitTransitiveDependenciesFromAndroidRClasses(
                dataContext.omitTransitiveResourcesFromAndroidRClasses());

    parsed.asDataBindingContext().supplyLayoutInfo(builder::setDataBindingInfoZip);

    if (dataContext.getAndroidConfig().useRTxtFromMergedResources()) {
      builder.setAapt2RTxtOut(
          dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_R_TXT));
    }

    return builder
        .setManifestOut(
            dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST))
        .setClassJarOut(
            dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR))
        .build(dataContext, parsed);
  }

  public static MergedAndroidResources of(
      ParsedAndroidResources parsed,
      Artifact mergedResources,
      Artifact classJar,
      Artifact aapt2RTxt,
      @Nullable Artifact dataBindingInfoZip,
      ResourceDependencies resourceDependencies,
      ProcessedAndroidManifest manifest) {
    return new MergedAndroidResources(
        parsed,
        mergedResources,
        classJar,
        aapt2RTxt,
        dataBindingInfoZip,
        resourceDependencies,
        manifest);
  }

  MergedAndroidResources(MergedAndroidResources other) {
    this(
        other,
        other.mergedResources,
        other.classJar,
        other.aapt2RTxt,
        other.dataBindingInfoZip,
        other.resourceDependencies,
        other.manifest);
  }

  protected MergedAndroidResources(
      ParsedAndroidResources other,
      Artifact mergedResources,
      Artifact classJar,
      Artifact aapt2RTxt,
      @Nullable Artifact dataBindingInfoZip,
      ResourceDependencies resourceDependencies,
      ProcessedAndroidManifest manifest) {
    super(other, manifest);
    this.mergedResources = mergedResources;
    this.classJar = classJar;
    this.aapt2RTxt = aapt2RTxt;
    this.dataBindingInfoZip = dataBindingInfoZip;
    this.resourceDependencies = resourceDependencies;
    this.manifest = manifest;
  }

  /**
   * Gets an Artifact containing a zip of merged resources.
   *
   * <p>If assets were processed together with resources, the zip will also contain merged assets.
   *
   * @deprecated This artifact is produced by an often-expensive action and should not be used if
   *     another option is available. Furthermore, it will be replaced by flat files once we
   *     completely move to aapt2.
   */
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

  @Nullable
  Artifact getAapt2RTxt() {
    return aapt2RTxt;
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
   * <p>See {@link ValidatedAndroidResources#validateFrom(AndroidDataContext,
   * MergedAndroidResources, AndroidAaptVersion)}. This method is a convenience method for calling
   * that one.
   */
  public ValidatedAndroidResources validate(
      AndroidDataContext dataContext, AndroidAaptVersion aaptVersion) throws InterruptedException {
    return ValidatedAndroidResources.validateFrom(dataContext, this, aaptVersion);
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
                    aapt2RTxt,
                    dataBindingInfoZip,
                    resourceDependencies,
                    manifest));
  }

  @Override
  public boolean equals(Object object) {
    if (!super.equals(object) || !(object instanceof MergedAndroidResources)) {
      return false;
    }

    MergedAndroidResources other = (MergedAndroidResources) object;
    return Objects.equals(mergedResources, other.mergedResources)
        && classJar.equals(other.classJar)
        && Objects.equals(aapt2RTxt, other.aapt2RTxt)
        && Objects.equals(dataBindingInfoZip, other.dataBindingInfoZip)
        && resourceDependencies.equals(other.resourceDependencies);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        super.hashCode(),
        mergedResources,
        classJar,
        aapt2RTxt,
        dataBindingInfoZip,
        resourceDependencies);
  }
}
