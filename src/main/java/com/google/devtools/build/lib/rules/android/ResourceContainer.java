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

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/** The resources contributed by a single target. */
@AutoValue
@Immutable
public abstract class ResourceContainer implements ValidatedAndroidData {
  /** The type of resource in question: either asset or a resource. */
  public enum ResourceType {
    ASSETS("assets"),
    RESOURCES("resources");

    private final String attribute;

    private ResourceType(String attribute) {
      this.attribute = attribute;
    }

    public String getAttribute() {
      return attribute;
    }
  }

  public abstract Label getLabel();

  @Nullable
  public abstract String getJavaPackage();

  @Override
  @Nullable
  public abstract Artifact getApk();

  @Override
  public ProcessedAndroidManifest getProcessedManifest() {
    return new ProcessedAndroidManifest(getManifest(), getJavaPackage(), isManifestExported());
  }

  @Override
  public abstract Artifact getManifest();

  @Override
  @Nullable
  public abstract Artifact getJavaSourceJar();

  @Nullable
  public abstract Artifact getJavaClassJar();

  @Override
  public ImmutableList<Artifact> getAssets() {
    return getAndroidAssets().getAssets();
  }

  public abstract AndroidAssets getAndroidAssets();

  @Override
  public ImmutableList<Artifact> getResources() {
    return getAndroidResources().getResources();
  }

  @VisibleForTesting
  public abstract AndroidResources getAndroidResources();

  /** @deprecated We are moving towards decoupling assets and resources */
  @Deprecated
  public ImmutableList<Artifact> getArtifacts(ResourceType resourceType) {
    return resourceType == ResourceType.ASSETS ? getAssets() : getResources();
  }

  @Override
  public Iterable<Artifact> getArtifacts() {
    return Iterables.concat(getAssets(), getResources());
  }

  @Override
  public ImmutableList<PathFragment> getResourceRoots() {
    return getAndroidResources().getResourceRoots();
  }

  @Override
  public ImmutableList<PathFragment> getAssetRoots() {
    return getAndroidAssets().getAssetRoots();
  }

  /**
   * Gets the directories containing the resources of a specific type.
   *
   * <p>TODO(b/30308041): Stop using these directories, and remove this code.
   *
   * @deprecated We are moving towards passing around the actual artifacts, rather than the
   *     directories that contain them. If the resources were provided with a glob() that excludes
   *     some files, the resource directory will still contain those files, resulting in unwanted
   *     inputs.
   */
  @Deprecated
  public ImmutableList<PathFragment> getRoots(ResourceType resourceType) {
    return resourceType == ResourceType.ASSETS ? getAssetRoots() : getResourceRoots();
  }

  @Override
  public abstract boolean isManifestExported();

  @Nullable
  public abstract Artifact getRTxt();

  @Nullable
  public abstract Artifact getSymbols();

  @Nullable
  public abstract Artifact getCompiledSymbols();

  @Nullable
  public abstract Artifact getStaticLibrary();

  @Nullable
  public abstract Artifact getAapt2RTxt();

  @Nullable
  public abstract Artifact getAapt2JavaSourceJar();

  @Nullable
  @Override
  @Deprecated
  public abstract Artifact getMergedResources();

  // The limited hashCode and equals behavior is necessary to avoid duplication when building with
  // fat_apk_cpu set. Artifacts generated in different configurations will naturally be different
  // and non-equal objects, causing the ResourceContainer not to be automatically deduplicated at
  // the android_binary level.
  // TODO(bazel-team): deduplicate explicitly and remove hashCode and equals overrides to avoid
  // breaking "equals means interchangeable"
  @Override
  public int hashCode() {
    return Objects.hash(getLabel(), getRTxt(), getSymbols());
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ResourceContainer)) {
      return false;
    }
    ResourceContainer other = (ResourceContainer) obj;
    return Objects.equals(getLabel(), other.getLabel())
        && Objects.equals(getRTxt(), other.getRTxt())
        && Objects.equals(getSymbols(), other.getSymbols());
  }

  /** Converts this container back into a builder to create a modified copy. */
  public abstract Builder toBuilder();

  /**
   * Returns a copy of this container with filtered resources, or the original if no resources
   * should be filtered. The original container is unchanged.
   */
  public ResourceContainer filter(
      RuleErrorConsumer errorConsumer, ResourceFilter filter, boolean isDependency)
      throws RuleErrorException {
    Optional<? extends AndroidResources> filteredResources =
        getAndroidResources().maybeFilter(errorConsumer, filter, isDependency);

    if (!filteredResources.isPresent()) {
      // No filtering was done; return this container
      return this;
    }
    return toBuilder().setAndroidResources(filteredResources.get()).build();
  }

  /** Creates a new builder with default values. */
  public static Builder builder() {
    return new AutoValue_ResourceContainer.Builder()
        .setJavaPackageFrom(Builder.JavaPackageSource.MANIFEST)
        .setAndroidAssets(AndroidAssets.empty())
        .setAndroidResources(AndroidResources.empty());
  }

  /**
   * Creates a new builder with the label, Java package, manifest package override, Java source jar,
   * and manifest-export switch set according to the given rule.
   */
  public static Builder builderFromRule(RuleContext ruleContext) throws InterruptedException {
    return builder().forRuleContext(ruleContext);
  }

  /** Builder to construct resource containers. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Enum to determine what to do if a package hasn't been manually set. */
    public enum JavaPackageSource {
      /**
       * Uses the package from the manifest, i.e., the generated ResourceContainer will return null
       * from {@link ResourceContainer#getJavaPackage()}.
       */
      MANIFEST,
      /**
       * Uses the package from the path to the source jar (or, if the rule context has it set, the
       * {@code custom_package} attribute). If the source jar is not under a valid Java root, this
       * will result in an error being added to the rule context. This can only be used if the
       * builder was created by {@link ResourceContainer#builderFromRule(RuleContext)}.
       */
      SOURCE_JAR_PATH
    }

    @Nullable private RuleContext ruleContext;
    @Nullable private JavaPackageSource javaPackageSource;

    private Builder forRuleContext(RuleContext ruleContext) throws InterruptedException {
      Preconditions.checkNotNull(ruleContext);
      this.ruleContext = ruleContext;
      return this.setLabel(ruleContext.getLabel())
          .setJavaSourceJar(
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR))
          .setJavaPackageFrom(JavaPackageSource.SOURCE_JAR_PATH)
          .setManifestExported(AndroidCommon.getExportsManifest(ruleContext));
    }

    /**
     * Sets the Java package from the given source. Overrides earlier calls to {@link
     * #setJavaPackageFrom(JavaPackageSource)} or {@link #setJavaPackageFromString(String)}.
     *
     * <p>To set the package from {@link JavaPackageSource#SOURCE_JAR_PATH}, this instance must have
     * been created using {@link ResourceContainer#builderFromRule(RuleContext)}. Also in this case,
     * the source jar must be set non-{@code null} when the {@link #build()} method is called. It
     * defaults to the source jar implicit output when creating a builder out of a rule context.
     */
    public Builder setJavaPackageFrom(JavaPackageSource javaPackageSource) {
      Preconditions.checkNotNull(javaPackageSource);
      Preconditions.checkArgument(
          !(javaPackageSource == JavaPackageSource.SOURCE_JAR_PATH && ruleContext == null),
          "setJavaPackageFrom(SOURCE_JAR_PATH) is only permitted when using builderFromRule.");
      this.javaPackageSource = javaPackageSource;
      return this.setJavaPackage(null);
    }

    /**
     * Sets the Java package from the given string. Overrides earlier calls to {@link
     * #setJavaPackageFrom(JavaPackageSource)} or {@link #setJavaPackageFromString(String)}.
     *
     * <p>To make {@link ResourceContainer#getJavaPackage()} return {@code null}, call {@code
     * setJavaPackageFrom(MANIFEST)} instead.
     */
    public Builder setJavaPackageFromString(String javaPackageOverride) {
      Preconditions.checkNotNull(javaPackageOverride);
      this.javaPackageSource = null;
      return this.setJavaPackage(javaPackageOverride);
    }

    public abstract Builder setLabel(Label label);

    abstract Builder setJavaPackage(@Nullable String javaPackage);

    public abstract Builder setApk(@Nullable Artifact apk);

    public abstract Builder setManifest(Artifact manifest);

    @Nullable
    abstract Artifact getJavaSourceJar();

    public abstract Builder setJavaSourceJar(@Nullable Artifact javaSourceJar);

    public abstract Builder setJavaClassJar(@Nullable Artifact javaClassJar);

    public abstract Builder setAndroidAssets(AndroidAssets assets);

    public abstract Builder setAndroidResources(AndroidResources resources);

    public abstract Builder setManifestExported(boolean manifestExported);

    public abstract Builder setRTxt(@Nullable Artifact rTxt);

    public abstract Builder setSymbols(@Nullable Artifact symbols);

    public abstract Builder setCompiledSymbols(@Nullable Artifact compiledSymbols);

    public abstract Builder setStaticLibrary(@Nullable Artifact staticLibrary);

    public abstract Builder setAapt2JavaSourceJar(@Nullable Artifact javaSourceJar);

    public abstract Builder setAapt2RTxt(@Nullable Artifact rTxt);

    public abstract Builder setMergedResources(@Nullable Artifact mergedResources);

    abstract ResourceContainer autoBuild();

    /**
     * Builds and returns the ResourceContainer.
     *
     * <p>May result in the rule context adding a rule error if the Java package was to be set from
     * the source jar path, but the source jar does not have an acceptable Java package.
     */
    public ResourceContainer build() {
      if (javaPackageSource == JavaPackageSource.SOURCE_JAR_PATH) {
        Preconditions.checkState(
            !(javaPackageSource == JavaPackageSource.SOURCE_JAR_PATH && ruleContext == null),
            "setJavaPackageFrom(SOURCE_JAR_PATH) is only permitted when using builderFromRule.");
        Preconditions.checkState(
            getJavaSourceJar() != null,
            "setJavaPackageFrom(SOURCE_JAR_PATH) was called, but no source jar was set.");
        setJavaPackage(getJavaPackageFromSourceJarPath());
      }
      return autoBuild();
    }

    @Nullable
    private String getJavaPackageFromSourceJarPath() {
      if (javaPackageSource != JavaPackageSource.SOURCE_JAR_PATH) {
        return null;
      }
      if (hasCustomPackage(ruleContext)) {
        return ruleContext.attributes().get("custom_package", Type.STRING);
      }

      return AndroidManifest.getJavaPackageFromPath(ruleContext, getJavaSourceJar().getExecPath());
    }

    private static boolean hasCustomPackage(RuleContext ruleContext) {
      return ruleContext.attributes().isAttributeValueExplicitlySpecified("custom_package");
    }
  }
}
