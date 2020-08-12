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

import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidManifestMerger;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/** Wraps an Android Manifest and provides utilities for working with it */
@Immutable
public class AndroidManifest {
  private static final String CUSTOM_PACKAGE_ATTR = "custom_package";

  private final Artifact manifest;
  /** The Android package. Will be null if and only if this is an aar_import target. */
  @Nullable private final String pkg;
  private final boolean exported;

  public static StampedAndroidManifest forAarImport(Artifact manifest) {
    return new StampedAndroidManifest(manifest, /* pkg = */ null, /* exported = */ true);
  }

  /**
   * Gets the manifest for this rule.
   *
   * <p>If no manifest is specified in the rule's attributes, an empty manifest will be generated.
   *
   * <p>Unlike {@link #fromAttributes(RuleContext, AndroidDataContext,AndroidSemantics)}, the
   * AndroidSemantics-specific manifest processing methods will not be applied in this method. The
   * manifest returned by this method will be the same regardless of the AndroidSemantics being
   * used.
   */
  public static AndroidManifest fromAttributes(
      RuleContext ruleContext, AndroidDataContext dataContext)
      throws InterruptedException, RuleErrorException {
    return fromAttributes(ruleContext, dataContext, null);
  }

  /**
   * Gets the manifest for this rule.
   *
   * <p>If no manifest is specified in the rule's attributes, an empty manifest will be generated.
   *
   * <p>If a non-null {@link AndroidSemantics} is passed, AndroidSemantics-specific manifest
   * processing will be preformed on this manifest. Otherwise, basic manifest renaming will be
   * performed if needed.
   */
  public static AndroidManifest fromAttributes(
      RuleContext ruleContext,
      AndroidDataContext dataContext,
      @Nullable AndroidSemantics androidSemantics)
      throws RuleErrorException, InterruptedException {
    Artifact rawManifest = null;
    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      AndroidResources.validateRuleContext(ruleContext);
      rawManifest = ruleContext.getPrerequisiteArtifact("manifest", TransitionMode.TARGET);
    }

    return from(
        dataContext,
        ruleContext,
        rawManifest,
        androidSemantics,
        getAndroidPackage(ruleContext),
        AndroidCommon.getExportsManifest(ruleContext));
  }

  /**
   * Creates an AndroidManifest object, with correct preprocessing, from explicit variables.
   *
   * <p>Attributes included in the RuleContext will not be used; use {@link
   * #fromAttributes(RuleContext, AndroidDataContext)} instead.
   *
   * <p>In addition, the AndroidSemantics-specific manifest processing methods will not be applied
   * in this method. The manifest returned by this method will be the same regardless of the
   * AndroidSemantics being used. use {@link #fromAttributes(RuleContext, AndroidDataContext,
   * AndroidSemantics)} instead if you want AndroidSemantics-specific behavior.
   */
  public static AndroidManifest from(
      AndroidDataContext dataContext,
      RuleErrorConsumer errorConsumer,
      @Nullable Artifact rawManifest,
      @Nullable String pkg,
      boolean exportsManifest)
      throws InterruptedException {
    return from(dataContext, errorConsumer, rawManifest, null, pkg, exportsManifest);
  }

  /**
   * Inner method to create an AndroidManifest.
   *
   * @param rawManifest If non-null, the returned object will wrap this manifest. Otherwise, the
   *     returned object will wrap a generated dummy manifest.
   * @param androidSemantics If non-null, will invoke
   *     AndroidSemantics#renameManifest(AndroidDataContext, AndroidManifest)} to do
   *     platform-specific processing on the manifest.
   * @param pkg If non-null, this Android package will be used for the manifest, and the manifest
   *     will be stamped with it when the {@link #stamp(AndroidDataContext)} method is called.
   *     Otherwise, the default package, based on the current target's Bazel package, will be used.
   */
  public static AndroidManifest from(
      AndroidDataContext dataContext,
      RuleErrorConsumer errorConsumer,
      @Nullable Artifact rawManifest,
      @Nullable AndroidSemantics androidSemantics,
      @Nullable String pkg,
      boolean exportsManifest)
      throws InterruptedException {
    if (pkg == null) {
      pkg =
          getDefaultPackage(
              dataContext.getLabel(), dataContext.getActionConstructionContext(), errorConsumer);
    }

    if (rawManifest == null) {
      // Generate a dummy manifest
      return StampedAndroidManifest.createEmpty(
          dataContext.getActionConstructionContext(), pkg, exportsManifest);
    }

    AndroidManifest raw = new AndroidManifest(rawManifest, pkg, exportsManifest);

    if (androidSemantics != null) {
      return androidSemantics.renameManifest(dataContext, raw);
    }
    return raw.renameManifestIfNeeded(dataContext);
  }

  AndroidManifest renameManifestIfNeeded(AndroidDataContext dataContext)
      throws InterruptedException {
    if (manifest.getFilename().equals("AndroidManifest.xml")) {
      return this;
    } else {
      /*
       * If the manifest file is not named AndroidManifest.xml, we create a symlink named
       * AndroidManifest.xml to it. aapt requires the manifest to be named as such.
       */
      Artifact manifestSymlink =
          dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_SYMLINKED_MANIFEST);
      dataContext.registerAction(SymlinkAction.toArtifact(
          dataContext.getActionConstructionContext().getActionOwner(),
          manifest,
          manifestSymlink,
          "Renaming Android manifest for " + dataContext.getLabel()));
      return updateManifest(manifestSymlink);
    }
  }

  public AndroidManifest updateManifest(Artifact manifest) {
    return new AndroidManifest(manifest, pkg, exported);
  }

  /**
   * Creates a manifest wrapper without doing any processing. From within a rule, use {@link
   * #fromAttributes(RuleContext, AndroidDataContext, AndroidSemantics)} instead.
   */
  public AndroidManifest(Artifact manifest, @Nullable String pkg, boolean exported) {
    this.manifest = manifest;
    this.pkg = pkg;
    this.exported = exported;
  }

  /** If needed, stamps the manifest with the correct Java package */
  public StampedAndroidManifest stamp(AndroidDataContext dataContext) {
    Artifact outputManifest = getManifest();
    if (!isNullOrEmpty(pkg)) {
      outputManifest = dataContext.getUniqueDirectoryArtifact("_renamed", "AndroidManifest.xml");
      new ManifestMergerActionBuilder()
          .setManifest(manifest)
          .setLibrary(true)
          .setCustomPackage(pkg)
          .setManifestOutput(outputManifest)
          .build(dataContext);
    }

    return new StampedAndroidManifest(outputManifest, pkg, exported);
  }

  /**
   * Merges the manifest with any dependent manifests
   *
   * <p>The manifest will also be stamped with any manifest values specified
   *
   * <p>If there is no merging to be done and no manifest values are specified, the manifest will
   * remain unstamped.
   *
   * @param manifestMerger if not null, a string dictating which manifest merger to use
   */
  public StampedAndroidManifest mergeWithDeps(
      AndroidDataContext dataContext,
      AndroidSemantics androidSemantics,
      RuleErrorConsumer errorConsumer,
      ResourceDependencies resourceDeps,
      Map<String, String> manifestValues,
      @Nullable String manifestMerger) {
    Map<Artifact, Label> mergeeManifests =
        getMergeeManifests(
            resourceDeps.getResourceContainers(),
            dataContext.getAndroidConfig().getManifestMergerOrder());

    Artifact newManifest;
    if (useLegacyMerging(errorConsumer, dataContext.getAndroidConfig(), manifestMerger)) {
      newManifest =
          androidSemantics
              .maybeDoLegacyManifestMerging(mergeeManifests, dataContext, manifest)
              .orElse(manifest);

    } else if (!mergeeManifests.isEmpty() || !manifestValues.isEmpty()) {
      newManifest = dataContext.getUniqueDirectoryArtifact("_merged", "AndroidManifest.xml");

      new ManifestMergerActionBuilder()
          .setManifest(manifest)
          .setMergeeManifests(mergeeManifests)
          .setLibrary(false)
          .setManifestValues(manifestValues)
          .setCustomPackage(pkg)
          .setManifestOutput(newManifest)
          .setLogOut(dataContext.getUniqueDirectoryArtifact("_merged", "manifest_merger_log.txt"))
          .build(dataContext);

    } else {
      newManifest = manifest;
    }

    return new StampedAndroidManifest(newManifest, pkg, exported);
  }

  /**
   * Checks if the legacy manifest merger should be used, based on an optional string specifying the
   * merger to use.
   */
  private static boolean useLegacyMerging(
      RuleErrorConsumer errorConsumer,
      AndroidConfiguration androidConfig,
      @Nullable String mergerString) {
    if (androidConfig.getManifestMerger() == AndroidManifestMerger.FORCE_ANDROID) {
      return false;
    }

    AndroidManifestMerger merger = AndroidManifestMerger.fromString(mergerString);
    if (merger == null) {
      merger = androidConfig.getManifestMerger();
    }
    if (merger == AndroidManifestMerger.LEGACY) {
      errorConsumer.ruleWarning(
          "manifest_merger 'legacy' is deprecated. Please update to 'android'.\n"
              + "See https://developer.android.com/studio/build/manifest-merge.html for more "
              + "information about the manifest merger.");
    }

    return merger == AndroidManifestMerger.LEGACY;
  }

  private static Map<Artifact, Label> getMergeeManifests(
      NestedSet<ValidatedAndroidResources> transitiveData,
      AndroidConfiguration.ManifestMergerOrder manifestMergerOrder) {
    ImmutableMap.Builder<Artifact, Label> builder = new ImmutableMap.Builder<>();
    for (ValidatedAndroidResources d : transitiveData.toList()) {
      if (d.isManifestExported()) {
        builder.put(d.getManifest(), d.getLabel());
      }
    }
    switch (manifestMergerOrder) {
      case ALPHABETICAL:
        return ImmutableSortedMap.copyOf(builder.build(), Artifact.EXEC_PATH_COMPARATOR);
      case ALPHABETICAL_BY_CONFIGURATION:
        return ImmutableSortedMap.copyOf(builder.build(), Artifact.ROOT_RELATIVE_PATH_COMPARATOR);
      case DEPENDENCY:
        return builder.build();
    }
    throw new AssertionError(manifestMergerOrder);
  }

  public Artifact getManifest() {
    return manifest;
  }

  @Nullable
  String getPackage() {
    return pkg;
  }

  boolean isExported() {
    return exported;
  }

  /** Gets the Android package for this target, from either rule configuration or Java package */
  private static String getAndroidPackage(RuleContext ruleContext) {
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified(CUSTOM_PACKAGE_ATTR)) {
      return ruleContext.attributes().get(CUSTOM_PACKAGE_ATTR, Type.STRING);
    }

    return getDefaultPackage(ruleContext.getLabel(), ruleContext, ruleContext);
  }

  /**
   * Gets the default Java package for this target, based on the path to it.
   *
   * <p>For example, target "//some/path/java/com/foo/bar:baz" will have the default Java package of
   * "com.foo.bar".
   *
   * <p>A rule error will be registered if this path does not contain a "java" or "javatests"
   * segment indicating where the package begins.
   *
   * <p>This method should not be called if the target specifies a custom package; in that case,
   * that package should be used instead.
   */
  public static String getDefaultPackage(
      Label label, ActionConstructionContext context, RuleErrorConsumer errorConsumer) {
    PathFragment dummyJar =
        // For backwards compatibility, also include the target's name in case it contains multiple
        // directories - for example, target "//foo/bar:java/baz/quux" is a legal one and results in
        // Java path of "baz/quux"
        context.getPackageDirectory().getRelative(label.getName() + "Dummy.jar");
    return getJavaPackageFromPath(context, errorConsumer, dummyJar);
  }

  /**
   * Gets the Java package of a JAR file based on it's path.
   *
   * <p>Bazel requires that all Java code (including Android code) be in a path prefixed with "java"
   * or "javatests" followed by the Java package; this method validates and takes advantage of that
   * requirement.
   *
   * @param jarPathFragment The path to a JAR file contained in the current BUILD file's directory.
   * @return the Java package, as a String
   */
  static String getJavaPackageFromPath(
      ActionConstructionContext context,
      RuleErrorConsumer errorConsumer,
      PathFragment jarPathFragment) {
    // TODO(bazel-team): JavaUtil.getJavaPackageName does not check to see if the path is valid.
    // So we need to check for the JavaRoot.
    if (JavaUtil.getJavaRoot(jarPathFragment) == null) {
      errorConsumer.ruleError(
          "The location of your BUILD file determines the Java package used for "
              + "Android resource processing. A directory named \"java\" or \"javatests\" will "
              + "be used as your Java source root and the path of your BUILD file relative to "
              + "the Java source root will be used as the package for Android resource "
              + "processing. The Java source root could not be determined for \""
              + context.getPackageDirectory()
              + "\". Move your BUILD file under a java or javatests directory, or set the "
              + "'custom_package' attribute.");
    }
    return JavaUtil.getJavaPackageName(jarPathFragment);
  }

  @Override
  public boolean equals(Object object) {
    if (!(object instanceof AndroidManifest)) {
      return false;
    }

    AndroidManifest other = (AndroidManifest) object;

    return manifest.equals(other.manifest)
        && Objects.equals(pkg, other.pkg)
        && exported == other.exported;
  }

  @Override
  public int hashCode() {
    // Hash the current class with the other fields to distinguish between this AndroidManifest and
    // classes that extend it.
    return Objects.hash(manifest, pkg, exported, getClass());
  }
}
