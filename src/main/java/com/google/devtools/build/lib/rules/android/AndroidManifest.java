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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.syntax.Type;
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
   * <p>Unlike {@link #fromAttributes(RuleContext, AndroidSemantics)}, the AndroidSemantics-specific
   * manifest processing methods will not be applied in this method. The manifest returned by this
   * method will be the same regardless of the AndroidSemantics being used.
   */
  public static AndroidManifest fromAttributes(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    return fromAttributes(ruleContext, null);
  }

  /**
   * Gets the manifest for this rule.
   *
   * <p>If no manifest is specified in the rule's attributes, an empty manifest will be generated.
   *
   * <p>If a non-null {@link AndroidSemantics} is passed, AndroidSemantics-specific manifest
   * processing will be preformed on this manifest. Otherwise, basic manifest renaming will be
   * performed if needed.
   *
   */
  public static AndroidManifest fromAttributes(
      RuleContext ruleContext, @Nullable AndroidSemantics androidSemantics)
      throws RuleErrorException, InterruptedException {
    Artifact rawManifest = null;
    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      AndroidResources.validateRuleContext(ruleContext);
      rawManifest = ApplicationManifest.getManifestFromAttributes(ruleContext);
    }

    return from(
        ruleContext,
        rawManifest,
        androidSemantics,
        getAndroidPackage(ruleContext),
        AndroidCommon.getExportsManifest(ruleContext));
  }

  /**
   * Creates an AndroidManifest object, with correct preprocessing, from explicit variables.
   *
   * <p>Attributes included in the RuleContext will not be used; use {@link #from(RuleContext)}
   * instead.
   *
   * <p>In addition, the AndroidSemantics-specific manifest processing methods will not be applied
   * in this method. The manifest returned by this method will be the same regardless of the
   * AndroidSemantics being used. use {@link #from(RuleContext, AndroidSemantics)} instead if you
   * want AndroidSemantics-specific behavior.
   */
  public static AndroidManifest from(
      RuleContext ruleContext,
      @Nullable Artifact rawManifest,
      @Nullable String pkg,
      boolean exportsManifest)
      throws InterruptedException {
    return from(ruleContext, rawManifest, null, pkg, exportsManifest);
  }

  /**
   * Inner method to create an AndroidManifest.
   *
   * <p>AndroidSemantics-specific processing will be used if a non-null AndroidSemantics is passed.
   */
  static AndroidManifest from(
      RuleContext ruleContext,
      @Nullable Artifact rawManifest,
      @Nullable AndroidSemantics androidSemantics,
      @Nullable String pkg,
      boolean exportsManifest)
      throws InterruptedException {
    if (pkg == null) {
      pkg = getDefaultPackage(ruleContext);
    }

    if (rawManifest == null) {
      // Generate a dummy manifest
      return StampedAndroidManifest.createEmpty(ruleContext, pkg, /* exported = */ false);
    }

    Artifact renamedManifest;
    if (androidSemantics != null) {
      renamedManifest = androidSemantics.renameManifest(ruleContext, rawManifest);
    } else {
      renamedManifest = ApplicationManifest.renameManifestIfNeeded(ruleContext, rawManifest);
    }

    return new AndroidManifest(renamedManifest, pkg, exportsManifest);
  }

  AndroidManifest(AndroidManifest other, Artifact manifest) {
    this(manifest, other.pkg, other.exported);
  }

  /**
   * Creates a manifest wrapper without doing any processing. From within a rule, use {@link
   * #from(RuleContext, AndroidSemantics)} instead.
   */
  public AndroidManifest(Artifact manifest, @Nullable String pkg, boolean exported) {
    this.manifest = manifest;
    this.pkg = pkg;
    this.exported = exported;
  }

  /** If needed, stamps the manifest with the correct Java package */
  public StampedAndroidManifest stamp(RuleContext ruleContext) {
    return new StampedAndroidManifest(
        ApplicationManifest.maybeSetManifestPackage(ruleContext, manifest, pkg).orElse(manifest),
        pkg,
        exported);
  }

  /**
   * Stamps the manifest with values from the "manifest_values" attributes.
   *
   * <p>If no manifest values are specified, the manifest will remain unstamped.
   */
  public StampedAndroidManifest stampWithManifestValues(RuleContext ruleContext) {
    return mergeWithDeps(
        ruleContext,
        ResourceDependencies.empty(),
        ApplicationManifest.getManifestValues(ruleContext),
        ApplicationManifest.useLegacyMerging(ruleContext));
  }

  /**
   * Merges the manifest with any dependent manifests, extracted from rule attributes.
   *
   * <p>The manifest will also be stamped with any manifest values specified in the target's
   * attributes
   *
   * <p>If there is no merging to be done and no manifest values are specified, the manifest will
   * remain unstamped.
   */
  public StampedAndroidManifest mergeWithDeps(RuleContext ruleContext) {
    return mergeWithDeps(
        ruleContext,
        ResourceDependencies.fromRuleDeps(ruleContext, /* neverlink = */ false),
        ApplicationManifest.getManifestValues(ruleContext),
        ApplicationManifest.useLegacyMerging(ruleContext));
  }

  public StampedAndroidManifest mergeWithDeps(
      RuleContext ruleContext,
      ResourceDependencies resourceDeps,
      Map<String, String> manifestValues,
      boolean useLegacyMerger) {
    Artifact newManifest =
        ApplicationManifest.maybeMergeWith(
            ruleContext, manifest, resourceDeps, manifestValues, useLegacyMerger, pkg)
            .orElse(manifest);

    return new StampedAndroidManifest(newManifest, pkg, exported);
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

    return getDefaultPackage(ruleContext);
  }

  /** Gets the default Java package */
  public static String getDefaultPackage(RuleContext ruleContext) {
    PathFragment dummyJar = ruleContext.getPackageDirectory().getChild("Dummy.jar");
    return getJavaPackageFromPath(ruleContext, dummyJar);
  }

  /**
   * Gets the Java package of a JAR file based on it's path.
   *
   * <p>Bazel requires that all Java code (including Android code) be in a path prefixed with "java"
   * or "javatests" followed by the Java package; this method validates and takes advantage of that
   * requirement.
   *
   * @param ruleContext the current context
   * @param jarPathFragment The path to a JAR file contained in the current BUILD file's directory.
   * @return the Java package, as a String
   */
  static String getJavaPackageFromPath(RuleContext ruleContext, PathFragment jarPathFragment) {
    // TODO(bazel-team): JavaUtil.getJavaPackageName does not check to see if the path is valid.
    // So we need to check for the JavaRoot.
    if (JavaUtil.getJavaRoot(jarPathFragment) == null) {
      ruleContext.ruleError(
          "The location of your BUILD file determines the Java package used for "
              + "Android resource processing. A directory named \"java\" or \"javatests\" will "
              + "be used as your Java source root and the path of your BUILD file relative to "
              + "the Java source root will be used as the package for Android resource "
              + "processing. The Java source root could not be determined for \""
              + ruleContext.getPackageDirectory()
              + "\". Move your BUILD file under a java or javatests directory, or set the "
              + "'custom_package' attribute.");
    }
    return JavaUtil.getJavaPackageName(jarPathFragment);
  }

  @Override
  public boolean equals(Object object) {
    if (object == null || getClass() != object.getClass()) {
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
