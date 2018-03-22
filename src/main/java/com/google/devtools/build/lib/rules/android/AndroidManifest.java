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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** Wraps an Android Manifest and provides utilities for working with it */
@Immutable
public class AndroidManifest {
  private static final SafeImplicitOutputsFunction MERGED_MANIFEST =
      ImplicitOutputsFunction.fromTemplates("{name}_manifest/AndroidManifest.xml");
  private static final SafeImplicitOutputsFunction MERGE_LOG =
      ImplicitOutputsFunction.fromTemplates("{name}_manifest/manifest_merger_log.txt");

  private final RuleContext ruleContext;
  /**
   * The current manifest. May be null if this rule has no manifest and we have not yet generated
   * one through merging or stamping.
   */
  @Nullable private final Artifact manifest;
  private final String pkg;
  private final boolean isDummy;

  public static AndroidManifest empty(RuleContext ruleContext) {
    return of(ruleContext, null, null);
  }

  /**
   * @param ruleContext the current context
   * @param manifest this target's manifest. Can be null if this target has no manifest, in which
   *     case a dummy manifest will be generated.
   * @param customPackage this target's custom package. If null, the default package, derived from
   *     BUILD file location, will be used.
   * @return an AndroidManifest object wrapping the manifest and package
   */
  public static AndroidManifest of(
      RuleContext ruleContext, @Nullable Artifact manifest, @Nullable String customPackage) {
    return new AndroidManifest(
        ruleContext,
        manifest,
        customPackage == null ? getDefaultPackage(ruleContext) : customPackage,
        manifest == null);
  }

  AndroidManifest(
      RuleContext ruleContext, @Nullable Artifact manifest, String pkg, boolean isDummy) {
    this.ruleContext = ruleContext;
    this.manifest = manifest;
    this.pkg = pkg;
    this.isDummy = isDummy;
  }

  /**
   * Merges the current manifest with manifests from the specified deps and stamps the result.
   *
   * <p>Manifests will not be merged if the dependencies do not provide {@link AndroidManifestInfo}
   * or the provided manifests are generated dummies.
   *
   * <p>The resulting manifest will always be stamped if needed, even if no merging is done.
   */
  public StampedAndroidManifest stampAndMergeWith(ImmutableList<ConfiguredTarget> deps)
      throws InterruptedException {
    ImmutableMap.Builder<Artifact, Label> mergeeBuilder = ImmutableMap.builder();
    for (ConfiguredTarget dep : deps) {
      AndroidManifestInfo info = dep.get(AndroidManifestInfo.PROVIDER);
      if (info == null || info.isDummy()) {
        continue;
      }
      mergeeBuilder.put(info.getManifest(), dep.getLabel());
    }

    ImmutableMap<Artifact, Label> mergeeManifests = mergeeBuilder.build();

    if (mergeeManifests.isEmpty()) {
      return stamp();
    }

    Artifact merged = ruleContext.getImplicitOutputArtifact(MERGED_MANIFEST);

    // Since we're already invoking an action to merge, we may as well stamp here as well.
    getActionBuilder(merged)
        .setMergeeManifests(mergeeManifests)
        .setLogOut(ruleContext.getImplicitOutputArtifact(MERGE_LOG))
        .build(ruleContext);

    return new StampedAndroidManifest(ruleContext, merged, pkg, false);
  }

  /** If needed, stamps the manifest with the correct Java package */
  StampedAndroidManifest stamp() throws InterruptedException {
    Artifact stamped = ruleContext.getImplicitOutputArtifact(MERGED_MANIFEST);

    getActionBuilder(stamped).build(ruleContext);

    return new StampedAndroidManifest(ruleContext, stamped, pkg, isDummy);
  }

  /**
   * Gets the manifest artifact wrapped by this object. May be null if the manifest is to be
   * generated but has not been.
   */
  @Nullable
  Artifact getManifest() {
    return manifest;
  }

  String getPackage() {
    return pkg;
  }

  boolean isDummy() {
    return isDummy;
  }

  /** Gets a {@link ManifestMergerActionBuilder} with common settings always used by this object. */
  private ManifestMergerActionBuilder getActionBuilder(Artifact manifestOutput) {
    return new ManifestMergerActionBuilder(ruleContext)
        .setCustomPackage(pkg)
        // The current manifest merger action uses the "custom_package" value when working on
        // "library" targets, and ignores it and removes any tool annotations from the manifest
        // otherwise. As this method is not intended to produce a final merged manifest, even when
        // run on a binary, always use the "library" settings here.
        .setLibrary(true)
        .setManifest(manifest)
        .setManifestOutput(manifestOutput);
  }

  /** Gets the default Java package */
  static String getDefaultPackage(RuleContext ruleContext) {
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
}
