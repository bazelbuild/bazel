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
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** Wraps an Android Manifest and provides utilities for working with it */
@Immutable
public class AndroidManifest {

  private final Artifact manifest;
  /** The Android package. Will be null if and only if this is an aar_import target. */
  @Nullable private final String pkg;

  public static AndroidManifest forAarImport(RuleContext ruleContext, Artifact manifest) {
    return new AndroidManifest(manifest, null);
  }

  AndroidManifest(Artifact manifest, @Nullable String pkg) {
    this.manifest = manifest;
    this.pkg = pkg;
  }

  /** If needed, stamps the manifest with the correct Java package */
  public StampedAndroidManifest stamp(RuleContext ruleContext) {
    return new StampedAndroidManifest(
        ApplicationManifest.maybeSetManifestPackage(ruleContext, manifest, pkg).orElse(manifest),
        pkg);
  }

  Artifact getManifest() {
    return manifest;
  }

  @Nullable
  String getPackage() {
    return pkg;
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
