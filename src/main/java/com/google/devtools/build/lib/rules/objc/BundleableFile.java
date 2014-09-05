// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import com.google.devtools.build.xcode.util.Value;

/**
 * Represents a file which is processed to another file and bundled. It contains the
 * {@code Artifact} corresponding to the original file as well as the {@code Artifact} for the file
 * converted to its bundled form. Examples of files that fit this pattern are .strings and .xib
 * files.
 */
public final class BundleableFile extends Value<BundleableFile> {
  private final Artifact original;
  private final Artifact bundled;
  private final String bundlePath;

  private BundleableFile(Artifact original, Artifact bundled, String bundlePath) {
    super(new ImmutableMap.Builder<String, Object>()
        .put("original", original)
        .put("bundled", bundled)
        .put("bundlePath", bundlePath)
        .build());
    this.original = original;
    this.bundled = bundled;
    this.bundlePath = bundlePath;
  }

  private static String bundlePath(Artifact name) {
    PathFragment path = name.getRootRelativePath();
    String containingDir = path.getParentDirectory().getBaseName();
    return (containingDir.endsWith(".lproj") ? (containingDir + "/") : "") + path.getBaseName();
  }

  /**
   * Returns an instance corresponding to a strings file whose original form is
   * {@code originalFile}. The value returned by {@link #getBundled()} will be the plist file in
   * binary form.
   */
  private static BundleableFile fromStringsFile(RuleContext context, Artifact originalFile) {
    Artifact binaryFile = ObjcRuleClasses.artifactByAppendingToRootRelativePath(
        context, originalFile.getRootRelativePath(), ".binary");
    return new BundleableFile(originalFile, binaryFile, bundlePath(originalFile));
  }

  /**
   * Returns an instance corresponding to some xib file. The value returned by {@link #getBundled()}
   * will be the compiled file.
   */
  private static BundleableFile fromXibFile(RuleContext context, Artifact originalFile) {
    // Each .xib file is compiled to a single .nib file.
    Artifact nibFile = context.getRelatedArtifact(originalFile.getExecPath(), ".nib");
    return new BundleableFile(originalFile, nibFile, bundlePath(nibFile));
  }

  /**
   * Returns an instance for every file, if any, specified by the {@code resources} attribute of the
   * given rule.
   */
  public static Iterable<BundleableFile> resourceFilesFromRule(RuleContext context) {
    ImmutableList.Builder<BundleableFile> result = new ImmutableList.Builder<>();
    for (Artifact file : context.getPrerequisiteArtifacts("resources", Mode.TARGET)) {
      result.add(new BundleableFile(file, file, bundlePath(file)));
    }
    return result.build();
  }

  /**
   * Returns an instance for every file, if any, specified by the {@code strings} attribute of the
   * given rule.
   */
  public static Iterable<BundleableFile> stringsFilesFromRule(RuleContext context) {
    ImmutableList.Builder<BundleableFile> result = new ImmutableList.Builder<>();
    for (Artifact originalFile : context.getPrerequisiteArtifacts("strings", Mode.TARGET)) {
      result.add(fromStringsFile(context, originalFile));
    }
    return result.build();
  }

  /**
   * Returns an instance for every file, if any, specified by the {@code strings} attribute of the
   * given rule.
   */
  public static Iterable<BundleableFile> xibFilesFromRule(RuleContext context) {
    ImmutableList.Builder<BundleableFile> result = new ImmutableList.Builder<>();
    for (Artifact originalFile : context.getPrerequisiteArtifacts("xibs", Mode.TARGET)) {
      result.add(fromXibFile(context, originalFile));
    }
    return result.build();
  }

  /**
   * The checked-in version of the bundled file.
   */
  public Artifact getOriginal() {
    return original;
  }

  /**
   * The artifact that is ultimately bundled.
   */
  public Artifact getBundled() {
    return bundled;
  }

  /**
   * Returns bundle files for each given strings file. These are used to merge the strings files to
   * the final application bundle.
   */
  public static Iterable<BundleFile> toBundleFiles(Iterable<BundleableFile> files) {
    ImmutableList.Builder<BundleFile> result = new ImmutableList.Builder<>();
    for (BundleableFile file : files) {
      result.add(BundleFile.newBuilder()
          .setBundlePath(file.bundlePath)
          .setSourceFile(file.bundled.getExecPathString())
          .build());
    }
    return result.build();
  }

  /**
   * Returns the input files to add to the bundlemerge action for a bundle that contains all the
   * given strings files.
   */
  public static Iterable<Artifact> toBundleMergeInputs(Iterable<BundleableFile> files) {
    ImmutableList.Builder<Artifact> result = new ImmutableList.Builder<>();
    for (BundleableFile file : files) {
      result.add(file.bundled);
    }
    return result.build();
  }
}
