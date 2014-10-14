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

import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.BUNDLE_IMPORTS;
import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.RESOURCES;
import static com.google.devtools.build.lib.rules.objc.ObjcCommon.BUNDLE_CONTAINER_TYPE;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.PathFragment;
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

  private final Artifact bundled;
  private final String bundlePath;

  BundleableFile(Artifact bundled, String bundlePath) {
    super(new ImmutableMap.Builder<String, Object>()
        .put("bundled", bundled)
        .put("bundlePath", bundlePath)
        .build());
    this.bundled = bundled;
    this.bundlePath = bundlePath;
  }

  static String bundlePath(Artifact name) {
    PathFragment path = name.getRootRelativePath();
    String containingDir = path.getParentDirectory().getBaseName();
    return (containingDir.endsWith(".lproj") ? (containingDir + "/") : "") + path.getBaseName();
  }

  /**
   * Returns an instance for every file, if any, specified by the {@code resources} attribute of the
   * given rule.
   */
  public static Iterable<BundleableFile> resourceFilesFromRule(RuleContext context) {
    ImmutableList.Builder<BundleableFile> result = new ImmutableList.Builder<>();
    for (Artifact file : RESOURCES.get(context)) {
      result.add(new BundleableFile(file, bundlePath(file)));
    }
    return result.build();
  }

  /**
   * Returns an instance for every file in a bundle directory.
   * <p>
   * This uses the parent-most container matching {@code *.bundle} as the bundle root.
   * TODO(bazel-team): add something like an import_root attribute to specify this explicitly, which
   * will be helpful if a bundle that appears to be nested needs to be imported alone.
   */
  public static Iterable<BundleableFile> bundleImportsFromRule(RuleContext context) {
    ImmutableList.Builder<BundleableFile> result = new ImmutableList.Builder<>();
    for (Artifact artifact : BUNDLE_IMPORTS.get(context)) {
      for (PathFragment container :
          ObjcCommon.farthestContainerMatching(BUNDLE_CONTAINER_TYPE, artifact).asSet()) {
        // TODO(bazel-team): Figure out if we need to remove symbols of architectures we aren't 
        // building for from the binary in the bundle.
        result.add(new BundleableFile(
            artifact,
            // The path from the artifact's container (including the container), to the artifact
            // itself. For instance, if artifact is foo/bar.bundle/baz, then this value
            // is bar.bundle/baz.
            artifact.getExecPath().relativeTo(container.getParentDirectory()).getSafePathString()));
      }
    }
    return result.build();
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
   * Returns the artifacts for the bundled files. These can be used, for instance, as the input
   * files to add to the bundlemerge action for a bundle that contains all the given files.
   */
  public static Iterable<Artifact> toArtifacts(Iterable<BundleableFile> files) {
    ImmutableList.Builder<Artifact> result = new ImmutableList.Builder<>();
    for (BundleableFile file : files) {
      result.add(file.bundled);
    }
    return result.build();
  }
}
