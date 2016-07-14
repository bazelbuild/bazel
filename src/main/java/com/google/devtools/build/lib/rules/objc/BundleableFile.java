// Copyright 2014 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.rules.objc.ObjcCommon.BUNDLE_CONTAINER_TYPE;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;

/**
 * Represents a file which is processed to another file and bundled. It contains the
 * {@code Artifact} corresponding to the original file as well as the {@code Artifact} for the file
 * converted to its bundled form. Examples of files that fit this pattern are .strings and .xib
 * files.
 */
public final class BundleableFile extends Value<BundleableFile> {
  static final int EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE = 0100755 << 16;
  static final int DEFAULT_EXTERNAL_FILE_ATTRIBUTE = 0100644 << 16;

  /** The field in the Skylark struct that holds the {@code bundled} artifact. */
  static final String BUNDLED_FIELD = "file";

  /** The field in the Skylark struct that holds the {@code bundlePath} string. */
  static final String BUNDLE_PATH_FIELD = "bundle_path";

  private final Artifact bundled;
  private final String bundlePath;
  private final int zipExternalFileAttribute;

  /**
   * Creates an instance whose {@code zipExternalFileAttribute} value is
   * {@link #DEFAULT_EXTERNAL_FILE_ATTRIBUTE}.
   */
  BundleableFile(Artifact bundled, String bundlePath) {
    this(bundled, bundlePath, DEFAULT_EXTERNAL_FILE_ATTRIBUTE);
  }

  /**
   * @param bundled the {@link Artifact} whose data is placed in the bundle
   * @param bundlePath the path of the file in the bundle
   * @param zipExternalFileAttribute external file attribute of the file in the central directory of
   *     the bundle (zip file). The lower 16 bits contain the MS-DOS file attributes. The upper 16
   *     bits contain the Unix file attributes, for instance 0100755 (octal) for a regular file with
   *     permissions {@code rwxr-xr-x}.
   */
  BundleableFile(Artifact bundled, String bundlePath, int zipExternalFileAttribute) {
    super(new ImmutableMap.Builder<String, Object>()
        .put("bundled", bundled)
        .put("bundlePath", bundlePath)
        .put("zipExternalFileAttribute", zipExternalFileAttribute)
        .build());
    this.bundled = bundled;
    this.bundlePath = bundlePath;
    this.zipExternalFileAttribute = zipExternalFileAttribute;
  }

  static String flatBundlePath(PathFragment path) {
    String containingDir = path.getParentDirectory().getBaseName();
    return (containingDir.endsWith(".lproj") ? (containingDir + "/") : "") + path.getBaseName();
  }

  /**
   * Given a sequence of non-compiled resource files, returns a sequence of the same length of
   * instances of this class with the resource paths flattened (resources are put in the bundle
   * root) or, if the source file is in a directory ending in {@code .lproj}, in a directory of that
   * name directly under the bundle root.
   *
   * <p>Non-compiled resource files are resources which are not processed before placing them in the
   * final bundle. This is different from (for example) {@code .strings} and {@code .xib} files,
   * which must be converted to binary plist form or compiled.
   *
   * @param files a sequence of artifacts corresponding to non-compiled resource files
   */
  public static Iterable<BundleableFile> flattenedRawResourceFiles(Iterable<Artifact> files) {
    ImmutableList.Builder<BundleableFile> result = new ImmutableList.Builder<>();
    for (Artifact file : files) {
      result.add(new BundleableFile(file, flatBundlePath(file.getExecPath())));
    }
    return result.build();
  }

  /**
   * Given a sequence of non-compiled resource files, returns a sequence of the same length of
   * instances of this class with the resource paths copied as-is into the bundle root.
   *
   * <p>Non-compiled resource files are resources which are not processed before placing them in the
   * final bundle. This is different from (for example) {@code .strings} and {@code .xib} files,
   * which must be converted to binary plist form or compiled.
   *
   * @param files a sequence of artifacts corresponding to non-compiled resource files
   */
  public static Iterable<BundleableFile> structuredRawResourceFiles(Iterable<Artifact> files) {
    ImmutableList.Builder<BundleableFile> result = new ImmutableList.Builder<>();
    for (Artifact file : files) {
      result.add(new BundleableFile(file, ownerBundlePath(file)));
    }
    return result.build();
  }

  private static String ownerBundlePath(Artifact file) {
    PathFragment packageFragment = file.getArtifactOwner().getLabel().getPackageFragment();
    return file.getRootRelativePath().relativeTo(packageFragment).toString();
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
   * Returns the location into which this file should be put in, relative to the bundle root.
   */
  public String getBundlePath() {
    return bundlePath;
  }

  /**
   * Returns the artifact representing the source for this bundleable file.
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
          .setExternalFileAttribute(file.zipExternalFileAttribute)
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
