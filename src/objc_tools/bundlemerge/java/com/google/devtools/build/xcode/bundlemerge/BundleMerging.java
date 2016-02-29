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

package com.google.devtools.build.xcode.bundlemerge;

import static com.google.devtools.build.singlejar.ZipCombiner.DOS_EPOCH;
import static com.google.devtools.build.singlejar.ZipCombiner.OutputMode.FORCE_DEFLATE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.singlejar.ZipCombiner;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import com.google.devtools.build.xcode.plmerge.PlistMerging;
import com.google.devtools.build.xcode.zip.ZipFiles;
import com.google.devtools.build.xcode.zip.ZipInputEntry;
import com.google.devtools.build.zip.ZipFileEntry;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import javax.annotation.CheckReturnValue;

/**
 * Implementation of the final steps to create an iOS application bundle.
 *
 * <p>TODO(bazel-team): Add asset catalog compilation and bundling to this logic.
 */
public final class BundleMerging {
  @VisibleForTesting final FileSystem fileSystem;
  @VisibleForTesting final Path outputZip;
  @VisibleForTesting final ImmutableList<ZipInputEntry> inputs;
  @VisibleForTesting final ImmutableList<MergeZip> mergeZips;

  /**
   * We can instantiate this class for testing purposes. For typical uses, just use
   * {@link #merge(FileSystem, Control)}.
   */
  private BundleMerging(FileSystem fileSystem,
      Path outputZip, ImmutableList<ZipInputEntry> inputs, ImmutableList<MergeZip> mergeZips) {
    this.fileSystem = Preconditions.checkNotNull(fileSystem);
    this.outputZip = Preconditions.checkNotNull(outputZip);
    this.inputs = Preconditions.checkNotNull(inputs);
    this.mergeZips = Preconditions.checkNotNull(mergeZips);
  }

  /**
   * Joins two paths to be used in a zip file. The {@code right} part of the path must be relative.
   * The {@code left} part could or could not have a trailing slash. These paths are used in .ipa
   * (.zip) files, which must use forward slashes, so they are hard-coded here.
   * <p>
   * TODO(bazel-team): This is messy. See if we can use some common joining function that handles
   * empty paths and doesn't automatically inherit the path conventions of the host platform.
   */
  private static String joinPath(String left, String right) {
    Preconditions.checkArgument(!right.startsWith("/"), "'right' must be relative: %s", right);
    if (left.isEmpty() || right.isEmpty() || left.endsWith("/")) {
      return left + right;
    } else {
      return left + "/" + right;
    }
  }

  private static final String INFOPLIST_FILENAME = "Info.plist";
  private static final String PKGINFO_FILENAME = "PkgInfo";

  /**
   * Adds merge artifacts from the given {@code control} into builders that collect merge zips and
   * individual files. {@code bundleRoot} is prepended to each path, except the paths in the merge
   * zips.
   */
  private static void mergeInto(
      Path tempDir, FileSystem fileSystem, Control control, String bundleRoot,
      ImmutableList.Builder<ZipInputEntry> packagedFilesBuilder,
      ImmutableList.Builder<MergeZip> mergeZipsBuilder, boolean includePkgInfo) throws IOException {
    Path tempMergedPlist = Files.createTempFile(tempDir, null, INFOPLIST_FILENAME);
    Path tempPkgInfo = Files.createTempFile(tempDir, null, PKGINFO_FILENAME);

    if (control.hasBundleInfoPlistFile()) {
      Path bundleInfoPlist = fileSystem.getPath(control.getBundleInfoPlistFile());
      new PlistMerging(PlistMerging.readPlistFile(bundleInfoPlist))
          .setBundleIdentifier(
              control.hasPrimaryBundleIdentifier() ? control.getPrimaryBundleIdentifier() : null,
              control.hasFallbackBundleIdentifier() ? control.getFallbackBundleIdentifier() : null)
          .writePlist(tempMergedPlist)
          .writePkgInfo(tempPkgInfo);
    }

    bundleRoot = joinPath(bundleRoot, control.getBundleRoot());

    // Add files to zip configuration which creates the final application bundle.
    packagedFilesBuilder
        .add(new ZipInputEntry(tempMergedPlist, joinPath(bundleRoot, INFOPLIST_FILENAME)));
    if (includePkgInfo) {
      packagedFilesBuilder
          .add(new ZipInputEntry(tempPkgInfo, joinPath(bundleRoot, PKGINFO_FILENAME)));
    }
    for (BundleFile bundleFile : control.getBundleFileList()) {
      int externalFileAttribute = bundleFile.hasExternalFileAttribute()
          ? bundleFile.getExternalFileAttribute() : ZipInputEntry.DEFAULT_EXTERNAL_FILE_ATTRIBUTE;
      packagedFilesBuilder.add(
          new ZipInputEntry(
              fileSystem.getPath(bundleFile.getSourceFile()),
              joinPath(bundleRoot, bundleFile.getBundlePath()),
              externalFileAttribute));
    }

    mergeZipsBuilder.addAll(control.getMergeZipList());

    for (Control nestedControl : control.getNestedBundleList()) {
      mergeInto(tempDir, fileSystem, nestedControl, bundleRoot, packagedFilesBuilder,
          mergeZipsBuilder, /*includePkgInfo=*/false);
    }
  }

  /**
   * Returns a zipper configuration that can be executed to create the application bundle.
   */
  @CheckReturnValue
  @VisibleForTesting
  static BundleMerging merging(Path tempDir, FileSystem fileSystem, Control control)
      throws IOException {
    ImmutableList.Builder<MergeZip> mergeZipsBuilder = new ImmutableList.Builder<>();
    ImmutableList.Builder<ZipInputEntry> packagedFilesBuilder =
        new ImmutableList.Builder<ZipInputEntry>();

    mergeInto(tempDir, fileSystem, control, /*bundleRoot=*/"", packagedFilesBuilder,
        mergeZipsBuilder, /*includePkgInfo=*/true);

    return new BundleMerging(fileSystem, fileSystem.getPath(control.getOutFile()),
        packagedFilesBuilder.build(), mergeZipsBuilder.build());
  }

  /**
   * Copies all entries from the source zip into a destination zip using the given combiner. The
   * contents of the source zip can appear to be in a sub-directory of the destination zip by
   * passing a non-empty string for the entry names prefix with a trailing '/'.
   */
  private void addEntriesFromOtherZip(ZipCombiner combiner, Path sourceZip, String entryNamesPrefix)
      throws IOException {
    Map<String, Integer> externalFileAttributes = ZipFiles.unixExternalFileAttributes(sourceZip);
    try (ZipInputStream zipIn = new ZipInputStream(Files.newInputStream(sourceZip))) {
      while (true) {
        ZipEntry zipInEntry = zipIn.getNextEntry();
        if (zipInEntry == null) {
          break;
        }
        // TODO(bazel-dev): Add support for soft links because we will need them for MacOS support
        // in frameworks at the very least. https://github.com/bazelbuild/bazel/issues/289
        String name = entryNamesPrefix + zipInEntry.getName();
        if (zipInEntry.isDirectory()) {
          // If we already have a directory entry with this name then don't attempt to
          // add it again. It's not an error to attempt to merge in two zip files that contain
          // the same directories. It's only an error to attempt to merge in two zip files with the
          // same leaf files.
          if (!combiner.containsFile(name)) {
            combiner.addDirectory(name, DOS_EPOCH);
          }
          continue;
        }
        Integer externalFileAttr = externalFileAttributes.get(zipInEntry.getName());
        if (externalFileAttr == null) {
          externalFileAttr = ZipInputEntry.DEFAULT_EXTERNAL_FILE_ATTRIBUTE;
        }
        ZipFileEntry zipOutEntry = new ZipFileEntry(name);
        zipOutEntry.setTime(DOS_EPOCH.getTime());
        zipOutEntry.setVersion(ZipInputEntry.MADE_BY_VERSION);
        zipOutEntry.setExternalAttributes(externalFileAttr);
        combiner.addFile(zipOutEntry, zipIn);
      }
    }
  }  
  
  @VisibleForTesting
  void execute() throws IOException {
    try (OutputStream out = Files.newOutputStream(outputZip);
        ZipCombiner combiner = new ZipCombiner(FORCE_DEFLATE, out)) {
      ZipInputEntry.addAll(combiner, inputs);
      for (MergeZip mergeZip : mergeZips) {
        addEntriesFromOtherZip(
            combiner, fileSystem.getPath(mergeZip.getSourcePath()), mergeZip.getEntryNamePrefix());
      }
    }
  }

  /**
   * Creates an {@code .ipa} file for an iOS application.
   * @param fileSystem used to resolve paths specified in {@code control} 
   * @param control specifies the locations of input and output files and other parameters used to
   *     create the final {@code .ipa} file.
   */
  public static void merge(FileSystem fileSystem, Control control) throws IOException {
    merging(Files.createTempDirectory("mergebundle"), fileSystem, control).execute();
  }
}
