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

package com.google.devtools.build.xcode.bundlemerge;

import static com.google.devtools.build.singlejar.ZipCombiner.DOS_EPOCH;
import static com.google.devtools.build.singlejar.ZipCombiner.OutputMode.FORCE_DEFLATE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.singlejar.ZipCombiner;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import com.google.devtools.build.xcode.common.Platform;
import com.google.devtools.build.xcode.common.TargetDeviceFamily;
import com.google.devtools.build.xcode.plmerge.PlistMerging;
import com.google.devtools.build.xcode.zip.ZipInputEntry;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import javax.annotation.CheckReturnValue;

/**
 * Implementation of the final steps to create an iOS application bundle.
 *
 * TODO(bazel-team): Add asset catalog compilation and bundling to this logic.
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
   * Returns a zipper configuration that can be executed to create the application bundle.
   */
  @CheckReturnValue
  @VisibleForTesting
  static BundleMerging merging(Path tempDir, FileSystem fileSystem, Control control)
      throws IOException {
    Path output = fileSystem.getPath(control.getOutFile());
    Path tempMergedPlist = Files.createTempFile(tempDir, null, "Info.plist");
    Path tempPkgInfo = Files.createTempFile(tempDir, null, "PkgInfo");

    // Generate the Info.plist and PkgInfo files to include in the app bundle.
    ImmutableList.Builder<Path> sourcePlistFilesBuilder = new ImmutableList.Builder<>();
    for (String sourcePlist : control.getSourcePlistFileList()) {
      sourcePlistFilesBuilder.add(fileSystem.getPath(sourcePlist));
    }
    ImmutableList<Path> sourcePlistFiles = sourcePlistFilesBuilder.build();
    ImmutableSet.Builder<TargetDeviceFamily> targetDeviceFamiliesBuilder =
        new ImmutableSet.Builder<>();
    for (String targetDeviceFamily : control.getTargetDeviceFamilyList()) {
      targetDeviceFamiliesBuilder.add(TargetDeviceFamily.valueOf(targetDeviceFamily));
    }
    PlistMerging
        .from(
            sourcePlistFiles,
            PlistMerging.automaticEntries(
                targetDeviceFamiliesBuilder.build(),
                Platform.valueOf(control.getPlatform()),
                control.getSdkVersion(),
                control.getMinimumOsVersion()))
        .write(tempMergedPlist, tempPkgInfo);

    // Generate zip configuration which creates the final application bundle.
    String bundleRootSlash = control.getBundleRoot();
    if (!bundleRootSlash.endsWith("/")) {
      bundleRootSlash = bundleRootSlash + "/";
    }
    ImmutableList.Builder<ZipInputEntry> packagedFilesBuilder =
        new ImmutableList.Builder<ZipInputEntry>()
            .add(new ZipInputEntry(tempMergedPlist, bundleRootSlash + "Info.plist"))
            .add(new ZipInputEntry(tempPkgInfo, bundleRootSlash + "PkgInfo"));
    for (BundleFile bundleFile : control.getBundleFileList()) {
      packagedFilesBuilder.add(new ZipInputEntry(fileSystem.getPath(bundleFile.getSourceFile()),
          bundleRootSlash + bundleFile.getBundlePath()));
    }

    ImmutableList.Builder<MergeZip> mergeZipsBuilder = new ImmutableList.Builder<>();
    for (String mergeZip : control.getMergeWithoutNamePrefixZipList()) {
      mergeZipsBuilder.add(MergeZip.newBuilder()
          .setSourcePath(mergeZip)
          .build());
    }
    mergeZipsBuilder.addAll(control.getMergeZipList());
    return new BundleMerging(
        fileSystem, output, packagedFilesBuilder.build(), mergeZipsBuilder.build());
  }

  /**
   * Copies all entries from the source zip into a destination zip using the given combiner. The
   * contents of the source zip can appear to be in a sub-directory of the destination zip by
   * passing a non-empty string for the entry names prefix with a trailing '/'.
   */
  private void addEntriesFromOtherZip(ZipCombiner combiner, Path sourceZip, String entryNamesPrefix)
      throws IOException {
    try (ZipInputStream zipIn = new ZipInputStream(Files.newInputStream(sourceZip))) {
      while (true) {
        ZipEntry zipInEntry = zipIn.getNextEntry();
        if (zipInEntry == null) {
          break;
        }
        combiner.addFile(entryNamesPrefix + zipInEntry.getName(), DOS_EPOCH, zipIn);
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
