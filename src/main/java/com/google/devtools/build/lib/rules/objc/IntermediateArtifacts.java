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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.AnalysisEnvironment;
import com.google.devtools.build.lib.view.AnalysisUtils;

/**
 * Factory class for generating artifacts which are used as intermediate output.
 */
final class IntermediateArtifacts {
  private final AnalysisEnvironment analysisEnvironment;
  private final Root binDirectory;
  private final Label ownerLabel;

  IntermediateArtifacts(
      AnalysisEnvironment analysisEnvironment, Root binDirectory, Label ownerLabel) {
    this.analysisEnvironment = Preconditions.checkNotNull(analysisEnvironment);
    this.binDirectory = Preconditions.checkNotNull(binDirectory);
    this.ownerLabel = Preconditions.checkNotNull(ownerLabel);
  }

  public AnalysisEnvironment getAnalysisEnvironment() {
    return analysisEnvironment;
  }

  public Root getBinDirectory() {
    return binDirectory;
  }

  public Label getOwnerLabel() {
    return ownerLabel;
  }

  /**
   * Returns a derived artifact in the bin directory obtained by appending some extension to the end
   * of the given {@link PathFragment}.
   */
  private Artifact appendExtension(PathFragment original, String extension) {
    return analysisEnvironment.getDerivedArtifact(
        FileSystemUtils.appendExtension(original, extension), binDirectory);
  }

  /**
   * Returns a derived artifact in the bin directory obtained by appending some extension to the end
   * of the {@link PathFragment} corresponding to the owner {@link Label}.
   */
  private Artifact appendExtension(String extension) {
    return appendExtension(ownerLabel.toPathFragment(), extension);
  }

  /**
   * The output of using {@code actooloribtoolzip} to run {@code actool} for a given bundle which is
   * merged under the {@code .app} or {@code .bundle} directory root.
   */
  public Artifact actoolzipOutput() {
    return appendExtension(".actool.zip");
  }

  /**
   * The Info.plist file for a bundle which is comprised of more than one originating plist file.
   * This is not needed for a bundle which has no source Info.plist files, or only one Info.plist
   * file, since no merging occurs in that case.
   */
  public Artifact mergedInfoplist() {
    return appendExtension("-MergedInfo.plist");
  }

  /**
   * The .objlist file, which contains a list of paths of object files to archive  and is read by
   * libtool in the archive action.
   */
  public Artifact objList() {
    return appendExtension(".objlist");
  }

  /**
   * The artifact which is the binary (or library) which is comprised of one or more .a files linked
   * together. There is either 0 or 1 such binary in a given bundle.
   */
  public Artifact linkedBinary(String bundleDirSuffix) {
    String baseName = ownerLabel.toPathFragment().getBaseName();
    return appendExtension(bundleDirSuffix + "/" + baseName);
  }

  /**
   * The {@code .a} file which contains all the compiled sources for a rule.
   */
  public Artifact archive() {
    PathFragment labelPath = ownerLabel.toPathFragment();
    PathFragment rootRelative =
        labelPath
            .getParentDirectory()
            .getRelative(String.format("lib%s.a", labelPath.getBaseName()));
    return analysisEnvironment.getDerivedArtifact(rootRelative, binDirectory);
  }

  /**
   * The artifact for the .o file that should be generated when compiling the {@code source}
   * artifact.
   */
  public Artifact objFile(Artifact source) {
    return analysisEnvironment.getDerivedArtifact(
        FileSystemUtils.replaceExtension(
            AnalysisUtils.getUniqueDirectory(ownerLabel, new PathFragment("_objs"))
                .getRelative(source.getRootRelativePath()),
            ".o"),
        binDirectory);
  }

  /**
   * Returns the artifact corresponding to the pbxproj control file, which specifies the information
   * required to generate the Xcode project file.
   */
  public Artifact pbxprojControlArtifact() {
    return appendExtension(".xcodeproj-control");
  }

  /**
   * The artifact which contains the zipped-up results of compiling the storyboard. This is merged
   * into the final bundle under the {@code .app} or {@code .bundle} directory root.
   */
  public Artifact compiledStoryboardZip(Artifact input) {
    return appendExtension("/" + BundleableFile.bundlePath(input) + ".zip");
  }

  /**
   * Returns the artifact which is the output of building an entire xcdatamodel[d] made of artifacts
   * specified by a single rule.
   *
   * @param containerDir the containing *.xcdatamodeld or *.xcdatamodel directory
   * @return the artifact for the zipped up compilation results.
   */
  public Artifact compiledMomZipArtifact(PathFragment containerDir) {
    return appendExtension(
        "/" + FileSystemUtils.replaceExtension(containerDir, ".zip").getBaseName());
  }

  /**
   * Returns the compiled (i.e. converted to binary plist format) artifact corresponding to the
   * given {@code .strings} file.
   */
  public Artifact convertedStringsFile(Artifact originalFile) {
    return appendExtension(originalFile.getExecPath(), ".binary");
  }

  /**
   * Returns the artifact corresponding to the compiled form of the given {@code .xib} file.
   */
  public Artifact compiledXibFile(Artifact originalFile) {
    return analysisEnvironment.getDerivedArtifact(
        FileSystemUtils.replaceExtension(originalFile.getExecPath(), ".nib"),
        binDirectory);
  }
}
