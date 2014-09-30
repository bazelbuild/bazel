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

  private Artifact appendExtension(String extension) {
    return analysisEnvironment.getDerivedArtifact(
        FileSystemUtils.appendExtension(ownerLabel.toPathFragment(), extension),
        binDirectory);
  }

  public Artifact actoolzipOutput() {
    return appendExtension(".actool.zip");
  }

  public Artifact mergedInfoplist() {
    return appendExtension("-MergedInfo.plist");
  }

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
}
