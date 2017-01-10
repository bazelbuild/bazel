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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * A ConfiguredTarget for an InputFile.
 *
 * All InputFiles for the same target are equivalent, so configuration does not
 * play any role here and is always set to <b>null</b>.
 */
public final class InputFileConfiguredTarget extends FileConfiguredTarget {
  private final Artifact artifact;
  private final NestedSet<TargetLicense> licenses;

  InputFileConfiguredTarget(TargetContext targetContext, InputFile inputFile, Artifact artifact) {
    super(targetContext, artifact);
    Preconditions.checkArgument(targetContext.getTarget() == inputFile, getLabel());
    Preconditions.checkArgument(getConfiguration() == null, getLabel());
    this.artifact = artifact;

    if (inputFile.getLicense() != License.NO_LICENSE) {
      licenses = NestedSetBuilder.create(Order.LINK_ORDER,
          new TargetLicense(getLabel(), inputFile.getLicense()));
    } else {
      licenses = NestedSetBuilder.emptySet(Order.LINK_ORDER);
    }
  }

  @Override
  public InputFile getTarget() {
    return (InputFile) super.getTarget();
  }

  @Override
  public Artifact getArtifact() {
    return artifact;
  }

  @Override
  public String toString() {
    return "InputFileConfiguredTarget(" + getTarget().getLabel() + ")";
  }

  @Override
  public final NestedSet<TargetLicense> getTransitiveLicenses() {
    return licenses;
  }

  @Override
  public TargetLicense getOutputLicenses() {
    return null;
  }

  @Override
  public boolean hasOutputLicenses() {
    return false;
  }
}
