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

package com.google.devtools.build.lib.analysis.configuredtargets;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.analysis.TargetContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.Instantiator;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import java.util.Objects;

/**
 * A ConfiguredTarget for an InputFile.
 *
 * <p>All InputFiles for the same target are equivalent, so configuration does not play any role
 * here and is always set to <b>null</b>.
 */
@AutoCodec
@Immutable // (and Starlark-hashable)
public final class InputFileConfiguredTarget extends FileConfiguredTarget implements StarlarkValue {
  private final SourceArtifact artifact;
  private final NestedSet<TargetLicense> licenses;

  @Instantiator
  @VisibleForSerialization
  InputFileConfiguredTarget(
      Label label,
      NestedSet<PackageGroupContents> visibility,
      SourceArtifact artifact,
      NestedSet<TargetLicense> licenses) {
    super(label, null, visibility, artifact, null, null, null);
    this.artifact = artifact;
    this.licenses = licenses;
  }

  public InputFileConfiguredTarget(
      TargetContext targetContext, InputFile inputFile, SourceArtifact artifact) {
    this(inputFile.getLabel(), targetContext.getVisibility(), artifact, makeLicenses(inputFile));
    Preconditions.checkArgument(getConfigurationKey() == null, getLabel());
    Preconditions.checkArgument(targetContext.getTarget() == inputFile, getLabel());
  }

  private static NestedSet<TargetLicense> makeLicenses(InputFile inputFile) {
    License license = inputFile.getLicense();
    return Objects.equals(license, License.NO_LICENSE)
        ? NestedSetBuilder.emptySet(Order.LINK_ORDER)
        : NestedSetBuilder.create(
            Order.LINK_ORDER, new TargetLicense(inputFile.getLabel(), license));
  }

  @Override
  public final Artifact getArtifact() {
    return artifact;
  }

  @Override
  public String toString() {
    return "InputFileConfiguredTarget(" + getLabel() + ")";
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

  @Override
  public void repr(Printer printer) {
    printer.append("<input file target " + getLabel() + ">");
  }

  @Override
  public SourceArtifact getSourceArtifact() {
    return artifact;
  }
}
