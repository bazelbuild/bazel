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
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.LicensesProviderImpl;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.TargetContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.Instantiator;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.syntax.Printer;

/** A ConfiguredTarget for an OutputFile. */
@AutoCodec
@Immutable // (and Starlark-hashable)
public class OutputFileConfiguredTarget extends FileConfiguredTarget {

  private final Artifact artifact;
  private final TransitiveInfoCollection generatingRule;

  public OutputFileConfiguredTarget(
      TargetContext targetContext, OutputFile outputFile,
      TransitiveInfoCollection generatingRule, Artifact outputArtifact) {
    this(
        targetContext.getLabel(),
        targetContext.getConfigurationKey(),
        targetContext.getVisibility(),
        outputArtifact,
        generatingRule);
    Preconditions.checkArgument(targetContext.getTarget() == outputFile);
  }

  @Instantiator
  @VisibleForSerialization
  OutputFileConfiguredTarget(
      Label label,
      BuildConfigurationValue.Key configurationKey,
      NestedSet<PackageGroupContents> visibility,
      Artifact artifact,
      TransitiveInfoCollection generatingRule) {

    super(
        label,
        configurationKey,
        visibility,
        artifact,
        instrumentedFilesInfo(generatingRule),
        generatingRule.getProvider(RequiredConfigFragmentsProvider.class),
        Preconditions.checkNotNull(generatingRule).get(OutputGroupInfo.SKYLARK_CONSTRUCTOR));

    this.artifact = artifact;
    this.generatingRule = Preconditions.checkNotNull(generatingRule);
  }

  private static InstrumentedFilesInfo instrumentedFilesInfo(
      TransitiveInfoCollection generatingRule) {
    Preconditions.checkNotNull(generatingRule);
    InstrumentedFilesInfo provider = generatingRule.get(InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR);
    return provider == null ? InstrumentedFilesInfo.EMPTY : provider;
  }

  public TransitiveInfoCollection getGeneratingRule() {
    return generatingRule;
  }

  @Override
  public final Artifact getArtifact() {
    return artifact;
  }

  @Override
  public NestedSet<TargetLicense> getTransitiveLicenses() {
    return getProvider(LicensesProvider.class, LicensesProviderImpl.EMPTY)
        .getTransitiveLicenses();
  }

  @Override
  public TargetLicense getOutputLicenses() {
    return getProvider(LicensesProvider.class, LicensesProviderImpl.EMPTY)
        .getOutputLicenses();
  }

  @Override
  public boolean hasOutputLicenses() {
    return getProvider(LicensesProvider.class, LicensesProviderImpl.EMPTY)
        .hasOutputLicenses();
  }

  /**
   * Returns the corresponding provider from the generating rule, if it is non-null, or {@code
   * defaultValue} otherwise.
   */
  private <T extends TransitiveInfoProvider> T getProvider(Class<T> clazz, T defaultValue) {
    if (generatingRule != null) {
      T result = generatingRule.getProvider(clazz);
      if (result != null) {
        return result;
      }
    }
    return defaultValue;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<output file target " + getLabel() + ">");
  }
}
