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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.LicensesProviderImpl;
import com.google.devtools.build.lib.analysis.TargetContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProviderImpl;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.Instantiator;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.util.Pair;

/** A ConfiguredTarget for an OutputFile. */
@AutoCodec
public class OutputFileConfiguredTarget extends FileConfiguredTarget
    implements InstrumentedFilesProvider {

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
    super(label, configurationKey, visibility, artifact);
    this.artifact = artifact;
    this.generatingRule = Preconditions.checkNotNull(generatingRule);
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


  @Override
  public NestedSet<Artifact> getInstrumentedFiles() {
    return getProvider(InstrumentedFilesProvider.class, InstrumentedFilesProviderImpl.EMPTY)
        .getInstrumentedFiles();
  }

  @Override
  public NestedSet<Artifact> getInstrumentationMetadataFiles() {
    return getProvider(InstrumentedFilesProvider.class, InstrumentedFilesProviderImpl.EMPTY)
        .getInstrumentationMetadataFiles();
  }

  @Override
  public NestedSet<Artifact> getBaselineCoverageInstrumentedFiles() {
    return getProvider(InstrumentedFilesProvider.class, InstrumentedFilesProviderImpl.EMPTY)
        .getBaselineCoverageInstrumentedFiles();
  }

  @Override
  public NestedSet<Artifact> getBaselineCoverageArtifacts() {
    return getProvider(InstrumentedFilesProvider.class, InstrumentedFilesProviderImpl.EMPTY)
        .getBaselineCoverageArtifacts();
  }

  @Override
  public NestedSet<Artifact> getCoverageSupportFiles() {
    return getProvider(InstrumentedFilesProvider.class, InstrumentedFilesProviderImpl.EMPTY)
        .getCoverageSupportFiles();
  }

  @Override
  public NestedSet<Pair<String, String>> getCoverageEnvironment() {
    return getProvider(InstrumentedFilesProvider.class, InstrumentedFilesProviderImpl.EMPTY)
        .getCoverageEnvironment();
  }

  @Override
  public NestedSet<Pair<String, String>> getReportedToActualSources() {
    return getProvider(InstrumentedFilesProvider.class, InstrumentedFilesProviderImpl.EMPTY)
        .getReportedToActualSources();
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
  public void repr(SkylarkPrinter printer) {
    printer.append("<output file target " + getLabel() + ">");
  }
}
