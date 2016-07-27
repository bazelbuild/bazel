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
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProviderImpl;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * A ConfiguredTarget for an OutputFile.
 */
public class OutputFileConfiguredTarget extends FileConfiguredTarget
    implements InstrumentedFilesProvider {

  private final TransitiveInfoCollection generatingRule;

  OutputFileConfiguredTarget(
      TargetContext targetContext, OutputFile outputFile,
      TransitiveInfoCollection generatingRule, Artifact outputArtifact) {
    super(targetContext, outputArtifact);
    Preconditions.checkArgument(targetContext.getTarget() == outputFile);
    this.generatingRule = generatingRule;
  }

  @Override
  public OutputFile getTarget() {
    return (OutputFile) super.getTarget();
  }

  public TransitiveInfoCollection getGeneratingRule() {
    return generatingRule;
  }

  @Override
  public NestedSet<TargetLicense> getTransitiveLicenses() {
    return getProvider(LicensesProvider.class, LicensesProviderImpl.EMPTY)
        .getTransitiveLicenses();
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
}
