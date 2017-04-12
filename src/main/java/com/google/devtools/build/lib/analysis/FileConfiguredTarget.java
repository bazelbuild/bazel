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
import com.google.devtools.build.lib.packages.FileTarget;
import com.google.devtools.build.lib.rules.fileset.FilesetProvider;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.util.FileType;

/**
 * A ConfiguredTarget for a source FileTarget.  (Generated files use a
 * subclass, OutputFileConfiguredTarget.)
 */
public abstract class FileConfiguredTarget extends AbstractConfiguredTarget
    implements FileType.HasFilename, LicensesProvider {

  private final Artifact artifact;
  private final TransitiveInfoProviderMap providers;

  FileConfiguredTarget(TargetContext targetContext, Artifact artifact) {
    super(targetContext);
    NestedSet<Artifact> filesToBuild = NestedSetBuilder.create(Order.STABLE_ORDER, artifact);
    this.artifact = artifact;
    FileProvider fileProvider = new FileProvider(filesToBuild);
    FilesToRunProvider filesToRunProvider =
        FilesToRunProvider.fromSingleExecutableArtifact(artifact);
    TransitiveInfoProviderMap.Builder builder =
        TransitiveInfoProviderMap.builder()
            .put(VisibilityProvider.class, this)
            .put(LicensesProvider.class, this)
            .add(fileProvider)
            .add(filesToRunProvider);
    if (this instanceof FilesetProvider) {
      builder.put(FilesetProvider.class, this);
    }
    if (this instanceof InstrumentedFilesProvider) {
      builder.put(InstrumentedFilesProvider.class, this);
    }
    this.providers = builder.build();
  }

  @Override
  public FileTarget getTarget() {
    return (FileTarget) super.getTarget();
  }

  public Artifact getArtifact() {
    return artifact;
  }

  /**
   *  Returns the file type of this file target.
   */
  @Override
  public String getFilename() {
    return getTarget().getFilename();
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
    AnalysisUtils.checkProvider(provider);
    return providers.getProvider(provider);
  }

  @Override
  public Object get(String providerKey) {
    return null;
  }
}
