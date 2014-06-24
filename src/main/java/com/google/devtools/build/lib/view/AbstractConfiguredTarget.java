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

package com.google.devtools.build.lib.view;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import java.util.List;

import javax.annotation.Nullable;

/**
 * An abstract implementation of ConfiguredTarget in which all properties are
 * assigned trivial default values.
 */
@SkylarkBuiltin(name = "", doc = "")
public abstract class AbstractConfiguredTarget
    implements ConfiguredTarget, FileProvider, FilesToRunProvider, VisibilityProvider {
  private final Target target;
  private final BuildConfiguration configuration;

  private final NestedSet<PackageSpecification> visibility;

  /**
   * This target's filesToBuild. Intentionally not final: this field is initialized to the empty
   * set, and individual implementations are expected to set this to something else. Only in the
   * rare case when the configured target does not produce any files or when it emits an error
   * should this be left alone.
   */
  protected NestedSet<Artifact> filesToBuild = NestedSetBuilder.emptySet(Order.STABLE_ORDER);

  AbstractConfiguredTarget(Target target,
                           BuildConfiguration configuration) {
    this.target = target;
    this.configuration = configuration;
    this.visibility = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  AbstractConfiguredTarget(TargetContext targetContext) {
    this.target = targetContext.getTarget();
    this.configuration = targetContext.getConfiguration();
    this.visibility = targetContext.getVisibility();
  }

  @Override
  public NestedSet<PackageSpecification> getVisibility() {
    return visibility;
  }

  @Override
  public Target getTarget() {
    return target;
  }

  @Override
  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  @Override
  public Label getLabel() {
    return getTarget().getLabel();
  }

  @Override
  public String toString() {
    return "ConfiguredTarget(" + getTarget().getLabel() + ", " + getConfiguration() + ")";
  }

  @Override
  public NestedSet<Artifact> getFilesToBuild() {
    return filesToBuild;
  }

  /**
   * Like getFilesToBuild(), except that it also includes the runfiles middleman, if any.
   * Middlemen are expanded in the SpawnStrategy or by the Distributor.
   */
  @Override
  public ImmutableList<Artifact> getFilesToRun() {
    return RuleContext.getFilesToRun(getRunfilesSupport(), getFilesToBuild());
  }

  /**
   * Convenience method to filter the files to build for a certain filetype.
   *
   * @param allowedType the allowed filetype
   * @return all members of filesToBuild that are of one of the
   *     allowed filetypes
   */
  public List<Artifact> getFilesToBuild(FileType allowedType) {
    return Artifact.filterFiles(getFilesToBuild(), allowedType);
  }

  @Override
  public Artifact getExecutable() { return null; }

  @Override
  @Nullable
  public Artifact getRunfilesManifest() {
    RunfilesSupport runfilesSupport = getRunfilesSupport();
    if (runfilesSupport != null) {
      return runfilesSupport.getRunfilesManifest();
    } else {
      return null;
    }
  }

  @Override
  public RunfilesSupport getRunfilesSupport() { return null; }

  @Override
  public Iterable<Class<? extends TransitiveInfoProvider>> getImplementedProviders() {
    return TransitiveInfoProviderCache.getProviderClasses(this.getClass());
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
    AnalysisUtils.checkProvider(provider);
    if (provider.isAssignableFrom(getClass())) {
      return provider.cast(this);
    } else {
      return null;
    }
  }

  // TODO(bazel-team): This function is not nice here. Figure out a way to get rid of this.
  @SkylarkCallable(
      doc = "Returns the value provided by this target associated with the provider_key.")
  public Object get(String providerKey) {
    return null;
  }
}
