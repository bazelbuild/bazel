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

import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import javax.annotation.Nullable;

/**
 * A {@link ConfiguredTarget} is conceptually a {@link TransitiveInfoCollection} coupled with the
 * {@link com.google.devtools.build.lib.packages.Target} and {@link
 * com.google.devtools.build.lib.analysis.config.BuildConfiguration} objects it was created from.
 *
 * <p>This interface is supposed to only be used in {@link BuildView} and above. In particular, rule
 * implementations should not be able to access the {@link ConfiguredTarget} objects associated with
 * their direct dependencies, only the corresponding {@link TransitiveInfoCollection}s. Also, {@link
 * ConfiguredTarget} objects should not be accessible from the action graph.
 */
public interface ConfiguredTarget extends TransitiveInfoCollection, ClassObject, StarlarkValue {

  /**
   *  All <code>ConfiguredTarget</code>s have a "label" field.
   */
  String LABEL_FIELD = "label";

  /**
   *  All <code>ConfiguredTarget</code>s have a "files" field.
   */
  String FILES_FIELD = "files";

  default String getConfigurationChecksum() {
    return getConfigurationKey() == null
        ? null
        : getConfigurationKey().getOptionsDiff().getChecksum();
  }

  /**
   * Returns the {@link BuildConfigurationValue.Key} naming the {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfiguration} for which this configured
   * target is defined. Configuration is defined for all configured targets with exception of {@link
   * com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget} and {@link
   * com.google.devtools.build.lib.analysis.configuredtargets.PackageGroupConfiguredTarget} for
   * which it is always <b>null</b>.
   */
  @Nullable
  BuildConfigurationValue.Key getConfigurationKey();

  /** Returns keys for a legacy Skylark provider. */
  @Override
  ImmutableCollection<String> getFieldNames();

  /**
   * Returns a legacy Skylark provider.
   *
   * Overrides {@link ClassObject#getValue(String)}, but does not allow EvalException to
   * be thrown.
   */
  @Nullable
  @Override
  Object getValue(String name);

  /** Returns a source artifact if this is an input file. */
  @Nullable
  default SourceArtifact getSourceArtifact() {
    return null;
  }
}
