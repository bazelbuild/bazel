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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.Structure;

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
public interface ConfiguredTarget extends TransitiveInfoCollection, Structure {

  /** All <code>ConfiguredTarget</code>s have a "label" field. */
  String LABEL_FIELD = "label";

  /** All <code>ConfiguredTarget</code>s have a "kind" field. */
  String KIND_FIELD = "kind";

  /** All <code>ConfiguredTarget</code>s have a "files" field. */
  String FILES_FIELD = "files";

  @Nullable
  default String getConfigurationChecksum() {
    return getConfigurationKey() == null ? null : getConfigurationKey().getOptions().checksum();
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

  /** Returns keys for a legacy Starlark provider. */
  @Override
  ImmutableCollection<String> getFieldNames();

  /**
   * Returns a legacy Starlark provider.
   *
   * <p>Overrides {@link Structure#getValue(String)}, but does not allow EvalException to be thrown.
   */
  @Nullable
  @Override
  Object getValue(String name);

  /**
   * If the configured target is an alias, return the actual target, otherwise return the current
   * target. This follows alias chains.
   */
  default ConfiguredTarget getActual() {
    return this;
  }

  /**
   * If the configured target is an alias, return the original label, otherwise return the current
   * label. This is not the same as {@code getActual().getLabel()}, because it does not follow alias
   * chains.
   */
  default Label getOriginalLabel() {
    return getLabel();
  }

  /**
   * The configuration conditions that trigger this configured target's configurable attributes. For
   * targets that do not support configurable attributes, this will be an empty map.
   */
  default ImmutableMap<Label, ConfigMatchingProvider> getConfigConditions() {
    return ImmutableMap.of();
  }
}
