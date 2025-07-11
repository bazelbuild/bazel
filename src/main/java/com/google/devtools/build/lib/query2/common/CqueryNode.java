// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.query2.common;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;

/**
 * A {@link CqueryNode} provides information necessary to traverse different types of nodes that can
 * be visited during a {@link CqueryCommand} call. This may include {@link ConfiguredTarget}, {@link
 * AspectKey}, and transition nodes.
 */
public interface CqueryNode {
  /** Returns a key that may be used to lookup this {@link CqueryNode}. */
  ActionLookupKey getLookupKey();

  default Label getLabel() {
    return getLookupKey().getLabel();
  }

  default String getDescription(LabelPrinter labelPrinter) {
    return labelPrinter.toString(getOriginalLabel());
  }

  @Nullable
  default String getConfigurationChecksum() {
    return getConfigurationKey() == null ? null : getConfigurationKey().getOptions().checksum();
  }

  /**
   * Returns the {@link BuildConfigurationKey} naming the {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfigurationValue} for which this cquery
   * node is defined. Configuration is defined for all configured targets with exception of {@link
   * com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget} and {@link
   * com.google.devtools.build.lib.analysis.configuredtargets.PackageGroupConfiguredTarget} for
   * which it is always <b>null</b>.
   *
   * <p>If this changes, {@link AspectResolver#aspectMatchesConfiguredTarget} should be updated.
   */
  @Nullable
  default BuildConfigurationKey getConfigurationKey() {
    return getLookupKey().getConfigurationKey();
  }

  /**
   * If the configured target is an alias, return the actual target, otherwise return the current
   * target. This follows alias chains.
   */
  default CqueryNode getActual() {
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

  default boolean isRuleConfiguredTarget() {
    return false;
  }

  /**
   * The base configured target if it has been merged with aspects otherwise the current value.
   *
   * <p>Unwrapping is recursive if there are multiple layers.
   */
  default CqueryNode unwrapIfMerged() {
    return this;
  }

  /**
   * This is only intended to be called from the query dialects of Starlark.
   *
   * @return a map of provider names to their values
   */
  default Dict<String, Object> getProvidersDictForQuery() {
    return Dict.empty();
  }
}
