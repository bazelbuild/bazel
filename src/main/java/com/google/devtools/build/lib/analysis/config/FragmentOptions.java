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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import java.io.Serializable;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/** Command-line build options for a Blaze module. */
public abstract class FragmentOptions extends OptionsBase implements Cloneable, Serializable {
  /**
   * Returns the labels contributed to the defaults package by this fragment.
   *
   * <p>The set of keys returned by this function should be constant, however, the values are
   * allowed to change depending on the value of the options.
   */
  @SuppressWarnings("unused")
  public Map<String, Set<Label>> getDefaultsLabels() {
    return ImmutableMap.of();
  }

  @Override
  public FragmentOptions clone() {
    try {
      return (FragmentOptions) super.clone();
    } catch (CloneNotSupportedException e) {
      // This can't happen.
      throw new IllegalStateException(e);
    }
  }

  /**
   * Creates a new FragmentOptions instance with all flags set to default.
   */
  public FragmentOptions getDefault() {
    return Options.getDefaults(getClass());
  }

  /**
   * Creates a new FragmentOptions instance with flags adjusted to host platform.
   */
  @SuppressWarnings("unused")
  public FragmentOptions getHost() {
    return getDefault();
  }

  /** Tracks limitations on referring to an option in a {@code config_setting}. */
  // TODO(bazel-team): There will likely also be a need to customize whether or not an option is
  // visible to users for setting on the command line (or perhaps even in a test of a Starlark
  // rule). This class may be a good place to add this functionality.
  public static final class SelectRestriction {

    private final boolean visibleWithinToolsPackage;

    @Nullable private final String errorMessage;

    public SelectRestriction(boolean visibleWithinToolsPackage, @Nullable String errorMessage) {
      this.visibleWithinToolsPackage = visibleWithinToolsPackage;
      this.errorMessage = errorMessage;
    }

    /**
     * Whether the option can still be seen by {@code config_setting}s that are defined by packages
     * underneath the tools repository's "tools" package, e.g. {@code @bazel_tools//tools/...}.
     */
    public boolean isVisibleWithinToolsPackage() {
      return visibleWithinToolsPackage;
    }

    /**
     * An additional explanation to append to the generic error message when a user attempts to use
     * this option. Should explain why this option is unavailable.
     *
     * <p>If null, no content will be appended to the generic error message.
     */
    @Nullable
    public String getErrorMessage() {
      return errorMessage;
    }
  }

  /**
   * Returns a map from options defined by this fragment to restrictions on whether the option may
   * appear in a {@code config_setting}. If an option defined by this fragment is not a key of this
   * map, then it has no restriction.
   *
   * <p>In addition to making options unconditionally non-selectable, this can also be used to gate
   * selectability based on the value of other flags in the same fragment -- for instance,
   * experimental or incompatible change flags.
   *
   * <p>The intended usage pattern is to define, for each flag {@code foo} to have a restriction, a
   * field
   *
   * <pre>{@code
   * private static final OptionDefinition FOO_DEFINITION =
   *     OptionsParser.getOptionDefinitionByName(ThisClass.class, "foo");
   * }</pre>
   *
   * This way, if the option is ever renamed (especially common for an experimental flag), if the
   * definition is not updated at the same time it will fail-fast during static initialization.
   */
  public Map<OptionDefinition, SelectRestriction> getSelectRestrictions() {
    return ImmutableMap.of();
  }
}
