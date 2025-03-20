// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParser;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Maps build option names as they appear to the user (e.g. {@code compilation_mode}) to structured
 * metadata.
 *
 * <p>For native options ({@code @Option} defined in a {@link FragmentOptions} implementation), this
 * tracks:
 *
 * <ul>
 *   <li>what {@link FragmentOptions} class defines the option
 *   <li>the option's current value
 *   <li>whether it allows multiple values to be specified ({@link Option#allowMultiple}
 *   <li>whether it is selectable, i.e., allowed to appear in a {@code config_setting}
 * </ul>
 *
 * <p>For Starlark options (defined in a Starlark {@code build_setting}), this tracks their value in
 * built-in Starlark-object form (post-parse, pre-implementation function form).
 */
public final class BuildOptionDetails {

  /** Builds a {@code BuildOptionDetails} for the given set of native options */
  @VisibleForTesting
  static BuildOptionDetails forOptionsForTesting(
      Iterable<? extends FragmentOptions> buildOptions) {
    return forOptions(buildOptions, ImmutableMap.of());
  }

  /** Builds a {@code BuildOptionDetails} for the given set of native and Starlark options. */
  static BuildOptionDetails forOptions(
      Iterable<? extends FragmentOptions> buildOptions, Map<Label, Object> starlarkOptions) {
    ImmutableMap.Builder<String, OptionDetails> map = ImmutableMap.builder();
    for (FragmentOptions options : buildOptions) {
      ImmutableList<? extends OptionDefinition> optionDefinitions =
          OptionsParser.getOptionDefinitions(options.getClass());

      for (OptionDefinition optionDefinition : optionDefinitions) {
        if (ImmutableList.copyOf(optionDefinition.getOptionMetadataTags())
            .contains(OptionMetadataTag.INTERNAL)) {
          // ignore internal options
          continue;
        }
        Object value = optionDefinition.getValue(options);
        map.put(
            optionDefinition.getOptionName(),
            new OptionDetails(options.getClass(), value, optionDefinition.allowsMultiple()));
      }
    }
    return new BuildOptionDetails(map.buildOrThrow(), ImmutableMap.copyOf(starlarkOptions));
  }

  private static final class OptionDetails {

    private OptionDetails(
        Class<? extends FragmentOptions> optionsClass, Object value, boolean allowsMultiple) {
      this.optionsClass = optionsClass;
      this.value = value;
      this.allowsMultiple = allowsMultiple;
    }

    /** The {@link FragmentOptions} class that defines this option. */
    private final Class<? extends FragmentOptions> optionsClass;

    /** The value of the given option (either explicitly defined or default). May be null. */
    @Nullable private final Object value;

    /** Whether or not this option supports multiple values. */
    private final boolean allowsMultiple;
  }

  /**
   * Maps native option names to the {@link OptionDetails} the option takes for this configuration.
   *
   * <p>This can be used to:
   *
   * <ol>
   *   <li>Find an option's (parsed) value given its command-line name
   *   <li>Parse alternative values for the option.
   * </ol>
   */
  private final ImmutableMap<String, OptionDetails> nativeOptionsMap;

  /** Maps Starlark option labels to values */
  private final ImmutableMap<Label, Object> starlarkOptionsMap;

  private BuildOptionDetails(
      ImmutableMap<String, OptionDetails> nativeOptionsMap,
      ImmutableMap<Label, Object> starlarkOptionsMap) {
    this.nativeOptionsMap = nativeOptionsMap;
    this.starlarkOptionsMap = starlarkOptionsMap;
  }

  /**
   * Returns the {@link FragmentOptions} class the defines the given option, null if the option
   * isn't recognized.
   *
   * <p>optionName is the name of the option as it appears on the command line e.g. {@link
   * OptionDefinition#getOptionName()}).
   */
  @Nullable
  public Class<? extends FragmentOptions> getOptionClass(String optionName) {
    OptionDetails optionDetails = nativeOptionsMap.get(optionName);
    return optionDetails == null ? null : optionDetails.optionsClass;
  }

  /**
   * Returns the value of the specified native option for this configuration or null if the option
   * isn't recognized. Since an option's legitimate value could be null, use {@link #getOptionClass}
   * to distinguish between that and an unknown option.
   *
   * <p>optionName is the name of the option as it appears on the command line e.g. {@link
   * OptionDefinition#getOptionName()}).
   */
  @Nullable
  public Object getOptionValue(String optionName) {
    OptionDetails optionDetails = nativeOptionsMap.get(optionName);
    return (optionDetails == null) ? null : optionDetails.value;
  }

  /** Returns the value of the specified Starlark option or null if it isn't recognized */
  @Nullable
  public Object getOptionValue(Label optionName) {
    return starlarkOptionsMap.get(optionName);
  }

  /**
   * Returns whether or not the given option supports multiple values at the command line (e.g.
   * "--myoption value1 --myOption value2 ..."). Returns false for unrecognized options. Use {@link
   * #getOptionClass} to distinguish between those and legitimate single-value options.
   *
   * <p>As declared in {@link OptionDefinition#allowsMultiple()}, multi-value options are expected
   * to be of type {@code List<T>}.
   */
  public boolean allowsMultipleValues(String optionName) {
    OptionDetails optionDetails = nativeOptionsMap.get(optionName);
    return optionDetails != null && optionDetails.allowsMultiple;
  }
}
