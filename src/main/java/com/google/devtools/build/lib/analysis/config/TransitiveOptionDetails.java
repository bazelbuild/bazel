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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.FragmentOptions.SelectRestriction;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParser;
import java.io.Serializable;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Introspector for option details, to be used when {@code select()}-ing over the options.
 *
 * <p>This tracks:
 *
 * <ul>
 *   <li>what {@link FragmentOptions} class the option is defined in
 *   <li>the option's current value
 *   <li>whether it allows multiple values to be specified ({@link Option#allowMultiple}
 *   <li>whether it is selectable, i.e., allowed to appear in a {@code config_setting}
 * </ul>
 *
 * <p>This is "transitive" in that it includes *all* options recognizable by a given configuration,
 * including those defined in child fragments.
 */
public final class TransitiveOptionDetails implements Serializable {

  /** Builds a {@code TransitiveOptionDetails} for the given set of options. */
  static TransitiveOptionDetails forOptions(Iterable<? extends FragmentOptions> buildOptions) {
    ImmutableMap.Builder<String, OptionDetails> map = ImmutableMap.builder();
    try {
      for (FragmentOptions options : buildOptions) {
        ImmutableList<OptionDefinition> optionDefinitions =
            OptionsParser.getOptionDefinitions(options.getClass());
        Map<OptionDefinition, SelectRestriction> selectRestrictions =
            options.getSelectRestrictions();

        for (OptionDefinition optionDefinition : optionDefinitions) {
          if (ImmutableList.copyOf(optionDefinition.getOptionMetadataTags())
              .contains(OptionMetadataTag.INTERNAL)) {
            // ignore internal options
            continue;
          }
          Object value = optionDefinition.getField().get(options);
          if (value == null && !optionDefinition.isSpecialNullDefault()) {
              // See {@link Option#defaultValue} for an explanation of default "null" strings.
              value = optionDefinition.getUnparsedDefaultValue();
          }
          map.put(
              optionDefinition.getOptionName(),
              new OptionDetails(
                  options.getClass(),
                  value,
                  optionDefinition.allowsMultiple(),
                  selectRestrictions.get(optionDefinition)));
        }
      }
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(
          "Unexpected illegal access trying to create this configuration's options map: ", e);
    }
    return new TransitiveOptionDetails(map.build());
  }

  private static final class OptionDetails implements Serializable {

    private OptionDetails(
        Class<? extends FragmentOptions> optionsClass,
        Object value,
        boolean allowsMultiple,
        @Nullable SelectRestriction selectRestriction) {
      this.optionsClass = optionsClass;
      this.value = value;
      this.allowsMultiple = allowsMultiple;
      this.selectRestriction = selectRestriction;
    }

    /** The {@link FragmentOptions} class that defines this option. */
    private final Class<? extends FragmentOptions> optionsClass;

    /** The value of the given option (either explicitly defined or default). May be null. */
    @Nullable private final Object value;

    /** Whether or not this option supports multiple values. */
    private final boolean allowsMultiple;

    /**
     * Information on whether this option is permitted to appear in {@code config_setting}s. Null if
     * there is no such restriction.
     */
    @Nullable private final SelectRestriction selectRestriction;
  }

  /**
   * Maps option names to the {@link OptionDetails} the option takes for this configuration.
   *
   * <p>This can be used to:
   *
   * <ol>
   *   <li>Find an option's (parsed) value given its command-line name
   *   <li>Parse alternative values for the option.
   * </ol>
   */
  private final ImmutableMap<String, OptionDetails> transitiveOptionsMap;

  private TransitiveOptionDetails(ImmutableMap<String, OptionDetails> transitiveOptionsMap) {
    this.transitiveOptionsMap = transitiveOptionsMap;
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
    OptionDetails optionDetails = transitiveOptionsMap.get(optionName);
    return optionDetails == null ? null : optionDetails.optionsClass;
  }

  /**
   * Returns the value of the specified option for this configuration or null if the option isn't
   * recognized. Since an option's legitimate value could be null, use {@link #getOptionClass} to
   * distinguish between that and an unknown option.
   *
   * <p>optionName is the name of the option as it appears on the command line e.g. {@link
   * OptionDefinition#getOptionName()}).
   */
  @Nullable
  public Object getOptionValue(String optionName) {
    OptionDetails optionDetails = transitiveOptionsMap.get(optionName);
    return (optionDetails == null) ? null : optionDetails.value;
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
    OptionDetails optionDetails = transitiveOptionsMap.get(optionName);
    return optionDetails != null && optionDetails.allowsMultiple;
  }

  /**
   * Returns information about whether an option may appear in a {@code config_setting}.
   *
   * <p>Returns null for unrecognized options or options that have no restriction.
   */
  @Nullable
  public SelectRestriction getSelectRestriction(String optionName) {
    OptionDetails optionDetails = transitiveOptionsMap.get(optionName);
    return optionDetails == null ? null : optionDetails.selectRestriction;
  }
}
