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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
import java.io.Serializable;
import java.lang.reflect.Field;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Introspector for option details - what OptionsBase class the option is defined in, the option's
 * current value, and whether the option allows multiple values to be specified.
 *
 * <p>This is "transitive" in that it includes *all* options recognizable by a given configuration,
 * including those defined in child fragments.
 */
public final class TransitiveOptionDetails implements Serializable {

  /**
   * Computes and returns the transitive optionName -> "option info" map for the given set of
   * options sets, using the given map as defaults for options which would otherwise be null.
   */
  public static TransitiveOptionDetails forOptionsWithDefaults(
      Iterable<? extends OptionsBase> buildOptions, Map<String, Object> lateBoundDefaults) {
    ImmutableMap.Builder<String, OptionDetails> map = ImmutableMap.builder();
    try {
      for (OptionsBase options : buildOptions) {
        for (Field field : options.getClass().getFields()) {
          if (field.isAnnotationPresent(Option.class)) {
            Option option = field.getAnnotation(Option.class);
            if (OptionsParser.documentationLevel(option).equals(OptionUsageRestrictions.INTERNAL)) {
              // ignore internal options
              continue;
            }
            Object value = field.get(options);
            if (value == null) {
              if (lateBoundDefaults.containsKey(option.name())) {
                value = lateBoundDefaults.get(option.name());
              } else if (!option.defaultValue().equals("null")) {
                // See {@link Option#defaultValue} for an explanation of default "null" strings.
                value = option.defaultValue();
              }
            }
            map.put(option.name(),
                new OptionDetails(options.getClass(), value, option.allowMultiple()));
          }
        }
      }
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(
          "Unexpected illegal access trying to create this configuration's options map: ", e);
    }
    return new TransitiveOptionDetails(map.build());
  }

  private static final class OptionDetails implements Serializable {
    private OptionDetails(Class<? extends OptionsBase> optionsClass, Object value,
        boolean allowsMultiple) {
      this.optionsClass = optionsClass;
      this.value = value;
      this.allowsMultiple = allowsMultiple;
    }

    /** The {@link FragmentOptions} class that defines this option. */
    private final Class<? extends OptionsBase> optionsClass;

    /** The value of the given option (either explicitly defined or default). May be null. */
    @Nullable private final Object value;

    /** Whether or not this option supports multiple values. */
    private final boolean allowsMultiple;
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
   * Returns the {@link Option} class the defines the given option, null if the option isn't
   * recognized.
   *
   * <p>optionName is the name of the option as it appears on the command line e.g. {@link
   * Option#name}).
   */
  public Class<? extends OptionsBase> getOptionClass(String optionName) {
    OptionDetails optionData = transitiveOptionsMap.get(optionName);
    return optionData == null ? null : optionData.optionsClass;
  }

  /**
   * Returns the value of the specified option for this configuration or null if the option isn't
   * recognized. Since an option's legitimate value could be null, use {@link #getOptionClass} to
   * distinguish between that and an unknown option.
   *
   * <p>optionName is the name of the option as it appears on the command line e.g. {@link
   * Option#name}).
   */
  public Object getOptionValue(String optionName) {
    OptionDetails optionData = transitiveOptionsMap.get(optionName);
    return (optionData == null) ? null : optionData.value;
  }

  /**
   * Returns whether or not the given option supports multiple values at the command line (e.g.
   * "--myoption value1 --myOption value2 ..."). Returns false for unrecognized options. Use
   * {@link #getOptionClass} to distinguish between those and legitimate single-value options.
   *
   * <p>As declared in {@link Option#allowMultiple}, multi-value options are expected to be
   * of type {@code List<T>}.
   */
  public boolean allowsMultipleValues(String optionName) {
    OptionDetails optionData = transitiveOptionsMap.get(optionName);
    return (optionData == null) ? false : optionData.allowsMultiple;
  }
}
