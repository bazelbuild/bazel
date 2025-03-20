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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.util.RegexFilter.RegexFilterConverter;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Models options that can be added to a command line when a label matches a given {@link
 * RegexFilter}.
 */
public final class PerLabelOptions {
  /** The filter used to match labels */
  private final RegexFilter regexFilter;

  /** The list of options to add when the filter matches a label */
  private final List<String> optionsList;

  /**
   * Converts a String to a {@link PerLabelOptions} object. The syntax of the string is {@code
   * regex_filter@option_1,option_2,...,option_n}. Where regex_filter stands for the String
   * representation of a {@link RegexFilter}, and {@code option_1} to {@code option_n} stand for
   * arbitrary command line options. If an option contains a comma it has to be quoted with a
   * backslash. Options can contain @. Only the first @ is used to split the string.
   */
  public static class PerLabelOptionsConverter extends Converter.Contextless<PerLabelOptions> {

    @Override
    public PerLabelOptions convert(String input) throws OptionsParsingException {
      int atIndex = input.indexOf('@');
      RegexFilterConverter converter = new RegexFilter.RegexFilterConverter();
      if (atIndex < 0) {
        return new PerLabelOptions(converter.convert(input), ImmutableList.of());
      } else {
        String filterPiece = input.substring(0, atIndex);
        String optionsPiece = input.substring(atIndex + 1);
        List<String> optionsList = new ArrayList<>();
        for (String option : optionsPiece.split("(?<!\\\\),")) { // Split on ',' but not on '\,'
          if (option != null && !option.trim().isEmpty()) {
            optionsList.add(option.replace("\\,", ","));
          }
        }
        return new PerLabelOptions(converter.convert(filterPiece), optionsList);
      }
    }

    @Override
    public boolean starlarkConvertible() {
      return true;
    }

    @Override
    public String reverseForStarlark(Object converted) {
      PerLabelOptions typedValue = (PerLabelOptions) converted;
      return String.format(
          "%s@%s",
          typedValue.getRegexFilter().toOriginalString(),
          String.join(",", typedValue.getOptions()));
    }

    @Override
    public String getTypeDescription() {
      return "a comma-separated list of regex expressions with prefix '-' specifying"
      + " excluded paths followed by an @ and a comma separated list of options";
    }
  }

  public PerLabelOptions(RegexFilter regexFilter, List<String> optionsList) {
    this.regexFilter = regexFilter;
    this.optionsList = optionsList;
  }

  /**
   * @return true if the given label is matched by the {@link RegexFilter}.
   */
  public boolean isIncluded(Label label) {
    return regexFilter.isIncluded(label.toString());
  }

  /**
   * @return true if the execution path (which includes the base name of the file)
   * of the given file is matched by the {@link RegexFilter}.
   */
  public boolean isIncluded(Artifact artifact) {
    return regexFilter.isIncluded(artifact.getExecPathString());
  }

  /**
   * Returns the list of options to add to a command line.
   */
  public List<String> getOptions() {
    return optionsList;
  }

  RegexFilter getRegexFilter() {
    return regexFilter;
  }

  @Override
  public String toString() {
    return regexFilter + " Options: " + optionsList;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof PerLabelOptions otherOptions)) {
      return false;
    }
    return this.regexFilter.equals(otherOptions.regexFilter)
        && this.optionsList.equals(otherOptions.optionsList);
  }

  @Override
  public int hashCode() {
    return Objects.hash(regexFilter, optionsList);
  }
}
