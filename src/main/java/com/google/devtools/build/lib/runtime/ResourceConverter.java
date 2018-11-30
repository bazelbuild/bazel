// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.function.DoubleBinaryOperator;
import java.util.function.Supplier;
import org.apache.commons.lang.math.NumberUtils;

/**
 * Converter for options that accept a value, or one of the keywords in the validKeywords map,
 * followed by an optional operator in the form [-|*]<float>. If a keyword is passed, the converter
 * returns the keyword's value in the validKeywords map, scaled by the operation that follows if
 * there is one. All values, explicit and derived, are adjusted for validity. A map of valid
 * keywords, and the functions they correspond to, is passed to the constructor.
 */
public abstract class ResourceConverter extends Converters.IntegerConverter {

  private static final ImmutableMap<String, DoubleBinaryOperator> OPERATORS =
      ImmutableMap.<String, DoubleBinaryOperator>builder()
          .put("-", (l, r) -> l - r)
          .put("*", (l, r) -> l * r)
          .build();

  private final ImmutableMap<String, Supplier<Integer>> validKeywords;

  private final String flagSyntax;

  /**
   * Creates a converter whose behavior is defined by the provided map of keywords to functions. A
   * supplier is used so that the converter responds correctly if host resources are configured
   * after it is constructed.
   *
   * @param keywords a map of keywords to suppliers of their values. Keywords must not be starting
   *     substrings of each other and are case sensitive. A typical keyword set would contain "auto"
   *     linked to a function that calculates a reasonable flag value based on host resources, and a
   *     keyword containing "HOST", defining the host's capacity in terms of threads, CPUs, RAM,
   *     etc. Valid functions return an integer.
   */
  protected ResourceConverter(ImmutableMap<String, Supplier<Integer>> keywords)
      throws OptionsParsingException {

    for (String keyword : keywords.keySet()) {
      if (keyword.matches("(" + String.join("|", keywords.keySet()) + ").+")) {
        throw new OptionsParsingException(
            String.format(
                "Keywords (%s) must not be starting substrings of each other.",
                String.join(",", keywords.keySet())));
      }
    }

    validKeywords = keywords;
    flagSyntax =
        "["
            + String.join("|", validKeywords.keySet())
            + "]["
            + String.join("|", OPERATORS.keySet())
            + "]<float>";
  }

  /**
   * {@inheritDoc}
   *
   * @param input A value or a member of validKeywords, optionally followed by [-|*]<float>
   */
  @Override
  public final Integer convert(String input) throws OptionsParsingException {
    if (isNullOrEmpty(input)) {
      return null;
    }
    if (NumberUtils.isNumber(input)) {
      return adjustValue(super.convert(input));
    }
    for (String keyword : validKeywords.keySet()) {
      if (input.startsWith(keyword)) {
        int resourceValue = applyOperator(input.replace(keyword, ""), validKeywords.get(keyword));
        return adjustValue(resourceValue);
      }
    }
    // Not numeric and not valid parameter format.
    throw new OptionsParsingException(
        String.format(
            "Parameter '%s' does not follow correct syntax. This flag takes %s.",
            input, flagSyntax));
  }

  /**
   * Applies function designated in functionString ([-|*]<float>) to value. Empty functionString
   * returns value (eg. for "auto" or unscaled host value input).
   */
  private Integer applyOperator(String functionString, Supplier<Integer> firstOperandSupplier)
      throws OptionsParsingException {
    if (isNullOrEmpty(functionString)) {
      return firstOperandSupplier.get();
    }
    for (String opString : OPERATORS.keySet()) {
      if (functionString.startsWith(opString)) {
        float adjustBy;
        try {
          adjustBy = Float.parseFloat(functionString.substring(opString.length()));
        } catch (NumberFormatException e) {
          throw new OptionsParsingException(
              String.format("'%s is not a float", functionString.substring(opString.length())), e);
        }
        return (int)
            Math.round(
                OPERATORS
                    .get(opString)
                    .applyAsDouble((float) firstOperandSupplier.get(), adjustBy));
      }
    }
    throw new OptionsParsingException(
        String.format("Parameter value '%s' does not contain a valid operator.", functionString));
  }

  /**
   * Checks validity of all values, both calculated and explicitly defined, based on test condition
   * or host capacity. Fixes value or throws an error. Default return original value.
   */
  protected Integer adjustValue(int value) throws OptionsParsingException {
    return value;
  }

  @Override
  public String getTypeDescription() {
    return "\"auto\" or \"" + flagSyntax + "\" or " + super.getTypeDescription();
  }
}
