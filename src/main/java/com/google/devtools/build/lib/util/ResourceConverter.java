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

package com.google.devtools.build.lib.util;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.function.DoubleBinaryOperator;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.apache.commons.lang.math.NumberUtils;

/**
 * Converter for options that configure Bazel's resource usage.
 *
 * <p>The option can take either a value or one of the keywords {@code auto}, {@code HOST_CPUS}, or
 * {@code HOST_RAM}, followed by an optional operator in the form {@code [-|*]<float>}.
 *
 * <p>If a keyword is passed, the converter returns the keyword's value in the {@link #keywords}
 * map, scaled by the operation that follows if there is one. All values, explicit and derived, are
 * adjusted for validity.
 *
 * <p>The supplier of the auto value, and, optionally, a max or min allowed value (inclusive), are
 * passed to the constructor.
 */
public class ResourceConverter extends Converters.IntegerConverter {

  private static final ImmutableMap<String, DoubleBinaryOperator> OPERATORS =
      ImmutableMap.<String, DoubleBinaryOperator>builder()
          .put("-", (l, r) -> l - r)
          .put("*", (l, r) -> l * r)
          .build();

  /** Description of the accepted inputs to the converter. */
  private static final String FLAG_SYNTAX =
      "an integer, or a keyword (\"auto\", \"HOST_CPUS\", \"HOST_RAM\"), optionally followed by "
          + "an operation ([-|*]<float>) eg. \"auto\", \"HOST_CPUS*.5\"";

  private final ImmutableMap<String, Supplier<Integer>> keywords;

  private final Pattern validInputPattern;

  private final int minValue;

  private final int maxValue;

  /**
   * Creates a builder for the ResourceConverter class. Requires {@code auto(Supplier<Integer>) }
   * defining the behavior of the {@code auto} keyword. Optional minValue() and maxValue().
   */
  public static ResourceConverterBuilder builder() {
    return new ResourceConverterBuilder();
  }

  /** Creates a converter for options that accept {@value #FLAG_SYNTAX} */
  public static final class ResourceConverterBuilder {

    private int minValue = 1;
    private int maxValue = Integer.MAX_VALUE;
    private ImmutableMap<String, Supplier<Integer>> keywords;
    private Pattern validInputPattern;

    private ResourceConverterBuilder() {};

    /** Defines the behavior of the {@code auto} keyword. */
    // A supplier is used so that the converter responds correctly if host resources are configured
    // after it is constructed.
    public ResourceConverterBuilder auto(Supplier<Integer> autoSupplier) {
      this.keywords =
          ImmutableMap.<String, Supplier<Integer>>builder()
              .put("auto", autoSupplier)
              .put(
                  "HOST_CPUS",
                  () -> (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getCpuUsage()))
              .put(
                  "HOST_RAM",
                  () -> (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getMemoryMb()))
              .build();

      this.validInputPattern =
          Pattern.compile(
              String.format(
                  "(?<keyword>%s)(?<expression>[%s][0-9]?(?:.[0-9]+)?)?",
                  String.join("|", this.keywords.keySet()), String.join("", OPERATORS.keySet())));
      return this;
    }

    /** Sets the minimum allowed value. Defaults to 1. */
    public ResourceConverterBuilder minValue(int minValue) {
      this.minValue = minValue;
      return this;
    }

    /**
     * Sets the maximum allowed value. Defaults to {@code Integer.MAX_VALUE}, effectively no upper
     * bound.
     */
    public ResourceConverterBuilder maxValue(int maxValue) {
      this.maxValue = maxValue;
      return this;
    }

    public ResourceConverter build() throws OptionsParsingException {
      if (keywords == null || validInputPattern == null) {
        throw new OptionsParsingException(
            "ResourceConverterBuilder must call auto() before build().");
      }
      return new ResourceConverter(keywords, validInputPattern, minValue, maxValue);
    }
  }

  private ResourceConverter(
      ImmutableMap<String, Supplier<Integer>> keywords,
      Pattern validInputPattern,
      int minValue,
      int maxValue) {
    this.keywords = keywords;
    this.validInputPattern = validInputPattern;
    this.minValue = minValue;
    this.maxValue = maxValue;
  }

  @Override
  public final Integer convert(String input) throws OptionsParsingException {
    int value;
    if (NumberUtils.isNumber(input)) {
      value = super.convert(input);
      checkValidity(value);
      return value;
    }
    Matcher matcher = validInputPattern.matcher(input);
    if (matcher.matches()) {
      Supplier<Integer> resourceSupplier = keywords.get(matcher.group("keyword"));
      if (resourceSupplier != null) {
        value = applyOperator(matcher.group("expression"), resourceSupplier);
        checkValidity(value);
        return value;
      }
    }
    throw new OptionsParsingException(
        String.format(
            "Parameter '%s' does not follow correct syntax. This flag takes %s.",
            input, FLAG_SYNTAX));
  }

  /** Applies function designated in {@code expression} ([-|*]<float>) to value. */
  private Integer applyOperator(@Nullable String expression, Supplier<Integer> firstOperandSupplier)
      throws OptionsParsingException {
    if (expression == null) {
      return firstOperandSupplier.get();
    }
    for (ImmutableMap.Entry<String, DoubleBinaryOperator> operator : OPERATORS.entrySet()) {
      if (expression.startsWith(operator.getKey())) {
        float secondOperand;
        try {
          secondOperand = Float.parseFloat(expression.substring(operator.getKey().length()));
        } catch (NumberFormatException e) {
          throw new OptionsParsingException(
              String.format("'%s is not a float", expression.substring(operator.getKey().length())),
              e);
        }
        return (int)
            Math.round(
                operator
                    .getValue()
                    .applyAsDouble((float) firstOperandSupplier.get(), secondOperand));
      }
    }
    // This should never happen because we've checked for a valid operator already.
    throw new OptionsParsingException(
        String.format("Parameter value '%s' does not contain a valid operator.", expression));
  }

  /**
   * Checks validity of all values, both calculated and explicitly defined, based on test condition
   * or host capacity.
   */
  private void checkValidity(int value) throws OptionsParsingException {
    if (value < minValue) {
      throw new OptionsParsingException(
          String.format("Value '(%d)' must be at least %d.", value, minValue));
    }
    if (value > maxValue) {
      throw new OptionsParsingException(
          String.format("Value '(%d)' cannot be greater than %d.", value, maxValue));
    }
  }

  @Override
  public String getTypeDescription() {
    return FLAG_SYNTAX;
  }
}
