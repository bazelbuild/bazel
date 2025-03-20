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
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import java.util.function.DoubleBinaryOperator;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

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
public abstract class ResourceConverter<T extends Number & Comparable<T>>
    extends Converter.Contextless<T> {
  public static final String AUTO_KEYWORD = "auto";
  public static final String HOST_CPUS_KEYWORD = "HOST_CPUS";
  public static final String HOST_RAM_KEYWORD = "HOST_RAM";

  public static final Supplier<Integer> HOST_CPUS_SUPPLIER =
      () -> (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getCpuUsage());
  public static final Supplier<Integer> HOST_RAM_SUPPLIER =
      () -> (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getMemoryMb());

  /** Resource converter for assignments. */
  public static class AssignmentConverter extends Converter.Contextless<Map.Entry<String, Double>> {
    private static final Converters.AssignmentConverter assignment =
        new Converters.AssignmentConverter();
    private static final ResourceConverter.DoubleConverter resource =
        new ResourceConverter.DoubleConverter(
            ImmutableMap.of(
                HOST_CPUS_KEYWORD, () -> (double) HOST_CPUS_SUPPLIER.get(),
                HOST_RAM_KEYWORD, () -> (double) HOST_RAM_SUPPLIER.get()),
            0.0,
            Double.MAX_VALUE);

    @Override
    public Map.Entry<String, Double> convert(String input) throws OptionsParsingException {
      Map.Entry<String, String> s = assignment.convert(input);
      return Map.entry(s.getKey(), resource.convert(s.getValue()));
    }

    @Override
    public String getTypeDescription() {
      return "a named double, 'name=value', where value is " + resource.getTypeDescription();
    }
  }

  /** Resource converter for integers. */
  public static class IntegerConverter extends ResourceConverter<Integer> {
    private static final Converters.IntegerConverter converter = new Converters.IntegerConverter();

    public IntegerConverter(Supplier<Integer> auto, int minValue, int maxValue) {
      this(
          ImmutableMap.of(
              AUTO_KEYWORD,
              auto,
              HOST_CPUS_KEYWORD,
              HOST_CPUS_SUPPLIER,
              HOST_RAM_KEYWORD,
              HOST_RAM_SUPPLIER),
          minValue,
          maxValue);
    }

    public IntegerConverter(
        ImmutableMap<String, Supplier<Integer>> keywords, int minValue, int maxValue) {
      super(keywords, minValue, maxValue);
    }

    @Override
    public Integer convert(String input) throws OptionsParsingException {
      return Ints.tryParse(input) != null
          ? checkAndLimit(converter.convert(input))
          : checkAndLimit((int) Math.round(convertKeyword(input)));
    }
  }

  /** Resource converter for doubles. */
  public static class DoubleConverter extends ResourceConverter<Double> {
    private static final Converters.DoubleConverter converter = new Converters.DoubleConverter();

    public DoubleConverter(Supplier<Double> auto, double minValue, double maxValue) {
      this(
          ImmutableMap.of(
              AUTO_KEYWORD, auto,
              HOST_CPUS_KEYWORD, () -> (double) HOST_CPUS_SUPPLIER.get(),
              HOST_RAM_KEYWORD, () -> (double) HOST_RAM_SUPPLIER.get()),
          minValue,
          maxValue);
    }

    public DoubleConverter(
        ImmutableMap<String, Supplier<Double>> keywords, double minValue, double maxValue) {
      super(keywords, minValue, maxValue);
    }

    @Override
    public Double convert(String input) throws OptionsParsingException {
      return Doubles.tryParse(input) != null
          ? checkAndLimit(converter.convert(input))
          : convertKeyword(input);
    }
  }

  private static final ImmutableMap<String, DoubleBinaryOperator> OPERATORS =
      ImmutableMap.<String, DoubleBinaryOperator>builder()
          .put("-", (l, r) -> l - r)
          .put("*", (l, r) -> l * r)
          .build();

  /** Description of the accepted inputs to the converter. */
  public static final String FLAG_SYNTAX =
      "an integer, or a keyword (\""
          + AUTO_KEYWORD
          + "\", \""
          + HOST_CPUS_KEYWORD
          + "\", \""
          + HOST_RAM_KEYWORD
          + "\"), optionally followed by an operation ([-|*]<float>) eg. \""
          + AUTO_KEYWORD
          + "\", \""
          + HOST_CPUS_KEYWORD
          + "*.5\"";

  private final ImmutableMap<String, Supplier<T>> keywords;

  private final Pattern validInputPattern;

  protected final T minValue;

  protected final T maxValue;

  /**
   * Constructs a ResourceConverter for options that take keywords other than the default set.
   *
   * @param keywords a map of keyword to the suppliers of their values
   */
  public ResourceConverter(ImmutableMap<String, Supplier<T>> keywords, T minValue, T maxValue) {
    this.keywords = keywords;
    this.validInputPattern =
        Pattern.compile(
            String.format(
                "(?<keyword>%s)(?<expression>[%s][0-9]?(?:.[0-9]+)?)?",
                String.join("|", this.keywords.keySet()), String.join("", OPERATORS.keySet())));
    this.minValue = minValue;
    this.maxValue = maxValue;
  }

  public final Double convertKeyword(String input) throws OptionsParsingException {
    Matcher matcher = validInputPattern.matcher(input);
    if (matcher.matches()) {
      Supplier<T> resourceSupplier = keywords.get(matcher.group("keyword"));
      if (resourceSupplier != null) {
        return applyOperator(matcher.group("expression"), resourceSupplier);
      }
    }
    throw new OptionsParsingException(
        String.format(
            "Parameter '%s' does not follow correct syntax. This flag takes %s.",
            input, getTypeDescription()));
  }

  /** Applies function designated in {@code expression} ([-|*]<float>) to value. */
  private Double applyOperator(@Nullable String expression, Supplier<T> firstOperandSupplier)
      throws OptionsParsingException {
    if (expression == null) {
      return firstOperandSupplier.get().doubleValue();
    }
    for (Map.Entry<String, DoubleBinaryOperator> operator : OPERATORS.entrySet()) {
      if (expression.startsWith(operator.getKey())) {
        float secondOperand;
        try {
          secondOperand = Float.parseFloat(expression.substring(operator.getKey().length()));
        } catch (NumberFormatException e) {
          throw new OptionsParsingException(
              String.format("'%s is not a float", expression.substring(operator.getKey().length())),
              e);
        }
        return operator
            .getValue()
            .applyAsDouble(firstOperandSupplier.get().doubleValue(), secondOperand);
      }
    }
    // This should never happen because we've checked for a valid operator already.
    throw new OptionsParsingException(
        String.format("Parameter value '%s' does not contain a valid operator.", expression));
  }

  /**
   * Checks validity of a resource value against min/max constraints. Implementations may choose to
   * either raise an exception on out-of-bounds values, or adjust them to within the constraints.
   */
  @CanIgnoreReturnValue
  public T checkAndLimit(T value) throws OptionsParsingException {
    if (value.compareTo(minValue) < 0) {
      throw new OptionsParsingException(
          String.format(
              "Value '(%f)' must be at least %f.", value.doubleValue(), minValue.doubleValue()));
    }
    if (value.compareTo(maxValue) > 0) {
      throw new OptionsParsingException(
          String.format(
              "Value '(%f)' cannot be greater than %f.",
              value.doubleValue(), maxValue.doubleValue()));
    }
    return value;
  }

  @Override
  public String getTypeDescription() {
    return FLAG_SYNTAX;
  }
}
