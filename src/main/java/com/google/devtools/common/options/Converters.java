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
package com.google.devtools.common.options;

import com.google.common.base.Ascii;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.cache.CacheBuilderSpec;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.time.Duration;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/** Some convenient converters used by blaze. Note: These are specific to blaze. */
public final class Converters {

  private static final ImmutableList<String> ENABLED_REPS =
      ImmutableList.of("true", "1", "yes", "t", "y");

  private static final ImmutableList<String> DISABLED_REPS =
      ImmutableList.of("false", "0", "no", "f", "n");

  /** Standard converter for booleans. Accepts common shorthands/synonyms. */
  public static class BooleanConverter implements Converter<Boolean> {
    @Override
    public Boolean convert(String input) throws OptionsParsingException {
      if (input == null) {
        return false;
      }
      input = Ascii.toLowerCase(input);
      if (ENABLED_REPS.contains(input)) {
        return true;
      }
      if (DISABLED_REPS.contains(input)) {
        return false;
      }
      throw new OptionsParsingException("'" + input + "' is not a boolean");
    }

    @Override
    public String getTypeDescription() {
      return "a boolean";
    }
  }

  /** Standard converter for Strings. */
  public static class StringConverter implements Converter<String> {
    @Override
    public String convert(String input) {
      return input;
    }

    @Override
    public String getTypeDescription() {
      return "a string";
    }
  }

  /** Standard converter for integers. */
  public static class IntegerConverter implements Converter<Integer> {
    @Override
    public Integer convert(String input) throws OptionsParsingException {
      try {
        return Integer.decode(input);
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not an int");
      }
    }

    @Override
    public String getTypeDescription() {
      return "an integer";
    }
  }

  /** Standard converter for longs. */
  public static class LongConverter implements Converter<Long> {
    @Override
    public Long convert(String input) throws OptionsParsingException {
      try {
        return Long.decode(input);
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not a long");
      }
    }

    @Override
    public String getTypeDescription() {
      return "a long integer";
    }
  }

  /** Standard converter for doubles. */
  public static class DoubleConverter implements Converter<Double> {
    @Override
    public Double convert(String input) throws OptionsParsingException {
      try {
        return Double.parseDouble(input);
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not a double");
      }
    }

    @Override
    public String getTypeDescription() {
      return "a double";
    }
  }

  /** Standard converter for TriState values. */
  public static class TriStateConverter implements Converter<TriState> {
    @Override
    public TriState convert(String input) throws OptionsParsingException {
      if (input == null) {
        return TriState.AUTO;
      }
      input = Ascii.toLowerCase(input);
      if (input.equals("auto")) {
        return TriState.AUTO;
      }
      if (ENABLED_REPS.contains(input)) {
        return TriState.YES;
      }
      if (DISABLED_REPS.contains(input)) {
        return TriState.NO;
      }
      throw new OptionsParsingException("'" + input + "' is not a boolean");
    }

    @Override
    public String getTypeDescription() {
      return "a tri-state (auto, yes, no)";
    }
  }

  /**
   * Standard "converter" for Void. Should not actually be invoked. For instance, expansion flags
   * are usually Void-typed and do not invoke the converter.
   */
  public static class VoidConverter implements Converter<Void> {
    @Override
    public Void convert(String input) throws OptionsParsingException {
      if (input == null || input.equals("null")) {
        return null; // expected input, return is unused so null is fine.
      }
      throw new OptionsParsingException("'" + input + "' unexpected");
    }

    @Override
    public String getTypeDescription() {
      return "";
    }
  }

  /** Standard converter for the {@link java.time.Duration} type. */
  public static class DurationConverter implements Converter<Duration> {
    private final Pattern durationRegex = Pattern.compile("^([0-9]+)(d|h|m|s|ms)$");

    @Override
    public Duration convert(String input) throws OptionsParsingException {
      // To be compatible with the previous parser, '0' doesn't need a unit.
      if ("0".equals(input)) {
        return Duration.ZERO;
      }
      Matcher m = durationRegex.matcher(input);
      if (!m.matches()) {
        throw new OptionsParsingException("Illegal duration '" + input + "'.");
      }
      long duration = Long.parseLong(m.group(1));
      String unit = m.group(2);
      switch (unit) {
        case "d":
          return Duration.ofDays(duration);
        case "h":
          return Duration.ofHours(duration);
        case "m":
          return Duration.ofMinutes(duration);
        case "s":
          return Duration.ofSeconds(duration);
        case "ms":
          return Duration.ofMillis(duration);
        default:
          throw new IllegalStateException(
              "This must not happen. Did you update the regex without the switch case?");
      }
    }

    @Override
    public String getTypeDescription() {
      return "An immutable length of time.";
    }
  }

  // 1:1 correspondence with UsesOnlyCoreTypes.CORE_TYPES.
  /**
   * The converters that are available to the options parser by default. These are used if the
   * {@code @Option} annotation does not specify its own {@code converter}, and its type is one of
   * the following.
   */
  public static final ImmutableMap<Class<?>, Converter<?>> DEFAULT_CONVERTERS =
      new ImmutableMap.Builder<Class<?>, Converter<?>>()
          .put(String.class, new Converters.StringConverter())
          .put(int.class, new Converters.IntegerConverter())
          .put(long.class, new Converters.LongConverter())
          .put(double.class, new Converters.DoubleConverter())
          .put(boolean.class, new Converters.BooleanConverter())
          .put(TriState.class, new Converters.TriStateConverter())
          .put(Duration.class, new Converters.DurationConverter())
          .put(Void.class, new Converters.VoidConverter())
          .build();

  /**
   * Join a list of words as in English. Examples: "nothing" "one" "one or two" "one and two" "one,
   * two or three". "one, two and three". The toString method of each element is used.
   */
  static String joinEnglishList(Iterable<?> choices) {
    StringBuilder buf = new StringBuilder();
    for (Iterator<?> ii = choices.iterator(); ii.hasNext(); ) {
      Object choice = ii.next();
      if (buf.length() > 0) {
        buf.append(ii.hasNext() ? ", " : " or ");
      }
      buf.append(choice);
    }
    return buf.length() == 0 ? "nothing" : buf.toString();
  }

  public static class SeparatedOptionListConverter implements Converter<List<String>> {

    private final String separatorDescription;
    private final Splitter splitter;
    private final boolean allowEmptyValues;

    protected SeparatedOptionListConverter(
        char separator, String separatorDescription, boolean allowEmptyValues) {
      this.separatorDescription = separatorDescription;
      this.splitter = Splitter.on(separator);
      this.allowEmptyValues = allowEmptyValues;
    }

    @Override
    public List<String> convert(String input) throws OptionsParsingException {
      List<String> result =
          input.isEmpty() ? ImmutableList.of() : ImmutableList.copyOf(splitter.split(input));
      if (!allowEmptyValues && result.contains("")) {
        // If the list contains exactly the empty string, it means an empty value was passed and we
        // should instead return an empty list.
        if (result.size() == 1) {
          return ImmutableList.of();
        }

        throw new OptionsParsingException(
            "Empty values are not allowed as part of this " + getTypeDescription());
      }
      return result;
    }

    @Override
    public String getTypeDescription() {
      return separatorDescription + "-separated list of options";
    }
  }

  public static class CommaSeparatedOptionListConverter extends SeparatedOptionListConverter {
    public CommaSeparatedOptionListConverter() {
      super(',', "comma", true);
    }
  }

  public static class CommaSeparatedNonEmptyOptionListConverter
      extends SeparatedOptionListConverter {
    public CommaSeparatedNonEmptyOptionListConverter() {
      super(',', "comma", false);
    }
  }

  public static class ColonSeparatedOptionListConverter extends SeparatedOptionListConverter {
    public ColonSeparatedOptionListConverter() {
      super(':', "colon", true);
    }
  }

  public static class LogLevelConverter implements Converter<Level> {

    public static final Level[] LEVELS =
        new Level[] {
          Level.OFF, Level.SEVERE, Level.WARNING, Level.INFO, Level.FINE, Level.FINER, Level.FINEST
        };

    @Override
    public Level convert(String input) throws OptionsParsingException {
      try {
        int level = Integer.parseInt(input);
        return LEVELS[level];
      } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
        throw new OptionsParsingException("Not a log level: " + input);
      }
    }

    @Override
    public String getTypeDescription() {
      return "0 <= an integer <= " + (LEVELS.length - 1);
    }
  }

  /** Checks whether a string is part of a set of strings. */
  public static class StringSetConverter implements Converter<String> {

    // TODO(bazel-team): if this class never actually contains duplicates, we could s/List/Set/
    // here.
    private final List<String> values;

    public StringSetConverter(String... values) {
      this.values = ImmutableList.copyOf(values);
    }

    @Override
    public String convert(String input) throws OptionsParsingException {
      if (values.contains(input)) {
        return input;
      }

      throw new OptionsParsingException("Not one of " + values);
    }

    @Override
    public String getTypeDescription() {
      return joinEnglishList(values);
    }
  }

  /** Checks whether a string is a valid regex pattern and compiles it. */
  public static class RegexPatternConverter implements Converter<RegexPatternOption> {

    @Override
    public RegexPatternOption convert(String input) throws OptionsParsingException {
      try {
        return RegexPatternOption.create(Pattern.compile(input));
      } catch (PatternSyntaxException e) {
        throw new OptionsParsingException("Not a valid regular expression: " + e.getMessage());
      }
    }

    @Override
    public String getTypeDescription() {
      return "a valid Java regular expression";
    }
  }

  /** Limits the length of a string argument. */
  public static class LengthLimitingConverter implements Converter<String> {
    private final int maxSize;

    public LengthLimitingConverter(int maxSize) {
      this.maxSize = maxSize;
    }

    @Override
    public String convert(String input) throws OptionsParsingException {
      if (input.length() > maxSize) {
        throw new OptionsParsingException("Input must be " + getTypeDescription());
      }
      return input;
    }

    @Override
    public String getTypeDescription() {
      return "a string <= " + maxSize + " characters";
    }
  }

  /** Checks whether an integer is in the given range. */
  public static class RangeConverter implements Converter<Integer> {
    final int minValue;
    final int maxValue;

    public RangeConverter(int minValue, int maxValue) {
      this.minValue = minValue;
      this.maxValue = maxValue;
    }

    @Override
    public Integer convert(String input) throws OptionsParsingException {
      try {
        Integer value = Integer.parseInt(input);
        if (value < minValue) {
          throw new OptionsParsingException("'" + input + "' should be >= " + minValue);
        } else if (value < minValue || value > maxValue) {
          throw new OptionsParsingException("'" + input + "' should be <= " + maxValue);
        }
        return value;
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not an int");
      }
    }

    @Override
    public String getTypeDescription() {
      if (minValue == Integer.MIN_VALUE) {
        if (maxValue == Integer.MAX_VALUE) {
          return "an integer";
        } else {
          return "an integer, <= " + maxValue;
        }
      } else if (maxValue == Integer.MAX_VALUE) {
        return "an integer, >= " + minValue;
      } else {
        return "an integer in "
            + (minValue < 0 ? "(" + minValue + ")" : minValue)
            + "-"
            + maxValue
            + " range";
      }
    }
  }

  /**
   * A converter for variable assignments from the parameter list of a blaze command invocation.
   * Assignments are expected to have the form "name=value", where names and values are defined to
   * be as permissive as possible.
   */
  public static class AssignmentConverter implements Converter<Map.Entry<String, String>> {

    @Override
    public Map.Entry<String, String> convert(String input) throws OptionsParsingException {
      int pos = input.indexOf("=");
      if (pos <= 0) {
        throw new OptionsParsingException(
            "Variable definitions must be in the form of a 'name=value' assignment");
      }
      String name = input.substring(0, pos);
      String value = input.substring(pos + 1);
      return Maps.immutableEntry(name, value);
    }

    @Override
    public String getTypeDescription() {
      return "a 'name=value' assignment";
    }
  }

  /**
   * Base converter for assignments from a value to a list of values. Both the key type as well as
   * the type for all instances in the list of values are processed via passed converters.
   */
  public abstract static class AssignmentToListOfValuesConverter<K, V>
      implements Converter<Map.Entry<K, List<V>>> {

    /** Whether to allow keys in the assignment to be empty (i.e. just a list of values) */
    public enum AllowEmptyKeys {
      YES,
      NO
    }

    private static final Splitter SPLITTER = Splitter.on(',');

    private final Converter<K> keyConverter;
    private final Converter<V> valueConverter;
    private final AllowEmptyKeys allowEmptyKeys;

    public AssignmentToListOfValuesConverter(
        Converter<K> keyConverter, Converter<V> valueConverter, AllowEmptyKeys allowEmptyKeys) {
      this.keyConverter = keyConverter;
      this.valueConverter = valueConverter;
      this.allowEmptyKeys = allowEmptyKeys;
    }

    @Override
    public Map.Entry<K, List<V>> convert(String input) throws OptionsParsingException {
      int pos = input.indexOf("=");
      if (allowEmptyKeys == AllowEmptyKeys.NO && pos <= 0) {
        throw new OptionsParsingException(
            "Must be in the form of a 'key=value[,value]' assignment");
      }

      String key = pos <= 0 ? "" : input.substring(0, pos);
      List<String> values = SPLITTER.splitToList(input.substring(pos + 1));
      if (values.contains("")) {
        // If the list contains exactly the empty string, it means an empty value was passed and we
        // should instead return an empty list.
        if (values.size() == 1) {
          values = ImmutableList.of();
        } else {
          throw new OptionsParsingException(
              "Variable definitions must not contain empty strings or leading / trailing commas");
        }
      }
      ImmutableList.Builder<V> convertedValues = ImmutableList.builder();
      for (String value : values) {
        convertedValues.add(valueConverter.convert(value));
      }
      return Maps.immutableEntry(keyConverter.convert(key), convertedValues.build());
    }
  }

  /**
   * A converter for variable assignments from the parameter list of a blaze command invocation.
   * Assignments are expected to have the form {@code [name=]value1[,..,valueN]}, where names and
   * values are defined to be as permissive as possible. If no name is provided, "" is used.
   */
  public static class StringToStringListConverter
      extends AssignmentToListOfValuesConverter<String, String> {

    public StringToStringListConverter() {
      super(new StringConverter(), new StringConverter(), AllowEmptyKeys.YES);
    }

    @Override
    public String getTypeDescription() {
      return "a '[name=]value1[,..,valueN]' assignment";
    }
  }

  /**
   * A converter for variable assignments from the parameter list of a blaze command invocation.
   * Assignments are expected to have the form "name[=value]", where names and values are defined to
   * be as permissive as possible and value part can be optional (in which case it is considered to
   * be null).
   */
  public static class OptionalAssignmentConverter implements Converter<Map.Entry<String, String>> {

    @Override
    public Map.Entry<String, String> convert(String input) throws OptionsParsingException {
      int pos = input.indexOf('=');
      if (pos == 0 || input.length() == 0) {
        throw new OptionsParsingException(
            "Variable definitions must be in the form of a 'name=value' or 'name' assignment");
      } else if (pos < 0) {
        return Maps.immutableEntry(input, null);
      }
      String name = input.substring(0, pos);
      String value = input.substring(pos + 1);
      return Maps.immutableEntry(name, value);
    }

    @Override
    public String getTypeDescription() {
      return "a 'name=value' assignment with an optional value part";
    }
  }

  /**
   * A converter for named integers of the form "[name=]value". When no name is specified, an empty
   * string is used for the key.
   */
  public static class NamedIntegersConverter implements Converter<Map.Entry<String, Integer>> {

    @Override
    public Map.Entry<String, Integer> convert(String input) throws OptionsParsingException {
      int pos = input.indexOf('=');
      if (pos == 0 || input.length() == 0) {
        throw new OptionsParsingException(
            "Specify either 'value' or 'name=value', where 'value' is an integer");
      } else if (pos < 0) {
        try {
          return Maps.immutableEntry("", Integer.parseInt(input));
        } catch (NumberFormatException e) {
          throw new OptionsParsingException("'" + input + "' is not an int");
        }
      }
      String name = input.substring(0, pos);
      String value = input.substring(pos + 1);
      try {
        return Maps.immutableEntry(name, Integer.parseInt(value));
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + value + "' is not an int");
      }
    }

    @Override
    public String getTypeDescription() {
      return "an integer or a named integer, 'name=value'";
    }
  }

  public static class HelpVerbosityConverter extends EnumConverter<OptionsParser.HelpVerbosity> {
    public HelpVerbosityConverter() {
      super(OptionsParser.HelpVerbosity.class, "--help_verbosity setting");
    }
  }

  /**
   * A converter to check whether an integer denoting a percentage is in a valid range: [0, 100].
   */
  public static class PercentageConverter extends RangeConverter {
    public PercentageConverter() {
      super(0, 100);
    }
  }

  /**
   * A {@link Converter} for {@link CacheBuilderSpec}. The spec may be empty, in which case this
   * converter returns null.
   */
  public static class CacheBuilderSpecConverter implements Converter<CacheBuilderSpec> {
    @Override
    public CacheBuilderSpec convert(String spec) throws OptionsParsingException {
      try {
        return Strings.isNullOrEmpty(spec) ? null : CacheBuilderSpec.parse(spec);
      } catch (IllegalArgumentException e) {
        throw new OptionsParsingException("Failed to parse CacheBuilderSpec: " + e.getMessage(), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "Converts to a CacheBuilderSpec, or null if the input is empty";
    }
  }
}
