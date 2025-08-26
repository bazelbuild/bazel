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

import static com.google.devtools.common.options.OptionsParser.STARLARK_SKIPPED_PREFIXES;

import com.github.benmanes.caffeine.cache.CaffeineSpec;
import com.google.common.base.Ascii;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.StringEncoding;
import java.time.Duration;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.logging.Level;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import javax.annotation.Nullable;

/** Some convenient converters used by blaze. Note: These are specific to blaze. */
public final class Converters {
  /**
   * The name of the flag used for shorthand aliasing in blaze. {@see
   * com.google.devtools.build.lib.analysis.config.CoreOptions#commandLineFlagAliases} for the
   * option definition.
   */
  public static final String BLAZE_ALIASING_FLAG = "flag_alias";

  private static final ImmutableSet<String> ENABLED_REPS =
      ImmutableSet.of("true", "1", "yes", "t", "y");

  private static final ImmutableSet<String> DISABLED_REPS =
      ImmutableSet.of("false", "0", "no", "f", "n");

  /** Standard converter for booleans. Accepts common shorthands/synonyms. */
  public static class BooleanConverter extends Converter.Contextless<Boolean> {
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
  public static class StringConverter extends Converter.Contextless<String> {
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
  public static class IntegerConverter extends Converter.Contextless<Integer> {
    @Override
    public Integer convert(String input) throws OptionsParsingException {
      try {
        return Integer.decode(input);
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not an int", e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "an integer";
    }
  }

  /** Standard converter for longs. */
  public static class LongConverter extends Converter.Contextless<Long> {
    @Override
    public Long convert(String input) throws OptionsParsingException {
      try {
        return Long.decode(input);
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not a long", e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a long integer";
    }
  }

  /** Standard converter for doubles. */
  public static class DoubleConverter extends Converter.Contextless<Double> {
    @Override
    public Double convert(String input) throws OptionsParsingException {
      try {
        return Double.parseDouble(input);
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not a double", e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a double";
    }
  }

  /** Standard converter for TriState values. */
  public static class TriStateConverter extends Converter.Contextless<TriState> {
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
  public static class VoidConverter extends Converter.Contextless<Void> {
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
  public static class DurationConverter extends Converter.Contextless<Duration> {
    private static final Pattern DURATION_REGEX = Pattern.compile("^([0-9]+)(d|h|m|s|ms|ns)$");

    @Override
    public Duration convert(String input) throws OptionsParsingException {
      // To be compatible with the previous parser, '0' doesn't need a unit.
      if ("0".equals(input)) {
        return Duration.ZERO;
      }
      Matcher m = DURATION_REGEX.matcher(input);
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
        case "ns":
          return Duration.ofNanos(duration);
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

  /** Converter for a list of options, separated by some separator character. */
  public static class SeparatedOptionListConverter
      extends Converter.Contextless<ImmutableList<String>> {
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
    public ImmutableList<String> convert(String input) throws OptionsParsingException {
      ImmutableList<String> result =
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

  /**
   * Converter for options separated by some separator character, where order and count do not
   * matter, i.e. semantically it is a set, not a list.
   */
  public static class SeparatedOptionSetConverter extends SeparatedOptionListConverter {
    private final String separatorDescription;

    protected SeparatedOptionSetConverter(
        char separator, String separatorDescription, boolean allowEmptyValues) {
      super(separator, separatorDescription, allowEmptyValues);
      this.separatorDescription = separatorDescription;
    }

    @Override
    public ImmutableList<String> convert(String input) throws OptionsParsingException {
      ImmutableList<String> result = super.convert(input);
      return result.stream().distinct().sorted().collect(ImmutableList.toImmutableList());
    }

    @Override
    public String getTypeDescription() {
      return separatorDescription + "-separated set of options";
    }
  }

  /**
   * Converter for comma separated values, where
   * <li>order and multiplicity preserved
   * <li>empty values are preserved
   */
  public static class CommaSeparatedOptionListConverter extends SeparatedOptionListConverter {
    public CommaSeparatedOptionListConverter() {
      super(',', "comma", true);
    }
  }

  /**
   * Converter for comma separated values, where
   * <li>order and multiplicity preserved
   * <li>empty values are filtered out
   */
  public static class CommaSeparatedNonEmptyOptionListConverter
      extends SeparatedOptionListConverter {
    public CommaSeparatedNonEmptyOptionListConverter() {
      super(',', "comma", false);
    }
  }

  /**
   * Converter for colon separated values, where
   * <li>order and multiplicity preserved
   * <li>empty values are preserved
   */
  public static class ColonSeparatedOptionListConverter extends SeparatedOptionListConverter {
    public ColonSeparatedOptionListConverter() {
      super(':', "colon", true);
    }
  }

  /**
   * Converter for colon separated values, where
   * <li>order and multiplicity are assumed to not matter
   * <li>empty values are preserved
   */
  public static class CommaSeparatedOptionSetConverter extends SeparatedOptionSetConverter {
    public CommaSeparatedOptionSetConverter() {
      super(',', "comma", true);
    }
  }

  /** Converter for {@link Level}. */
  public static class LogLevelConverter extends Converter.Contextless<Level> {

    static final ImmutableList<Level> LEVELS =
        ImmutableList.of(
            Level.OFF,
            Level.SEVERE,
            Level.WARNING,
            Level.INFO,
            Level.FINE,
            Level.FINER,
            Level.FINEST);

    @Override
    public Level convert(String input) throws OptionsParsingException {
      try {
        int level = Integer.parseInt(input);
        return LEVELS.get(level);
      } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
        throw new OptionsParsingException("Not a log level: " + input, e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "0 <= an integer <= " + (LEVELS.size() - 1);
    }
  }

  /** Checks whether a string is part of a set of strings. */
  public static class StringSetConverter extends Converter.Contextless<String> {

    // TODO(bazel-team): if this class never actually contains duplicates, we could s/List/Set/
    // here.
    private final ImmutableList<String> values;

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
  public static class RegexPatternConverter extends Converter.Contextless<RegexPatternOption> {

    @Override
    public RegexPatternOption convert(String input) throws OptionsParsingException {
      try {
        return RegexPatternOption.create(
            Pattern.compile(StringEncoding.internalToUnicode(input), Pattern.DOTALL));
      } catch (PatternSyntaxException e) {
        throw new OptionsParsingException("Not a valid regular expression: " + e.getMessage());
      }
    }

    @Override
    public String getTypeDescription() {
      return "a valid Java regular expression";
    }
  }

  /** Checks whether an integer is in the given range. */
  public static class RangeConverter extends Converter.Contextless<Integer> {
    final int minValue;
    final int maxValue;

    public RangeConverter(int minValue, int maxValue) {
      this.minValue = minValue;
      this.maxValue = maxValue;
    }

    @Override
    public Integer convert(String input) throws OptionsParsingException {
      try {
        int value = Integer.parseInt(input);
        if (value < minValue) {
          throw new OptionsParsingException("'" + input + "' should be >= " + minValue);
        } else if (value < minValue || value > maxValue) {
          throw new OptionsParsingException("'" + input + "' should be <= " + maxValue);
        }
        return value;
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not an int", e);
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
  public static class AssignmentConverter extends Converter.Contextless<Map.Entry<String, String>> {

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

  /** A converter for for assignments from a string value to a float value. */
  public static class StringToDoubleAssignmentConverter
      extends Converter.Contextless<Map.Entry<String, Double>> {
    private static final AssignmentConverter baseConverter = new AssignmentConverter();

    @Override
    public Map.Entry<String, Double> convert(String input)
        throws OptionsParsingException, NumberFormatException {
      Map.Entry<String, String> stringEntry = baseConverter.convert(input);
      return Maps.immutableEntry(stringEntry.getKey(), Double.parseDouble(stringEntry.getValue()));
    }

    @Override
    public String getTypeDescription() {
      return "a named float, 'name=value'";
    }
  }

  /**
   * A converter for command line flag aliases. It does additional validation on the name and value
   * of the assignment to ensure they conform to the naming limitations.
   */
  public static class FlagAliasConverter extends AssignmentConverter {

    @Override
    public Map.Entry<String, String> convert(String input) throws OptionsParsingException {
      Map.Entry<String, String> entry = super.convert(input);
      String shortForm = entry.getKey();
      String longForm = entry.getValue();

      String cmdLineAlias = "--" + BLAZE_ALIASING_FLAG + "=" + input;

      if (!Pattern.matches("([\\w])*", shortForm)) {
        throw new OptionsParsingException(
            shortForm + " should only consist of word characters to be a valid alias name.",
            cmdLineAlias);
      }
      if (longForm.contains("=")) {
        throw new OptionsParsingException(
            "--" + BLAZE_ALIASING_FLAG + " does not support flag value assignment.", cmdLineAlias);
      }

      // Remove this check if native options are permitted to be aliased
      longForm = "--" + longForm;
      if (STARLARK_SKIPPED_PREFIXES.stream().noneMatch(longForm::startsWith)) {
        throw new OptionsParsingException(
            "--" + BLAZE_ALIASING_FLAG + " only supports Starlark build settings.", cmdLineAlias);
      }

      return entry;
    }

    @Override
    public String getTypeDescription() {
      return "a 'name=value' flag alias";
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
    public Map.Entry<K, List<V>> convert(String input, @Nullable Object conversionContext)
        throws OptionsParsingException {
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
        convertedValues.add(valueConverter.convert(value, conversionContext));
      }
      return Maps.immutableEntry(
          keyConverter.convert(key, conversionContext), convertedValues.build());
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

    public Map.Entry<String, List<String>> convert(String input) throws OptionsParsingException {
      return convert(input, /* conversionContext= */ null);
    }

    @Override
    public String getTypeDescription() {
      return "a '[name=]value1[,..,valueN]' assignment";
    }
  }

  /** A request to set or unset a particular environment variable. */
  public sealed interface EnvVar {
    /** The name of the environment variable. */
    String name();

    /** Set the environment variable to the given value. */
    @AutoCodec
    record Set(String name, String value) implements EnvVar {}

    /** Inherit the value of the environment variable from the client environment. */
    @AutoCodec
    record Inherit(String name) implements EnvVar {}

    /**
     * Unset the environment variable, i.e., remove any previous assignment or even explicitly unset
     * it if implicitly inheriting the client environment.
     */
    @AutoCodec
    record Unset(String name) implements EnvVar {}
  }

  /**
   * A converter for variable assignments from the parameter list of a blaze command invocation.
   * Assignments are expected to have the form "name[=value]", where names and values are defined to
   * be as permissive as possible and value part can be optional (in which case it is considered to
   * be inherited). The special syntax "=name" is also supported and interpreted as a request to
   * unset the variable with the given name.
   */
  public static class EnvVarsConverter extends Converter.Contextless<EnvVar> {

    @Override
    public EnvVar convert(String input) throws OptionsParsingException {
      int pos = input.indexOf('=');
      if (input.isEmpty() || input.equals("=")) {
        throw new OptionsParsingException(
            "Variable definitions must be in the form of a 'name=value', 'name', or '=name'"
                + " assignment");
      } else if (pos == 0) {
        return new EnvVar.Unset(input.substring(1));
      } else if (pos < 0) {
        return new EnvVar.Inherit(input);
      }
      String name = input.substring(0, pos);
      String value = input.substring(pos + 1);
      return new EnvVar.Set(name, value);
    }

    @Override
    public boolean starlarkConvertible() {
      return true;
    }

    @Override
    public String reverseForStarlark(Object converted) {
      if (converted instanceof EnvVar.Set set) {
        return set.name() + "=" + set.value();
      } else if (converted instanceof EnvVar.Inherit inherit) {
        return inherit.name();
      } else if (converted instanceof EnvVar.Unset unset) {
        return "=" + unset.name();
      } else {
        throw new IllegalArgumentException(
            "EnvVarsConverter can only reverse EnvVar types, got: " + converted);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a 'name[=value]' assignment with an optional value part or the special syntax '=name'"
          + " to unset a variable";
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

  /** Same as {@link PercentageConverter} but also supports being unset. */
  public static class OptionalPercentageConverter extends Converter.Contextless<OptionalInt> {
    public static final String UNSET = "-1";
    private static final PercentageConverter PERCENTAGE_CONVERTER = new PercentageConverter();

    @Override
    public String getTypeDescription() {
      return "an integer";
    }

    @Override
    public OptionalInt convert(String input) throws OptionsParsingException {
      return input.equals(UNSET)
          ? OptionalInt.empty()
          : OptionalInt.of(PERCENTAGE_CONVERTER.convert(input));
    }
  }

  /**
   * A {@link Converter} for {@link com.github.benmanes.caffeine.cache.CaffeineSpec}. The spec may
   * be empty, in which case this converter returns null.
   */
  public static final class CaffeineSpecConverter extends Converter.Contextless<CaffeineSpec> {
    @Override
    public CaffeineSpec convert(String spec) throws OptionsParsingException {
      try {
        return CaffeineSpec.parse(spec);
      } catch (IllegalArgumentException e) {
        throw new OptionsParsingException("Failed to parse CaffeineSpec: " + e.getMessage(), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "Converts to a CaffeineSpec, or null if the input is empty";
    }
  }

  /** A {@link Converter} for a size in bytes with an optional multiplier suffix. */
  public static final class ByteSizeConverter extends Converter.Contextless<Long> {
    private static final Pattern PATTERN =
        Pattern.compile("(?<value>[0-9]+)(?<multiplier>[KMGT]?)");

    private static final ImmutableMap<String, Long> MULTIPLIER_MAP =
        ImmutableMap.of(
            "K",
            1024L,
            "M",
            1024L * 1024L,
            "G",
            1024L * 1024L * 1024L,
            "T",
            1024L * 1024L * 1024L * 1024L);

    @Override
    public Long convert(String input) throws OptionsParsingException {
      Matcher m = PATTERN.matcher(input);
      if (!m.matches()) {
        throw new OptionsParsingException("Invalid size: " + input);
      }
      try {
        long value = Long.parseLong(m.group("value"));
        String mult = m.group("multiplier");
        if (!mult.isEmpty()) {
          value = Math.multiplyExact(value, (long) MULTIPLIER_MAP.get(mult));
        }
        return value;
      } catch (NumberFormatException | ArithmeticException e) {
        throw new OptionsParsingException("Invalid size: " + input, e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a size in bytes, optionally followed by a K, M, G or T multiplier";
    }
  }
}
