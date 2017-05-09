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

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ListMultimap;
import com.google.common.escape.Escaper;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.nio.file.FileSystem;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A parser for options. Typical use case in a main method:
 *
 * <pre>
 * OptionsParser parser = OptionsParser.newOptionsParser(FooOptions.class, BarOptions.class);
 * parser.parseAndExitUponError(args);
 * FooOptions foo = parser.getOptions(FooOptions.class);
 * BarOptions bar = parser.getOptions(BarOptions.class);
 * List&lt;String&gt; otherArguments = parser.getResidue();
 * </pre>
 *
 * <p>FooOptions and BarOptions would be options specification classes, derived from OptionsBase,
 * that contain fields annotated with @Option(...).
 *
 * <p>Alternatively, rather than calling {@link #parseAndExitUponError(OptionPriority, String,
 * String[])}, client code may call {@link #parse(OptionPriority,String,List)}, and handle parser
 * exceptions usage messages themselves.
 *
 * <p>This options parsing implementation has (at least) one design flaw. It allows both '--foo=baz'
 * and '--foo baz' for all options except void, boolean and tristate options. For these, the 'baz'
 * in '--foo baz' is not treated as a parameter to the option, making it is impossible to switch
 * options between void/boolean/tristate and everything else without breaking backwards
 * compatibility.
 *
 * @see Options a simpler class which you can use if you only have one options specification class
 */
public class OptionsParser implements OptionsProvider {

  /**
   * An unchecked exception thrown when there is a problem constructing a parser, e.g. an error
   * while validating an {@link Option} field in one of its {@link OptionsBase} subclasses.
   *
   * <p>This exception is unchecked because it generally indicates an internal error affecting all
   * invocations of the program. I.e., any such error should be immediately obvious to the
   * developer. Although unchecked, we explicitly mark some methods as throwing it as a reminder in
   * the API.
   */
  public static class ConstructionException extends RuntimeException {
    public ConstructionException(String message) {
      super(message);
    }

    public ConstructionException(Throwable cause) {
      super(cause);
    }

    public ConstructionException(String message, Throwable cause) {
      super(message, cause);
    }
  }

  /**
   * A cache for the parsed options data. Both keys and values are immutable, so
   * this is always safe. Only access this field through the {@link
   * #getOptionsData} method for thread-safety! The cache is very unlikely to
   * grow to a significant amount of memory, because there's only a fixed set of
   * options classes on the classpath.
   */
  private static final Map<ImmutableList<Class<? extends OptionsBase>>, OptionsData> optionsData =
      new HashMap<>();

  /**
   * Returns {@link OpaqueOptionsData} suitable for passing along to {@link
   * #newOptionsParser(OpaqueOptionsData optionsData)}.
   *
   * <p>This is useful when you want to do the work of analyzing the given {@code optionsClasses}
   * exactly once, but you want to parse lots of different lists of strings (and thus need to
   * construct lots of different {@link OptionsParser} instances).
   */
  public static OpaqueOptionsData getOptionsData(
      List<Class<? extends OptionsBase>> optionsClasses) throws ConstructionException {
    return getOptionsDataInternal(optionsClasses);
  }

  /**
   * Returns the {@link OptionsData} associated with the given list of options classes.
   */
  static synchronized OptionsData getOptionsDataInternal(
      List<Class<? extends OptionsBase>> optionsClasses) throws ConstructionException {
    ImmutableList<Class<? extends OptionsBase>> immutableOptionsClasses =
        ImmutableList.copyOf(optionsClasses);
    OptionsData result = optionsData.get(immutableOptionsClasses);
    if (result == null) {
      try {
        result = OptionsData.from(immutableOptionsClasses);
      } catch (Exception e) {
        throw new ConstructionException(e.getMessage(), e);
      }
      optionsData.put(immutableOptionsClasses, result);
    }
    return result;
  }

  /**
   * Returns the {@link OptionsData} associated with the given options class.
   */
  static OptionsData getOptionsDataInternal(Class<? extends OptionsBase> optionsClass)
      throws ConstructionException {
    return getOptionsDataInternal(ImmutableList.<Class<? extends OptionsBase>>of(optionsClass));
  }

  /**
   * @see #newOptionsParser(Iterable)
   */
  public static OptionsParser newOptionsParser(Class<? extends OptionsBase> class1)
      throws ConstructionException {
    return newOptionsParser(ImmutableList.<Class<? extends OptionsBase>>of(class1));
  }

  /**
   * @see #newOptionsParser(Iterable)
   */
  public static OptionsParser newOptionsParser(Class<? extends OptionsBase> class1,
                                               Class<? extends OptionsBase> class2)
      throws ConstructionException {
    return newOptionsParser(ImmutableList.of(class1, class2));
  }

  /** Create a new {@link OptionsParser}. */
  public static OptionsParser newOptionsParser(
      Iterable<? extends Class<? extends OptionsBase>> optionsClasses)
      throws ConstructionException {
    return newOptionsParser(
        getOptionsDataInternal(ImmutableList.<Class<? extends OptionsBase>>copyOf(optionsClasses)));
  }

  /**
   * Create a new {@link OptionsParser}, using {@link OpaqueOptionsData} previously returned from
   * {@link #getOptionsData}.
   */
  public static OptionsParser newOptionsParser(OpaqueOptionsData optionsData) {
    return new OptionsParser((OptionsData) optionsData);
  }

  private final OptionsParserImpl impl;
  private final List<String> residue = new ArrayList<String>();
  private boolean allowResidue = true;

  OptionsParser(OptionsData optionsData) {
    impl = new OptionsParserImpl(optionsData);
  }

  /**
   * Indicates whether or not the parser will allow a non-empty residue; that
   * is, iff this value is true then a call to one of the {@code parse}
   * methods will throw {@link OptionsParsingException} unless
   * {@link #getResidue()} is empty after parsing.
   */
  public void setAllowResidue(boolean allowResidue) {
    this.allowResidue = allowResidue;
  }

  /**
   * Indicates whether or not the parser will allow long options with a
   * single-dash, instead of the usual double-dash, too, eg. -example instead of just --example.
   */
  public void setAllowSingleDashLongOptions(boolean allowSingleDashLongOptions) {
    this.impl.setAllowSingleDashLongOptions(allowSingleDashLongOptions);
  }

  /** Enables the Parser to handle params files loacted insinde the provided {@link FileSystem}. */
  public void enableParamsFileSupport(FileSystem fs) {
    this.impl.setArgsPreProcessor(new ParamsFilePreProcessor(fs));
  }

  public void parseAndExitUponError(String[] args) {
    parseAndExitUponError(OptionPriority.COMMAND_LINE, "unknown", args);
  }

  /**
   * A convenience function for use in main methods. Parses the command line
   * parameters, and exits upon error. Also, prints out the usage message
   * if "--help" appears anywhere within {@code args}.
   */
  public void parseAndExitUponError(OptionPriority priority, String source, String[] args) {
    for (String arg : args) {
      if (arg.equals("--help")) {
        System.out.println(describeOptions(Collections.<String, String>emptyMap(),
                                           HelpVerbosity.LONG));
        System.exit(0);
      }
    }
    try {
      parse(priority, source, Arrays.asList(args));
    } catch (OptionsParsingException e) {
      System.err.println("Error parsing command line: " + e.getMessage());
      System.err.println("Try --help.");
      System.exit(2);
    }
  }

  /**
   * The metadata about an option.
   */
  public static final class OptionDescription {

    private final String name;

    // For valued flags
    private final Object defaultValue;
    private final Converter<?> converter;
    private final boolean allowMultiple;

    private final ImmutableList<OptionValueDescription> expansions;
    private final ImmutableList<OptionValueDescription> implicitRequirements;

    OptionDescription(
        String name,
        Object defaultValue,
        Converter<?> converter,
        boolean allowMultiple,
        ImmutableList<OptionValueDescription> expansions,
        ImmutableList<OptionValueDescription> implicitRequirements) {
      this.name = name;
      this.defaultValue = defaultValue;
      this.converter = converter;
      this.allowMultiple = allowMultiple;
      this.expansions = expansions;
      this.implicitRequirements = implicitRequirements;
    }

    public String getName() {
      return name;
    }

    public Object getDefaultValue() {
      return defaultValue;
    }

    public Converter<?> getConverter() {
      return converter;
    }

    public boolean getAllowMultiple() {
      return allowMultiple;
    }

    public ImmutableList<OptionValueDescription> getImplicitRequirements() {
      return implicitRequirements;
    }

    public ImmutableList<OptionValueDescription> getExpansions() {
      return expansions;
    }
  }

  /**
   * The name and value of an option with additional metadata describing its
   * priority, source, whether it was set via an implicit dependency, and if so,
   * by which other option.
   */
  public static class OptionValueDescription {
    private final String name;
    @Nullable private final String originalValueString;
    @Nullable private final Object value;
    @Nullable private final OptionPriority priority;
    @Nullable private final String source;
    @Nullable private final String implicitDependant;
    @Nullable private final String expandedFrom;
    private final boolean allowMultiple;

    public OptionValueDescription(
        String name,
        @Nullable String originalValueString,
        @Nullable Object value,
        @Nullable OptionPriority priority,
        @Nullable String source,
        @Nullable String implicitDependant,
        @Nullable String expandedFrom,
        boolean allowMultiple) {
      this.name = name;
      this.originalValueString = originalValueString;
      this.value = value;
      this.priority = priority;
      this.source = source;
      this.implicitDependant = implicitDependant;
      this.expandedFrom = expandedFrom;
      this.allowMultiple = allowMultiple;
    }

    public String getName() {
      return name;
    }

    public String getOriginalValueString() {
      return originalValueString;
    }

    // Need to suppress unchecked warnings, because the "multiple occurrence"
    // options use unchecked ListMultimaps due to limitations of Java generics.
    @SuppressWarnings({"unchecked", "rawtypes"})
    public Object getValue() {
      if (allowMultiple) {
        // Sort the results by option priority and return them in a new list.
        // The generic type of the list is not known at runtime, so we can't
        // use it here. It was already checked in the constructor, so this is
        // type-safe.
        List result = new ArrayList<>();
        ListMultimap realValue = (ListMultimap) value;
        for (OptionPriority priority : OptionPriority.values()) {
          // If there is no mapping for this key, this check avoids object creation (because
          // ListMultimap has to return a new object on get) and also an unnecessary addAll call.
          if (realValue.containsKey(priority)) {
            result.addAll(realValue.get(priority));
          }
        }
        return result;
      }
      return value;
    }

    /**
     * @return the priority of the thing that set this value for this flag
     */
    public OptionPriority getPriority() {
      return priority;
    }

    /**
     * @return the thing that set this value for this flag
     */
    public String getSource() {
      return source;
    }

    public String getImplicitDependant() {
      return implicitDependant;
    }

    public boolean isImplicitDependency() {
      return implicitDependant != null;
    }

    public String getExpansionParent() {
      return expandedFrom;
    }

    public boolean isExpansion() {
      return expandedFrom != null;
    }

    public boolean getAllowMultiple() {
      return allowMultiple;
    }

    @Override
    public String toString() {
      StringBuilder result = new StringBuilder();
      result.append("option '").append(name).append("' ");
      result.append("set to '").append(value).append("' ");
      result.append("with priority ").append(priority);
      if (source != null) {
        result.append(" and source '").append(source).append("'");
      }
      if (implicitDependant != null) {
        result.append(" implicitly by ");
      }
      return result.toString();
    }

    // Need to suppress unchecked warnings, because the "multiple occurrence"
    // options use unchecked ListMultimaps due to limitations of Java generics.
    @SuppressWarnings({"unchecked", "rawtypes"})
    void addValue(OptionPriority addedPriority, Object addedValue) {
      Preconditions.checkState(allowMultiple);
      ListMultimap optionValueList = (ListMultimap) value;
      if (addedValue instanceof List<?>) {
        optionValueList.putAll(addedPriority, (List<?>) addedValue);
      } else {
        optionValueList.put(addedPriority, addedValue);
      }
    }
  }

  /**
   * The name and unparsed value of an option with additional metadata describing its
   * priority, source, whether it was set via an implicit dependency, and if so,
   * by which other option.
   *
   * <p>Note that the unparsed value and the source parameters can both be null.
   */
  public static class UnparsedOptionValueDescription {
    private final String name;
    private final Field field;
    private final String unparsedValue;
    private final OptionPriority priority;
    private final String source;
    private final boolean explicit;

    public UnparsedOptionValueDescription(String name, Field field, String unparsedValue,
        OptionPriority priority, String source, boolean explicit) {
      this.name = name;
      this.field = field;
      this.unparsedValue = unparsedValue;
      this.priority = priority;
      this.source = source;
      this.explicit = explicit;
    }

    public String getName() {
      return name;
    }

    Field getField() {
      return field;
    }

    public boolean isBooleanOption() {
      return field.getType().equals(boolean.class);
    }

    private OptionUsageRestrictions optionUsageRestrictions() {
      return field.getAnnotation(Option.class).optionUsageRestrictions();
    }

    public boolean isDocumented() {
      return optionUsageRestrictions() == OptionUsageRestrictions.DOCUMENTED;
    }

    public boolean isHidden() {
      return optionUsageRestrictions() == OptionUsageRestrictions.HIDDEN
          || optionUsageRestrictions() == OptionUsageRestrictions.INTERNAL;
    }

    boolean isExpansion() {
      return OptionsData.isExpansionOption(field.getAnnotation(Option.class));
    }

    boolean isImplicitRequirement() {
      Option option = field.getAnnotation(Option.class);
      return option.implicitRequirements().length > 0;
    }

    boolean allowMultiple() {
      Option option = field.getAnnotation(Option.class);
      return option.allowMultiple();
    }

    public String getUnparsedValue() {
      return unparsedValue;
    }

    OptionPriority getPriority() {
      return priority;
    }

    public String getSource() {
      return source;
    }

    public boolean isExplicit() {
      return explicit;
    }

    @Override
    public String toString() {
      StringBuilder result = new StringBuilder();
      result.append("option '").append(name).append("' ");
      result.append("set to '").append(unparsedValue).append("' ");
      result.append("with priority ").append(priority);
      if (source != null) {
        result.append(" and source '").append(source).append("'");
      }
      return result.toString();
    }
  }

  /**
   * The verbosity with which option help messages are displayed: short (just
   * the name), medium (name, type, default, abbreviation), and long (full
   * description).
   */
  public enum HelpVerbosity { LONG, MEDIUM, SHORT }

  /**
   * The restrictions on an option. Only documented options are output as part of the help and are
   * intended for general user use. Undocumented options can be used by any user but aren't
   * advertised and in practice should be used by bazel developers or early adopters helping to test
   * a feature.
   *
   * <p>We use HIDDEN so that options that form the protocol between the client and the server are
   * not logged. These are flags, but should never be set by a user.
   *
   * <p>Options which are INTERNAL are not recognized by the parser at all, and so cannot be used as
   * flags.
   */
  public enum OptionUsageRestrictions {
    DOCUMENTED, UNDOCUMENTED, HIDDEN, INTERNAL
  }

  /**
   * Returns a description of all the options this parser can digest. In addition to {@link Option}
   * annotations, this method also interprets {@link OptionsUsage} annotations which give an
   * intuitive short description for the options. Options of the same category (see {@link
   * Option#category}) will be grouped together.
   *
   * @param categoryDescriptions a mapping from category names to category descriptions.
   *     Descriptions are optional; if omitted, a string based on the category name will be used.
   * @param helpVerbosity if {@code long}, the options will be described verbosely, including their
   *     types, defaults and descriptions. If {@code medium}, the descriptions are omitted, and if
   *     {@code short}, the options are just enumerated.
   */
  public String describeOptions(
      Map<String, String> categoryDescriptions, HelpVerbosity helpVerbosity) {
    OptionsData data = impl.getOptionsData();
    StringBuilder desc = new StringBuilder();
    if (!data.getOptionsClasses().isEmpty()) {
      List<Field> allFields = new ArrayList<>();
      for (Class<? extends OptionsBase> optionsClass : data.getOptionsClasses()) {
        allFields.addAll(data.getFieldsForClass(optionsClass));
      }
      Collections.sort(allFields, OptionsUsage.BY_CATEGORY);
      String prevCategory = null;

      for (Field optionField : allFields) {
        Option option = optionField.getAnnotation(Option.class);
        String category = option.category();
        if (!category.equals(prevCategory)
            && option.optionUsageRestrictions() == OptionUsageRestrictions.DOCUMENTED) {
          String description = categoryDescriptions.get(category);
          if (description == null) {
            description = "Options category '" + category + "'";
          }
          desc.append("\n").append(description).append(":\n");
          prevCategory = category;
        }

        if (option.optionUsageRestrictions() == OptionUsageRestrictions.DOCUMENTED) {
          OptionsUsage.getUsage(optionField, desc, helpVerbosity, impl.getOptionsData());
        }
      }
    }
    return desc.toString().trim();
  }

  /**
   * Returns a description of all the options this parser can digest.
   * In addition to {@link Option} annotations, this method also
   * interprets {@link OptionsUsage} annotations which give an intuitive short
   * description for the options.
   *
   * @param categoryDescriptions a mapping from category names to category
   *   descriptions.  Options of the same category (see {@link
   *   Option#category}) will be grouped together, preceded by the description
   *   of the category.
   */
  public String describeOptionsHtml(Map<String, String> categoryDescriptions, Escaper escaper) {
    OptionsData data = impl.getOptionsData();
    StringBuilder desc = new StringBuilder();
    if (!data.getOptionsClasses().isEmpty()) {
      List<Field> allFields = new ArrayList<>();
      for (Class<? extends OptionsBase> optionsClass : data.getOptionsClasses()) {
        allFields.addAll(data.getFieldsForClass(optionsClass));
      }
      Collections.sort(allFields, OptionsUsage.BY_CATEGORY);
      String prevCategory = null;

      for (Field optionField : allFields) {
        Option option = optionField.getAnnotation(Option.class);
        String category = option.category();
        if (!category.equals(prevCategory)
            && option.optionUsageRestrictions() == OptionUsageRestrictions.DOCUMENTED) {
          String description = categoryDescriptions.get(category);
          if (description == null) {
            description = "Options category '" + category + "'";
          }
          if (prevCategory != null) {
            desc.append("</dl>\n\n");
          }
          desc.append(escaper.escape(description)).append(":\n");
          desc.append("<dl>");
          prevCategory = category;
        }

        if (option.optionUsageRestrictions() == OptionUsageRestrictions.DOCUMENTED) {
          OptionsUsage.getUsageHtml(optionField, desc, escaper, impl.getOptionsData());
        }
      }
      desc.append("</dl>\n");
    }
    return desc.toString();
  }

  /**
   * Returns a string listing the possible flag completion for this command along with the command
   * completion if any. See {@link OptionsUsage#getCompletion(Field, StringBuilder)} for more
   * details on the format for the flag completion.
   */
  public String getOptionsCompletion() {
    OptionsData data = impl.getOptionsData();
    StringBuilder desc = new StringBuilder();

    // List all options
    List<Field> allFields = new ArrayList<>();
    for (Class<? extends OptionsBase> optionsClass : data.getOptionsClasses()) {
      allFields.addAll(data.getFieldsForClass(optionsClass));
    }
    // Sort field for deterministic ordering
    Collections.sort(allFields, new Comparator<Field>() {
      @Override
      public int compare(Field f1, Field f2) {
        String name1 = f1.getAnnotation(Option.class).name();
        String name2 = f2.getAnnotation(Option.class).name();
        return name1.compareTo(name2);
      }
    });
    for (Field optionField : allFields) {
      Option option = optionField.getAnnotation(Option.class);
      if (option.optionUsageRestrictions() == OptionUsageRestrictions.DOCUMENTED) {
        OptionsUsage.getCompletion(optionField, desc);
      }
    }

    return desc.toString();
  }

  /**
   * Returns a description of the option.
   *
   * @return The {@link OptionDescription} for the option, or null if there is no option by the
   *     given name.
   */
  public OptionDescription getOptionDescription(String name) throws OptionsParsingException {
    return impl.getOptionDescription(name);
  }

  /**
   * Returns a description of the option value set by the last previous call to
   * {@link #parse(OptionPriority, String, List)} that successfully set the given
   * option. If the option is of type {@link List}, the description will
   * correspond to any one of the calls, but not necessarily the last.
   *
   * @return The {@link OptionValueDescription} for the option, or null if the value has not been
   *        set.
   * @throws IllegalArgumentException if there is no option by the given name.
   */
  public OptionValueDescription getOptionValueDescription(String name) {
    return impl.getOptionValueDescription(name);
  }

  /**
   * A convenience method, equivalent to
   * {@code parse(OptionPriority.COMMAND_LINE, null, Arrays.asList(args))}.
   */
  public void parse(String... args) throws OptionsParsingException {
    parse(OptionPriority.COMMAND_LINE, null, Arrays.asList(args));
  }

  /**
   * A convenience method, equivalent to
   * {@code parse(OptionPriority.COMMAND_LINE, null, args)}.
   */
  public void parse(List<String> args) throws OptionsParsingException {
    parse(OptionPriority.COMMAND_LINE, null, args);
  }

  /**
   * Parses {@code args}, using the classes registered with this parser.
   * {@link #getOptions(Class)} and {@link #getResidue()} return the results.
   * May be called multiple times; later options override existing ones if they
   * have equal or higher priority. The source of options is a free-form string
   * that can be used for debugging. Strings that cannot be parsed as options
   * accumulates as residue, if this parser allows it.
   *
   * @see OptionPriority
   */
  public void parse(OptionPriority priority, String source,
      List<String> args) throws OptionsParsingException {
    parseWithSourceFunction(priority, Functions.constant(source), args);
  }

  /**
   * Parses {@code args}, using the classes registered with this parser.
   * {@link #getOptions(Class)} and {@link #getResidue()} return the results. May be called
   * multiple times; later options override existing ones if they have equal or higher priority.
   * The source of options is given as a function that maps option names to the source of the
   * option. Strings that cannot be parsed as options accumulates as* residue, if this parser
   * allows it.
   */
  public void parseWithSourceFunction(OptionPriority priority,
      Function<? super String, String> sourceFunction, List<String> args)
      throws OptionsParsingException {
    Preconditions.checkNotNull(priority);
    Preconditions.checkArgument(priority != OptionPriority.DEFAULT);
    residue.addAll(impl.parse(priority, sourceFunction, args));
    if (!allowResidue && !residue.isEmpty()) {
      String errorMsg = "Unrecognized arguments: " + Joiner.on(' ').join(residue);
      throw new OptionsParsingException(errorMsg);
    }
  }

  /**
   * Clears the given option.
   *
   * <p>This will not affect options objects that have already been retrieved from this parser
   * through {@link #getOptions(Class)}.
   *
   * @param optionName The full name of the option to clear.
   * @return A map of an option name to the old value of the options that were cleared.
   * @throws IllegalArgumentException If the flag does not exist.
   */
  public OptionValueDescription clearValue(String optionName)
      throws OptionsParsingException {
    OptionValueDescription clearedValue = impl.clearValue(optionName);
    return clearedValue;
  }

  @Override
  public List<String> getResidue() {
    return ImmutableList.copyOf(residue);
  }

  /**
   * Returns a list of warnings about problems encountered by previous parse calls.
   */
  public List<String> getWarnings() {
    return impl.getWarnings();
  }

  @Override
  public <O extends OptionsBase> O getOptions(Class<O> optionsClass) {
    return impl.getParsedOptions(optionsClass);
  }

  @Override
  public boolean containsExplicitOption(String name) {
    return impl.containsExplicitOption(name);
  }

  @Override
  public List<UnparsedOptionValueDescription> asListOfUnparsedOptions() {
    return impl.asListOfUnparsedOptions();
  }

  @Override
  public List<UnparsedOptionValueDescription> asListOfExplicitOptions() {
    return impl.asListOfExplicitOptions();
  }

  @Override
  public List<OptionValueDescription> asListOfEffectiveOptions() {
    return impl.asListOfEffectiveOptions();
  }

  @Override
  public List<String> canonicalize() {
    return impl.asCanonicalizedList();
  }

  /** Returns all options fields of the given options class, in alphabetic order. */
  public static Collection<Field> getFields(Class<? extends OptionsBase> optionsClass) {
    OptionsData data = OptionsParser.getOptionsDataInternal(optionsClass);
    return data.getFieldsForClass(optionsClass);
  }

  /**
   * Returns whether the given options class uses only the core types listed in {@link
   * OptionsBase#coreTypes}. These are guaranteed to be deeply immutable and serializable.
   */
  public static boolean getUsesOnlyCoreTypes(Class<? extends OptionsBase> optionsClass) {
    OptionsData data = OptionsParser.getOptionsDataInternal(optionsClass);
    return data.getUsesOnlyCoreTypes(optionsClass);
  }

  /**
   * Returns a mapping from each option {@link Field} in {@code optionsClass} (including inherited
   * ones) to its value in {@code options}.
   *
   * <p>The map is a mutable copy; changing the map won't affect {@code options} and vice versa.
   * The map entries appear sorted alphabetically by option name.
   *
   * If {@code options} is an instance of a subclass of {@code optionsClass}, any options defined
   * by the subclass are not included in the map.
   *
   * @throws IllegalArgumentException if {@code options} is not an instance of {@code optionsClass}
   */
  public static <O extends OptionsBase> Map<Field, Object> toMap(Class<O> optionsClass, O options) {
    OptionsData data = getOptionsDataInternal(optionsClass);
    // Alphabetized due to getFieldsForClass()'s order.
    Map<Field, Object> map = new LinkedHashMap<>();
    for (Field field : data.getFieldsForClass(optionsClass)) {
      try {
        map.put(field, field.get(options));
      } catch (IllegalAccessException e) {
        // All options fields of options classes should be public.
        throw new IllegalStateException(e);
      } catch (IllegalArgumentException e) {
        // This would indicate an inconsistency in the cached OptionsData.
        throw new IllegalStateException(e);
      }
    }
    return map;
  }

  /**
   * Given a mapping as returned by {@link #toMap}, and the options class it that its entries
   * correspond to, this constructs the corresponding instance of the options class.
   *
   * @throws IllegalArgumentException if {@code map} does not contain exactly the fields of {@code
   *     optionsClass}, with values of the appropriate type
   */
  public static <O extends OptionsBase> O fromMap(Class<O> optionsClass, Map<Field, Object> map) {
    // Instantiate the options class.
    OptionsData data = getOptionsDataInternal(optionsClass);
    O optionsInstance;
    try {
      Constructor<O> constructor = data.getConstructor(optionsClass);
      Preconditions.checkNotNull(constructor, "No options class constructor available");
      optionsInstance = constructor.newInstance();
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("Error while instantiating options class", e);
    }

    List<Field> fields = data.getFieldsForClass(optionsClass);
    // Ensure all fields are covered, no extraneous fields.
    validateFieldsSets(new LinkedHashSet<>(fields), new LinkedHashSet<>(map.keySet()));
    // Populate the instance.
    for (Field field : fields) {
      // Non-null as per above check.
      Object value = map.get(field);
      try {
        field.set(optionsInstance, value);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      }
      // May also throw IllegalArgumentException if map value is ill typed.
    }
    return optionsInstance;
  }

  /**
   * Raises a pretty {@link IllegalArgumentException} if the two sets of fields are not equal.
   *
   * <p>The entries in {@code fieldsFromMap} may be ill formed by being null or lacking an {@link
   * Option} annotation. (This isn't done for {@code fieldsFromClass} because they come from an
   * {@link OptionsData} object.)
   */
  private static void validateFieldsSets(
      LinkedHashSet<Field> fieldsFromClass, LinkedHashSet<Field> fieldsFromMap) {
    if (!fieldsFromClass.equals(fieldsFromMap)) {
      List<String> extraNamesFromClass = new ArrayList<>();
      List<String> extraNamesFromMap = new ArrayList<>();
      for (Field field : fieldsFromClass) {
        if (!fieldsFromMap.contains(field)) {
          extraNamesFromClass.add("'" + field.getAnnotation(Option.class).name() + "'");
        }
      }
      for (Field field : fieldsFromMap) {
        // Extra validation on the map keys since they don't come from OptionsData.
        if (!fieldsFromClass.contains(field)) {
          if (field == null) {
            extraNamesFromMap.add("<null field>");
          } else {
            Option annotation = field.getAnnotation(Option.class);
            if (annotation == null) {
              extraNamesFromMap.add("<non-Option field>");
            } else {
              extraNamesFromMap.add("'" + annotation.name() + "'");
            }
          }
        }
      }
      throw new IllegalArgumentException(
          "Map keys do not match fields of options class; extra map keys: {"
          + Joiner.on(", ").join(extraNamesFromMap) + "}; extra options class options: {"
          + Joiner.on(", ").join(extraNamesFromClass) + "}");
    }
  }
}
