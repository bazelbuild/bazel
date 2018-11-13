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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ListMultimap;
import com.google.common.escape.Escaper;
import com.google.devtools.common.options.OptionDefinition.NotAnOptionException;
import com.google.devtools.common.options.OptionsParserImpl.ResidueAndPriority;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

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
 * <p>Alternatively, rather than calling {@link
 * #parseAndExitUponError(OptionPriority.PriorityCategory, String, String[])}, client code may call
 * {@link #parse(OptionPriority.PriorityCategory,String,List)}, and handle parser exceptions usage
 * messages themselves.
 *
 * <p>This options parsing implementation has (at least) one design flaw. It allows both '--foo=baz'
 * and '--foo baz' for all options except void, boolean and tristate options. For these, the 'baz'
 * in '--foo baz' is not treated as a parameter to the option, making it is impossible to switch
 * options between void/boolean/tristate and everything else without breaking backwards
 * compatibility.
 *
 * @see Options a simpler class which you can use if you only have one options specification class
 */
public class OptionsParser implements OptionsParsingResult {

  // TODO(b/65049598) make ConstructionException checked.
  /**
   * An unchecked exception thrown when there is a problem constructing a parser, e.g. an error
   * while validating an {@link OptionDefinition} in one of its {@link OptionsBase} subclasses.
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
        Throwables.throwIfInstanceOf(e, ConstructionException.class);
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
    return getOptionsDataInternal(ImmutableList.of(optionsClass));
  }

  /**
   * @see #newOptionsParser(Iterable)
   */
  public static OptionsParser newOptionsParser(Class<? extends OptionsBase> class1)
      throws ConstructionException {
    return newOptionsParser(ImmutableList.<Class<? extends OptionsBase>>of(class1));
  }

  /** @see #newOptionsParser(Iterable) */
  public static OptionsParser newOptionsParser(
      Class<? extends OptionsBase> class1, Class<? extends OptionsBase> class2)
      throws ConstructionException {
    return newOptionsParser(ImmutableList.of(class1, class2));
  }

  /** Create a new {@link OptionsParser}. */
  public static OptionsParser newOptionsParser(
      Iterable<? extends Class<? extends OptionsBase>> optionsClasses)
      throws ConstructionException {
    return newOptionsParser(getOptionsDataInternal(ImmutableList.copyOf(optionsClasses)));
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
  private final List<String> postDoubleDashResidue = new ArrayList<>();
  private boolean allowResidue = true;
  private Map<String, Object> skylarkOptions = new HashMap<>();

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

  @Override
  public Map<String, Object> getSkylarkOptions() {
    return skylarkOptions;
  }

  @VisibleForTesting
  public void setSkylarkOptionsForTesting(Map<String, Object> skylarkOptions) {
    this.skylarkOptions = skylarkOptions;
  }

  /**
   * Indicates whether or not the parser will allow long options with a
   * single-dash, instead of the usual double-dash, too, eg. -example instead of just --example.
   */
  public void setAllowSingleDashLongOptions(boolean allowSingleDashLongOptions) {
    this.impl.setAllowSingleDashLongOptions(allowSingleDashLongOptions);
  }

  /**
   * Enables the Parser to handle params files using the provided {@link ParamsFilePreProcessor}.
   */
  public void enableParamsFileSupport(ParamsFilePreProcessor preProcessor) {
    this.impl.setArgsPreProcessor(preProcessor);
  }

  public void parseAndExitUponError(String[] args) {
    parseAndExitUponError(OptionPriority.PriorityCategory.COMMAND_LINE, "unknown", args);
  }

  /**
   * A convenience function for use in main methods. Parses the command line parameters, and exits
   * upon error. Also, prints out the usage message if "--help" appears anywhere within {@code
   * args}.
   */
  public void parseAndExitUponError(
      OptionPriority.PriorityCategory priority, String source, String[] args) {
    for (String arg : args) {
      if (arg.equals("--help")) {
        System.out.println(
            describeOptionsWithDeprecatedCategories(ImmutableMap.of(), HelpVerbosity.LONG));

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

  /** The metadata about an option, in the context of this options parser. */
  public static final class OptionDescription {
    private final OptionDefinition optionDefinition;
    private final ImmutableList<String> evaluatedExpansion;

    OptionDescription(OptionDefinition definition, OptionsData optionsData) {
      this.optionDefinition = definition;
      this.evaluatedExpansion = optionsData.getEvaluatedExpansion(optionDefinition);
    }

    public OptionDefinition getOptionDefinition() {
      return optionDefinition;
    }

    public boolean isExpansion() {
      return optionDefinition.isExpansionOption();
    }

    /** Return a list of flags that this option expands to. */
    public ImmutableList<String> getExpansion() throws OptionsParsingException {
      return evaluatedExpansion;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof OptionDescription) {
        OptionDescription other = (OptionDescription) obj;
        // Check that the option is the same, with the same expansion.
        return other.optionDefinition.equals(optionDefinition)
            && other.evaluatedExpansion.equals(evaluatedExpansion);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return optionDefinition.hashCode() + evaluatedExpansion.hashCode();
    }
  }

  /**
   * The verbosity with which option help messages are displayed: short (just
   * the name), medium (name, type, default, abbreviation), and long (full
   * description).
   */
  public enum HelpVerbosity { LONG, MEDIUM, SHORT }

  /**
   * Returns a description of all the options this parser can digest. In addition to {@link Option}
   * annotations, this method also interprets {@link OptionsUsage} annotations which give an
   * intuitive short description for the options. Options of the same category (see {@link
   * OptionDocumentationCategory}) will be grouped together.
   *
   * @param productName the name of this product (blaze, bazel)
   * @param helpVerbosity if {@code long}, the options will be described verbosely, including their
   *     types, defaults and descriptions. If {@code medium}, the descriptions are omitted, and if
   *     {@code short}, the options are just enumerated.
   */
  public String describeOptions(String productName, HelpVerbosity helpVerbosity) {
    StringBuilder desc = new StringBuilder();
    LinkedHashMap<OptionDocumentationCategory, List<OptionDefinition>> optionsByCategory =
        getOptionsSortedByCategory();
    ImmutableMap<OptionDocumentationCategory, String> optionCategoryDescriptions =
        OptionFilterDescriptions.getOptionCategoriesEnumDescription(productName);
    for (Map.Entry<OptionDocumentationCategory, List<OptionDefinition>> e :
        optionsByCategory.entrySet()) {
      String categoryDescription = optionCategoryDescriptions.get(e.getKey());
      List<OptionDefinition> categorizedOptionList = e.getValue();

      // Describe the category if we're going to end up using it at all.
      if (!categorizedOptionList.isEmpty()) {
        desc.append("\n").append(categoryDescription).append(":\n");
      }
      // Describe the options in this category.
      for (OptionDefinition optionDef : categorizedOptionList) {
        OptionsUsage.getUsage(optionDef, desc, helpVerbosity, impl.getOptionsData(), true);
      }
    }

    return desc.toString().trim();
  }

  /**
   * @return all documented options loaded in this parser, grouped by categories in display order.
   */
  private LinkedHashMap<OptionDocumentationCategory, List<OptionDefinition>>
      getOptionsSortedByCategory() {
    OptionsData data = impl.getOptionsData();
    if (data.getOptionsClasses().isEmpty()) {
      return new LinkedHashMap<>();
    }

    // Get the documented options grouped by category.
    ListMultimap<OptionDocumentationCategory, OptionDefinition> optionsByCategories =
        ArrayListMultimap.create();
    for (Class<? extends OptionsBase> optionsClass : data.getOptionsClasses()) {
      for (OptionDefinition optionDefinition :
          OptionsData.getAllOptionDefinitionsForClass(optionsClass)) {
        // Only track documented options.
        if (optionDefinition.getDocumentationCategory()
            != OptionDocumentationCategory.UNDOCUMENTED) {
          optionsByCategories.put(optionDefinition.getDocumentationCategory(), optionDefinition);
        }
      }
    }

    // Put the categories into display order and sort the options in each category.
    LinkedHashMap<OptionDocumentationCategory, List<OptionDefinition>> sortedCategoriesToOptions =
        new LinkedHashMap<>(OptionFilterDescriptions.documentationOrder.length, 1);
    for (OptionDocumentationCategory category : OptionFilterDescriptions.documentationOrder) {
      List<OptionDefinition> optionList = optionsByCategories.get(category);
      if (optionList != null) {
        optionList.sort(OptionDefinition.BY_OPTION_NAME);
        sortedCategoriesToOptions.put(category, optionList);
      }
    }
    return sortedCategoriesToOptions;
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
  @Deprecated
  public String describeOptionsWithDeprecatedCategories(
      Map<String, String> categoryDescriptions, HelpVerbosity helpVerbosity) {
    OptionsData data = impl.getOptionsData();
    StringBuilder desc = new StringBuilder();
    if (!data.getOptionsClasses().isEmpty()) {
      List<OptionDefinition> allFields = new ArrayList<>();
      for (Class<? extends OptionsBase> optionsClass : data.getOptionsClasses()) {
        allFields.addAll(OptionsData.getAllOptionDefinitionsForClass(optionsClass));
      }
      Collections.sort(allFields, OptionDefinition.BY_CATEGORY);
      String prevCategory = null;

      for (OptionDefinition optionDefinition : allFields) {
        String category = optionDefinition.getOptionCategory();
        if (!category.equals(prevCategory)
            && optionDefinition.getDocumentationCategory()
                != OptionDocumentationCategory.UNDOCUMENTED) {
          String description = categoryDescriptions.get(category);
          if (description == null) {
            description = "Options category '" + category + "'";
          }
          desc.append("\n").append(description).append(":\n");
          prevCategory = category;
        }

        if (optionDefinition.getDocumentationCategory()
            != OptionDocumentationCategory.UNDOCUMENTED) {
          OptionsUsage.getUsage(
              optionDefinition, desc, helpVerbosity, impl.getOptionsData(), false);
        }
      }
    }
    return desc.toString().trim();
  }

  /**
   * Returns a description of all the options this parser can digest. In addition to {@link Option}
   * annotations, this method also interprets {@link OptionsUsage} annotations which give an
   * intuitive short description for the options.
   *
   * @param categoryDescriptions a mapping from category names to category descriptions. Options of
   *     the same category (see {@link Option#category}) will be grouped together, preceded by the
   *     description of the category.
   */
  @Deprecated
  public String describeOptionsHtmlWithDeprecatedCategories(
      Map<String, String> categoryDescriptions, Escaper escaper) {
    OptionsData data = impl.getOptionsData();
    StringBuilder desc = new StringBuilder();
    if (!data.getOptionsClasses().isEmpty()) {
      List<OptionDefinition> allFields = new ArrayList<>();
      for (Class<? extends OptionsBase> optionsClass : data.getOptionsClasses()) {
        allFields.addAll(OptionsData.getAllOptionDefinitionsForClass(optionsClass));
      }
      Collections.sort(allFields, OptionDefinition.BY_CATEGORY);
      String prevCategory = null;

      for (OptionDefinition optionDefinition : allFields) {
        String category = optionDefinition.getOptionCategory();
        if (!category.equals(prevCategory)
            && optionDefinition.getDocumentationCategory()
                != OptionDocumentationCategory.UNDOCUMENTED) {
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

        if (optionDefinition.getDocumentationCategory()
            != OptionDocumentationCategory.UNDOCUMENTED) {
          OptionsUsage.getUsageHtml(optionDefinition, desc, escaper, impl.getOptionsData(), false);
        }
      }
      desc.append("</dl>\n");
    }
    return desc.toString();
  }

  /**
   * Returns a description of all the options this parser can digest. In addition to {@link Option}
   * annotations, this method also interprets {@link OptionsUsage} annotations which give an
   * intuitive short description for the options.
   */
  public String describeOptionsHtml(Escaper escaper, String productName) {
    StringBuilder desc = new StringBuilder();
    LinkedHashMap<OptionDocumentationCategory, List<OptionDefinition>> optionsByCategory =
        getOptionsSortedByCategory();
    ImmutableMap<OptionDocumentationCategory, String> optionCategoryDescriptions =
        OptionFilterDescriptions.getOptionCategoriesEnumDescription(productName);

    for (Map.Entry<OptionDocumentationCategory, List<OptionDefinition>> e :
        optionsByCategory.entrySet()) {
      desc.append("<dl>");
      String categoryDescription = optionCategoryDescriptions.get(e.getKey());
      List<OptionDefinition> categorizedOptionsList = e.getValue();

      // Describe the category if we're going to end up using it at all.
      if (!categorizedOptionsList.isEmpty()) {
        desc.append(escaper.escape(categoryDescription)).append(":\n");
      }
      // Describe the options in this category.
      for (OptionDefinition optionDef : categorizedOptionsList) {
        OptionsUsage.getUsageHtml(optionDef, desc, escaper, impl.getOptionsData(), true);
      }
      desc.append("</dl>\n");
    }
    return desc.toString();
  }

  /**
   * Returns a string listing the possible flag completion for this command along with the command
   * completion if any. See {@link OptionsUsage#getCompletion(OptionDefinition, StringBuilder)} for
   * more details on the format for the flag completion.
   */
  public String getOptionsCompletion() {
    StringBuilder desc = new StringBuilder();

    visitOptions(
        optionDefinition ->
            optionDefinition.getDocumentationCategory() != OptionDocumentationCategory.UNDOCUMENTED,
        optionDefinition -> OptionsUsage.getCompletion(optionDefinition, desc));

    return desc.toString();
  }

  public void visitOptions(
      Predicate<OptionDefinition> predicate, Consumer<OptionDefinition> visitor) {
    Preconditions.checkNotNull(predicate, "Missing predicate.");
    Preconditions.checkNotNull(visitor, "Missing visitor.");

    OptionsData data = impl.getOptionsData();
    data.getOptionsClasses()
        // List all options
        .stream()
        .flatMap(optionsClass -> OptionsData.getAllOptionDefinitionsForClass(optionsClass).stream())
        // Sort field for deterministic ordering
        .sorted(OptionDefinition.BY_OPTION_NAME)
        .filter(predicate)
        .forEach(visitor);
  }

  /**
   * Returns a description of the option.
   *
   * @return The {@link OptionDescription} for the option, or null if there is no option by the
   *     given name.
   */
  OptionDescription getOptionDescription(String name) throws OptionsParsingException {
    return impl.getOptionDescription(name);
  }

  /**
   * Returns the parsed options that get expanded from this option, whether it expands due to an
   * implicit requirement or expansion.
   *
   * @param expansionOption the option that might need to be expanded. If this option does not
   *     expand to other options, the empty list will be returned.
   * @param originOfExpansionOption the origin of the option that's being expanded. This function
   *     will take care of adjusting the source messages as necessary.
   */
  ImmutableList<ParsedOptionDescription> getExpansionValueDescriptions(
      OptionDefinition expansionOption, OptionInstanceOrigin originOfExpansionOption)
      throws OptionsParsingException {
    return impl.getExpansionValueDescriptions(expansionOption, originOfExpansionOption);
  }

  /**
   * Returns a description of the option value set by the last previous call to {@link
   * #parse(OptionPriority.PriorityCategory, String, List)} that successfully set the given option.
   * If the option is of type {@link List}, the description will correspond to any one of the calls,
   * but not necessarily the last.
   *
   * @return The {@link com.google.devtools.common.options.OptionValueDescription} for the option,
   *     or null if the value has not been set.
   * @throws IllegalArgumentException if there is no option by the given name.
   */
  public OptionValueDescription getOptionValueDescription(String name) {
    return impl.getOptionValueDescription(name);
  }

  /**
   * A convenience method, equivalent to {@code parse(PriorityCategory.COMMAND_LINE, null,
   * Arrays.asList(args))}.
   */
  public void parse(String... args) throws OptionsParsingException {
    parse(OptionPriority.PriorityCategory.COMMAND_LINE, null, Arrays.asList(args));
  }

  /**
   * A convenience method, equivalent to {@code parse(PriorityCategory.COMMAND_LINE, null, args)}.
   */
  public void parse(List<String> args) throws OptionsParsingException {
    parse(OptionPriority.PriorityCategory.COMMAND_LINE, null, args);
  }

  /**
   * Parses {@code args}, using the classes registered with this parser, at the given priority.
   *
   * <p>May be called multiple times; later options override existing ones if they have equal or
   * higher priority. Strings that cannot be parsed as options are accumulated as residue, if this
   * parser allows it.
   *
   * <p>{@link #getOptions(Class)} and {@link #getResidue()} will return the results.
   *
   * @param priority the priority at which to parse these options. Within this priority category,
   *     each option will be given an index to track its position. If parse() has already been
   *     called at this priority, the indexing will continue where it left off, to keep ordering.
   * @param source the source to track for each option parsed.
   * @param args the arg list to parse. Each element might be an option, a value linked to an
   *     option, or residue.
   */
  public void parse(OptionPriority.PriorityCategory priority, String source, List<String> args)
      throws OptionsParsingException {
    parseWithSourceFunction(priority, o -> source, args);
  }

  /**
   * Parses {@code args}, using the classes registered with this parser, at the given priority.
   *
   * <p>May be called multiple times; later options override existing ones if they have equal or
   * higher priority. Strings that cannot be parsed as options are accumulated as residue, if this
   * parser allows it.
   *
   * <p>{@link #getOptions(Class)} and {@link #getResidue()} will return the results.
   *
   * @param priority the priority at which to parse these options. Within this priority category,
   *     each option will be given an index to track its position. If parse() has already been
   *     called at this priority, the indexing will continue where it left off, to keep ordering.
   * @param sourceFunction a function that maps option names to the source of the option.
   * @param args the arg list to parse. Each element might be an option, a value linked to an
   *     option, or residue.
   */
  public void parseWithSourceFunction(
      OptionPriority.PriorityCategory priority,
      Function<OptionDefinition, String> sourceFunction,
      List<String> args)
      throws OptionsParsingException {
    Preconditions.checkNotNull(priority);
    Preconditions.checkArgument(priority != OptionPriority.PriorityCategory.DEFAULT);
    ResidueAndPriority residueAndPriority = impl.parse(priority, sourceFunction, args);
    residue.addAll(residueAndPriority.getResidue());
    postDoubleDashResidue.addAll(residueAndPriority.postDoubleDashResidue);
    if (!allowResidue && !residue.isEmpty()) {
      String errorMsg = "Unrecognized arguments: " + Joiner.on(' ').join(residue);
      throw new OptionsParsingException(errorMsg);
    }
  }

  /**
   * Parses the args at the priority of the provided option. This is useful for after-the-fact
   * expansion.
   *
   * @param optionToExpand the option that is being "expanded" after the fact. The provided args
   *     will have the same priority as this option.
   * @param source a description of where the expansion arguments came from.
   * @param args the arguments to parse as the expansion. Order matters, as the value of a flag may
   *     be in the following argument.
   */
  public void parseArgsAsExpansionOfOption(
      ParsedOptionDescription optionToExpand, String source, List<String> args)
      throws OptionsParsingException {
    Preconditions.checkNotNull(
        optionToExpand, "Option for expansion not specified for arglist " + args);
    Preconditions.checkArgument(
        optionToExpand.getPriority().getPriorityCategory()
            != OptionPriority.PriorityCategory.DEFAULT,
        "Priority cannot be default, which was specified for arglist " + args);
    ResidueAndPriority residueAndPriority =
        impl.parseArgsAsExpansionOfOption(optionToExpand, o -> source, args);
    residue.addAll(residueAndPriority.getResidue());
    postDoubleDashResidue.addAll(residueAndPriority.postDoubleDashResidue);
    if (!allowResidue && !residue.isEmpty()) {
      String errorMsg = "Unrecognized arguments: " + Joiner.on(' ').join(residue);
      throw new OptionsParsingException(errorMsg);
    }
  }

  /**
   * @param origin the origin of this option instance, it includes the priority of the value. If
   *     other values have already been or will be parsed at a higher priority, they might override
   *     the provided value. If this option already has a value at this priority, this value will
   *     have precedence, but this should be avoided, as it breaks order tracking.
   * @param option the option to add the value for.
   * @param value the value to add at the given priority.
   */
  void addOptionValueAtSpecificPriority(
      OptionInstanceOrigin origin, OptionDefinition option, String value)
      throws OptionsParsingException {
    impl.addOptionValueAtSpecificPriority(origin, option, value);
  }

  /**
   * Clears the given option.
   *
   * <p>This will not affect options objects that have already been retrieved from this parser
   * through {@link #getOptions(Class)}.
   *
   * @param option The option to clear.
   * @return The old value of the option that was cleared.
   * @throws IllegalArgumentException If the flag does not exist.
   */
  public OptionValueDescription clearValue(OptionDefinition option) throws OptionsParsingException {
    return impl.clearValue(option);
  }

  @Override
  public List<String> getResidue() {
    return ImmutableList.copyOf(residue);
  }

  @Override
  public List<String> getPreDoubleDashResidue() {
    return postDoubleDashResidue.isEmpty()
        ? ImmutableList.copyOf(residue)
        : residue.stream()
            .filter(residue -> !postDoubleDashResidue.contains(residue))
            .collect(Collectors.toList());
  }

  /** Returns a list of warnings about problems encountered by previous parse calls. */
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
  public List<ParsedOptionDescription> asCompleteListOfParsedOptions() {
    return impl.asCompleteListOfParsedOptions();
  }

  @Override
  public List<ParsedOptionDescription> asListOfExplicitOptions() {
    return impl.asListOfExplicitOptions();
  }

  @Override
  public List<ParsedOptionDescription> asListOfCanonicalOptions() {
    return impl.asCanonicalizedListOfParsedOptions();
  }

  @Override
  public List<OptionValueDescription> asListOfOptionValues() {
    return impl.asListOfEffectiveOptions();
  }

  @Override
  public List<String> canonicalize() {
    return impl.asCanonicalizedList();
  }

  /** Returns all options fields of the given options class, in alphabetic order. */
  public static ImmutableList<OptionDefinition> getOptionDefinitions(
      Class<? extends OptionsBase> optionsClass) {
    return OptionsData.getAllOptionDefinitionsForClass(optionsClass);
  }

  /**
   * Returns whether the given options class uses only the core types listed in {@link
   * UsesOnlyCoreTypes#CORE_TYPES}. These are guaranteed to be deeply immutable and serializable.
   */
  public static boolean getUsesOnlyCoreTypes(Class<? extends OptionsBase> optionsClass) {
    OptionsData data = OptionsParser.getOptionsDataInternal(optionsClass);
    return data.getUsesOnlyCoreTypes(optionsClass);
  }

  /**
   * Returns a mapping from each option {@link Field} in {@code optionsClass} (including inherited
   * ones) to its value in {@code options}.
   *
   * <p>To save space, the map directly stores {@code Fields} instead of the {@code
   * OptionDefinitions}.
   *
   * <p>The map is a mutable copy; changing the map won't affect {@code options} and vice versa. The
   * map entries appear sorted alphabetically by option name.
   *
   * <p>If {@code options} is an instance of a subclass of {@link OptionsBase}, any options defined
   * by the subclass are not included in the map, only the options declared in the provided class
   * are included.
   *
   * @throws IllegalArgumentException if {@code options} is not an instance of {@link OptionsBase}
   */
  public static <O extends OptionsBase> Map<Field, Object> toMap(Class<O> optionsClass, O options) {
    // Alphabetized due to getAllOptionDefinitionsForClass()'s order.
    Map<Field, Object> map = new LinkedHashMap<>();
    for (OptionDefinition optionDefinition :
        OptionsData.getAllOptionDefinitionsForClass(optionsClass)) {
      try {
        // Get the object value of the optionDefinition and place in map.
        map.put(optionDefinition.getField(), optionDefinition.getField().get(options));
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
   * @param map Field to Object, expecting an entry for each field in the optionsClass. This
   *     directly refers to the Field, without wrapping it in an OptionDefinition, see {@link
   *     #toMap}.
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

    List<OptionDefinition> optionDefinitions =
        OptionsData.getAllOptionDefinitionsForClass(optionsClass);
    // Ensure all fields are covered, no extraneous fields.
    validateFieldsSets(optionsClass, new LinkedHashSet<Field>(map.keySet()));
    // Populate the instance.
    for (OptionDefinition optionDefinition : optionDefinitions) {
      // Non-null as per above check.
      Object value = map.get(optionDefinition.getField());
      try {
        optionDefinition.getField().set(optionsInstance, value);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      }
      // May also throw IllegalArgumentException if map value is ill typed.
    }
    return optionsInstance;
  }

  /**
   * Raises a pretty {@link IllegalArgumentException} if the provided set of fields is a complete
   * set for the optionsClass.
   *
   * <p>The entries in {@code fieldsFromMap} may be ill formed by being null or lacking an {@link
   * Option} annotation.
   */
  private static void validateFieldsSets(
      Class<? extends OptionsBase> optionsClass, LinkedHashSet<Field> fieldsFromMap) {
    ImmutableList<OptionDefinition> optionDefsFromClasses =
        OptionsData.getAllOptionDefinitionsForClass(optionsClass);
    Set<Field> fieldsFromClass =
        optionDefsFromClasses.stream().map(OptionDefinition::getField).collect(Collectors.toSet());

    if (fieldsFromClass.equals(fieldsFromMap)) {
      // They are already equal, avoid additional checks.
      return;
    }

    List<String> extraNamesFromClass = new ArrayList<>();
    List<String> extraNamesFromMap = new ArrayList<>();
    for (OptionDefinition optionDefinition : optionDefsFromClasses) {
      if (!fieldsFromMap.contains(optionDefinition.getField())) {
        extraNamesFromClass.add("'" + optionDefinition.getOptionName() + "'");
      }
    }
    for (Field field : fieldsFromMap) {
      // Extra validation on the map keys since they don't come from OptionsData.
      if (!fieldsFromClass.contains(field)) {
        if (field == null) {
          extraNamesFromMap.add("<null field>");
        } else {
          OptionDefinition optionDefinition = null;
          try {
            // TODO(ccalvarin) This shouldn't be necessary, no option definitions should be found in
            // this optionsClass that weren't in the cache.
            optionDefinition = OptionDefinition.extractOptionDefinition(field);
            extraNamesFromMap.add("'" + optionDefinition.getOptionName() + "'");
          } catch (NotAnOptionException e) {
            extraNamesFromMap.add("<non-Option field>");
          }
        }
      }
    }
    throw new IllegalArgumentException(
        "Map keys do not match fields of options class; extra map keys: {"
            + Joiner.on(", ").join(extraNamesFromMap)
            + "}; extra options class options: {"
            + Joiner.on(", ").join(extraNamesFromClass)
            + "}");
  }
}
