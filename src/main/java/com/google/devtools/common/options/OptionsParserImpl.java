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

import static com.google.devtools.common.options.OptionPriority.PriorityCategory.INVOCATION_POLICY;
import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.toCollection;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterators;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionValueDescription.ExpansionBundle;
import com.google.devtools.common.options.OptionsParser.OptionDescription;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * The implementation of the options parser. This is intentionally package private for full
 * flexibility. Use {@link OptionsParser} or {@link Options} if you're a consumer.
 */
class OptionsParserImpl {

  /** Helper class to create a new instance of {@link OptionsParserImpl}. */
  static final class Builder {
    private OptionsData optionsData;
    private ArgsPreProcessor argsPreProcessor = args -> args;
    private final ArrayList<String> skippedPrefixes = new ArrayList<>();
    private boolean ignoreInternalOptions = true;
    @Nullable private String aliasFlag = null;
    private final Map<String, String> aliases = new HashMap<>();

    /** Set the {@link OptionsData} to be used in this instance. */
    public Builder optionsData(OptionsData optionsData) {
      this.optionsData = optionsData;
      return this;
    }

    /** Sets the {@link ArgsPreProcessor} to use during processing. */
    public Builder argsPreProcessor(ArgsPreProcessor preProcessor) {
      this.argsPreProcessor = preProcessor;
      return this;
    }

    /** Any flags with this prefix will be skipped during processing. */
    public Builder skippedPrefix(String skippedPrefix) {
      this.skippedPrefixes.add(skippedPrefix);
      return this;
    }

    /** Sets whether the parser should ignore internal-only options. */
    public Builder ignoreInternalOptions(boolean ignoreInternalOptions) {
      this.ignoreInternalOptions = ignoreInternalOptions;
      return this;
    }

    /**
     * Sets what flag the parser should use for flag aliasing. Defaults to null if not set,
     * effectively disabling the aliasing functionality.
     */
    public Builder withAliasFlag(@Nullable String aliasFlag) {
      if (aliasFlag != null) {
        this.aliasFlag = "--" + aliasFlag;
      }
      return this;
    }

    /**
     * Adds a map of flag aliases where the keys are the flags' alias names and the values are their
     * actual names.
     */
    public Builder withAliases(Map<String, String> aliases) {
      this.aliases.putAll(aliases);
      return this;
    }

    /** Returns a newly-initialized {@link OptionsParserImpl}. */
    public OptionsParserImpl build() {
      return new OptionsParserImpl(
          this.optionsData,
          this.argsPreProcessor,
          this.skippedPrefixes,
          this.ignoreInternalOptions,
          this.aliasFlag,
          this.aliases);
    }
  }

  /** Returns a new {@link Builder} with correct defaults applied. */
  public static Builder builder() {
    return new Builder();
  }

  private final OptionsData optionsData;

  /**
   * We store the results of option parsing in here - since there can only be one value per option
   * field, this is where the different instances of an option have been combined and the final
   * value is tracked. It'll look like
   *
   * <pre>
   *   OptionDefinition("--host") -> "www.google.com"
   *   OptionDefinition("--port") -> 80
   * </pre>
   *
   * This map is modified by repeated calls to {@link #parse(OptionPriority.PriorityCategory,
   * Function,List)}.
   */
  private final Map<OptionDefinition, OptionValueDescription> optionValues = new HashMap<>();

  /**
   * Since parse() expects multiple calls to it with the same {@link PriorityCategory} to be treated
   * as though the args in the later call have higher priority over the earlier calls, we need to
   * track the high water mark of option priority at each category. Each call to parse will start at
   * this level.
   */
  private final Map<PriorityCategory, OptionPriority> nextPriorityPerPriorityCategory =
      Arrays.stream(PriorityCategory.values())
          .collect(Collectors.toMap(p -> p, OptionPriority::lowestOptionPriorityAtCategory));

  /**
   * Explicit option tracking, tracking each option as it was provided, after they have been parsed.
   *
   * <p>The value is unconverted, still the string as it was read from the input, or partially
   * altered in cases where the flag was set by non {@code --flag=value} forms; e.g. {@code --nofoo}
   * becomes {@code --foo=0}.
   */
  private final List<ParsedOptionDescription> parsedOptions = new ArrayList<>();

  private final Map<String, String> flagAliasMappings;
  // We want to keep the invariant that warnings are produced as they are encountered, but only
  // show each one once.
  private final Set<String> warnings = new LinkedHashSet<>();
  private final ArgsPreProcessor argsPreProcessor;
  private final List<String> skippedPrefixes;
  private final boolean ignoreInternalOptions;
  @Nullable private final String aliasFlag;

  OptionsParserImpl(
      OptionsData optionsData,
      ArgsPreProcessor argsPreProcessor,
      List<String> skippedPrefixes,
      boolean ignoreInternalOptions,
      @Nullable String aliasFlag,
      Map<String, String> aliases) {
    this.optionsData = optionsData;
    this.argsPreProcessor = argsPreProcessor;
    this.skippedPrefixes = skippedPrefixes;
    this.ignoreInternalOptions = ignoreInternalOptions;
    this.aliasFlag = aliasFlag;
    this.flagAliasMappings = aliases;
  }

  /** Returns the {@link OptionsData} used in this instance. */
  OptionsData getOptionsData() {
    return optionsData;
  }

  /** Returns a {@link Builder} that is configured the same as this parser. */
  Builder toBuilder() {
    Builder builder = builder().optionsData(optionsData).argsPreProcessor(argsPreProcessor);
    for (String skippedPrefix : skippedPrefixes) {
      builder.skippedPrefix(skippedPrefix);
    }
    return builder;
  }

  /** Implements {@link OptionsParser#asCompleteListOfParsedOptions()}. */
  List<ParsedOptionDescription> asCompleteListOfParsedOptions() {
    return parsedOptions.stream()
        // It is vital that this sort is stable so that options on the same priority are not
        // reordered.
        .sorted(comparing(ParsedOptionDescription::getPriority))
        .collect(toCollection(ArrayList::new));
  }

  /** Implements {@link OptionsParser#asListOfExplicitOptions()}. */
  List<ParsedOptionDescription> asListOfExplicitOptions() {
    return parsedOptions.stream()
        .filter(ParsedOptionDescription::isExplicit)
        // It is vital that this sort is stable so that options on the same priority are not
        // reordered.
        .sorted(comparing(ParsedOptionDescription::getPriority))
        .collect(toCollection(ArrayList::new));
  }

  /** Implements {@link OptionsParser#canonicalize}. */
  List<String> asCanonicalizedList() {
    return asCanonicalizedListOfParsedOptions().stream()
        .map(ParsedOptionDescription::getDeprecatedCanonicalForm)
        .collect(ImmutableList.toImmutableList());
  }

  /** Implements {@link OptionsParser#canonicalize}. */
  List<ParsedOptionDescription> asCanonicalizedListOfParsedOptions() {
    return optionValues.keySet().stream()
        .map(optionDefinition -> optionValues.get(optionDefinition).getCanonicalInstances())
        .flatMap(Collection::stream)
        // Return the effective (canonical) options in the order they were applied.
        .sorted(comparing(ParsedOptionDescription::getPriority))
        .collect(ImmutableList.toImmutableList());
  }

  /** Implements {@link OptionsParser#asListOfOptionValues()}. */
  List<OptionValueDescription> asListOfEffectiveOptions() {
    List<OptionValueDescription> result = new ArrayList<>();
    for (Map.Entry<String, OptionDefinition> mapEntry : optionsData.getAllOptionDefinitions()) {
      OptionDefinition optionDefinition = mapEntry.getValue();
      OptionValueDescription optionValue = optionValues.get(optionDefinition);
      if (optionValue == null) {
        result.add(OptionValueDescription.getDefaultOptionValue(optionDefinition));
      } else {
        result.add(optionValue);
      }
    }
    return result;
  }

  private void maybeAddDeprecationWarning(
      OptionDefinition optionDefinition, PriorityCategory priority) {
    // Don't add a warning for deprecated flag set by the invocation policy.
    if (priority.equals(INVOCATION_POLICY)) {
      return;
    }
    // Continue to support the old behavior for @Deprecated options.
    String warning = optionDefinition.getDeprecationWarning();
    if (!warning.isEmpty() || optionDefinition.getField().isAnnotationPresent(Deprecated.class)) {
      addDeprecationWarning(optionDefinition.getOptionName(), warning);
    }
  }

  private void maybeAddOldNameWarning(ParsedOptionDescription parsedOption) {
    // Don't add a warning for old name options set by the invocation policy.
    if (parsedOption.getPriority().getPriorityCategory().equals(INVOCATION_POLICY)) {
      return;
    }
    String commandLineForm = parsedOption.getCommandLineForm();
    String oldOptionName = parsedOption.getOptionDefinition().getOldOptionName();
    String optionName = parsedOption.getOptionDefinition().getOptionName();
    if (commandLineForm.startsWith(String.format("--%s=", oldOptionName))) {
      addDeprecationWarning(oldOptionName, String.format("Use --%s instead", optionName));
    }
  }

  private void addDeprecationWarning(String optionName, String warning) {
    warnings.add(
        String.format(
            "Option '%s' is deprecated%s", optionName, (warning.isEmpty() ? "" : ": " + warning)));
  }

  OptionValueDescription clearValue(OptionDefinition optionDefinition)
      throws OptionsParsingException {
    return optionValues.remove(optionDefinition);
  }

  OptionValueDescription getOptionValueDescription(String name) {
    OptionDefinition optionDefinition = optionsData.getOptionDefinitionFromName(name);
    if (optionDefinition == null) {
      throw new IllegalArgumentException("No such option '" + name + "'");
    }
    return optionValues.get(optionDefinition);
  }

  OptionDescription getOptionDescription(String name) throws OptionsParsingException {
    OptionDefinition optionDefinition = optionsData.getOptionDefinitionFromName(name);
    if (optionDefinition == null) {
      return null;
    }
    return new OptionDescription(optionDefinition, optionsData);
  }

  /**
   * Implementation of {@link OptionsParser#getExpansionValueDescriptions(OptionDefinition,
   * OptionInstanceOrigin)}
   */
  ImmutableList<ParsedOptionDescription> getExpansionValueDescriptions(
      OptionDefinition expansionFlagDef, OptionInstanceOrigin originOfExpansionFlag)
      throws OptionsParsingException {
    ImmutableList.Builder<ParsedOptionDescription> builder = ImmutableList.builder();

    // Values needed to correctly track the origin of the expanded options.
    OptionPriority nextOptionPriority =
        OptionPriority.getChildPriority(originOfExpansionFlag.getPriority());
    String source;
    ParsedOptionDescription implicitDependent = null;
    ParsedOptionDescription expandedFrom = null;

    ImmutableList<String> options;
    ParsedOptionDescription expansionFlagParsedDummy =
        ParsedOptionDescription.newDummyInstance(expansionFlagDef, originOfExpansionFlag);
    if (expansionFlagDef.hasImplicitRequirements()) {
      options = ImmutableList.copyOf(expansionFlagDef.getImplicitRequirements());
      source =
          String.format(
              "implicitly required by %s (source: %s)",
              expansionFlagDef, originOfExpansionFlag.getSource());
      implicitDependent = expansionFlagParsedDummy;
    } else if (expansionFlagDef.isExpansionOption()) {
      options = optionsData.getEvaluatedExpansion(expansionFlagDef);
      source =
          String.format(
              "expanded by %s (source: %s)", expansionFlagDef, originOfExpansionFlag.getSource());
      expandedFrom = expansionFlagParsedDummy;
    } else {
      return ImmutableList.of();
    }

    Iterator<String> optionsIterator = options.iterator();
    while (optionsIterator.hasNext()) {
      String unparsedFlagExpression = optionsIterator.next();
      ParsedOptionDescription parsedOption =
          identifyOptionAndPossibleArgument(
              unparsedFlagExpression,
              optionsIterator,
              nextOptionPriority,
              o -> source,
              implicitDependent,
              expandedFrom);
      builder.add(parsedOption);
      nextOptionPriority = OptionPriority.nextOptionPriority(nextOptionPriority);
    }
    return builder.build();
  }

  boolean containsExplicitOption(String name) {
    OptionDefinition optionDefinition = optionsData.getOptionDefinitionFromName(name);
    if (optionDefinition == null) {
      throw new IllegalArgumentException("No such option '" + name + "'");
    }
    return optionValues.get(optionDefinition) != null;
  }

  /**
   * Parses the args, and returns what it doesn't parse. May be called multiple times, and may be
   * called recursively. The option's definition dictates how it reacts to multiple settings. By
   * default, the arg seen last at the highest priority takes precedence, overriding the early
   * values. Options that accumulate multiple values will track them in priority and appearance
   * order.
   */
  OptionsParserImplResult parse(
      PriorityCategory priorityCat,
      Function<OptionDefinition, String> sourceFunction,
      List<String> args)
      throws OptionsParsingException {
    OptionsParserImplResult optionsParserImplResult =
        parse(nextPriorityPerPriorityCategory.get(priorityCat), sourceFunction, null, null, args);
    nextPriorityPerPriorityCategory.put(priorityCat, optionsParserImplResult.nextPriority);
    return optionsParserImplResult;
  }

  /**
   * Parses the args, and returns what it doesn't parse. May be called multiple times, and may be
   * called recursively. Calls may contain intersecting sets of options; in that case, the arg seen
   * last takes precedence.
   *
   * <p>The method treats options that have neither an implicitDependent nor an expandedFrom value
   * as explicitly set.
   */
  private OptionsParserImplResult parse(
      OptionPriority priority,
      Function<OptionDefinition, String> sourceFunction,
      ParsedOptionDescription implicitDependent,
      ParsedOptionDescription expandedFrom,
      List<String> args)
      throws OptionsParsingException {
    List<String> unparsedArgs = new ArrayList<>();
    List<String> unparsedPostDoubleDashArgs = new ArrayList<>();

    Iterator<String> argsIterator = argsPreProcessor.preProcess(args).iterator();
    while (argsIterator.hasNext()) {
      String arg = argsIterator.next();

      if (!arg.startsWith("-")) {
        unparsedArgs.add(arg);
        continue; // not an option arg
      }

      arg = swapShorthandAlias(arg);

      if (containsSkippedPrefix(arg)) {
        unparsedArgs.add(arg);
        continue;
      }

      if (arg.equals("--")) { // "--" means all remaining args aren't options
        Iterators.addAll(unparsedPostDoubleDashArgs, argsIterator);
        break;
      }

      ParsedOptionDescription parsedOption =
          identifyOptionAndPossibleArgument(
              arg, argsIterator, priority, sourceFunction, implicitDependent, expandedFrom);
      handleNewParsedOption(parsedOption);
      priority = OptionPriority.nextOptionPriority(priority);
    }

    // Go through the final values and make sure they are valid values for their option. Unlike any
    // checks that happened above, this also checks that flags that were not set have a valid
    // default value. getValue() will throw if the value is invalid.
    for (OptionValueDescription valueDescription : asListOfEffectiveOptions()) {
      valueDescription.getValue();
    }

    return new OptionsParserImplResult(
        unparsedArgs, unparsedPostDoubleDashArgs, priority, flagAliasMappings);
  }

  /** A class that stores residue and priority information. */
  static final class OptionsParserImplResult {
    final List<String> postDoubleDashResidue;
    final List<String> preDoubleDashResidue;
    final OptionPriority nextPriority;
    final ImmutableMap<String, String> aliases;

    OptionsParserImplResult(
        List<String> preDashResidue,
        List<String> postDashResidue,
        OptionPriority nextPriority,
        Map<String, String> aliases) {
      this.preDoubleDashResidue = preDashResidue;
      this.postDoubleDashResidue = postDashResidue;
      this.nextPriority = nextPriority;
      this.aliases = ImmutableMap.copyOf(aliases);
    }

    public List<String> getResidue() {
      List<String> toReturn =
          new ArrayList<>(preDoubleDashResidue.size() + postDoubleDashResidue.size());
      toReturn.addAll(preDoubleDashResidue);
      toReturn.addAll(postDoubleDashResidue);
      return toReturn;
    }
  }

  /** Implements {@link OptionsParser#parseArgsAsExpansionOfOption} */
  OptionsParserImplResult parseArgsAsExpansionOfOption(
      ParsedOptionDescription optionToExpand,
      Function<OptionDefinition, String> sourceFunction,
      List<String> args)
      throws OptionsParsingException {
    return parse(
        OptionPriority.getChildPriority(optionToExpand.getPriority()),
        sourceFunction,
        null,
        optionToExpand,
        args);
  }

  /**
   * Implementation of {@link OptionsParser#addOptionValueAtSpecificPriority(OptionInstanceOrigin,
   * OptionDefinition, String)}
   */
  void addOptionValueAtSpecificPriority(
      OptionInstanceOrigin origin, OptionDefinition option, String unconvertedValue)
      throws OptionsParsingException {
    Preconditions.checkNotNull(option);
    Preconditions.checkNotNull(
        unconvertedValue,
        "Cannot set %s to a null value. Pass \"\" if an empty value is required.",
        option);
    Preconditions.checkNotNull(
        origin,
        "Cannot assign value \'%s\' to %s without a clear origin for this value.",
        unconvertedValue,
        option);
    PriorityCategory priorityCategory = origin.getPriority().getPriorityCategory();
    boolean isNotDefault = priorityCategory != OptionPriority.PriorityCategory.DEFAULT;
    Preconditions.checkArgument(
        isNotDefault,
        "Attempt to assign value \'%s\' to %s at priority %s failed. Cannot set options at "
            + "default priority - by definition, that means the option is unset.",
        unconvertedValue,
        option,
        priorityCategory);

    handleNewParsedOption(
        ParsedOptionDescription.newParsedOptionDescription(
            option,
            String.format("--%s=%s", option.getOptionName(), unconvertedValue),
            unconvertedValue,
            origin));
  }

  /** Takes care of tracking the parsed option's value in relation to other options. */
  private void handleNewParsedOption(ParsedOptionDescription parsedOption)
      throws OptionsParsingException {
    OptionDefinition optionDefinition = parsedOption.getOptionDefinition();
    // All options can be deprecated; check and warn before doing any option-type specific work.
    maybeAddDeprecationWarning(optionDefinition, parsedOption.getPriority().getPriorityCategory());
    // Check if the old option name is used and add a warning
    maybeAddOldNameWarning(parsedOption);
    // Track the value, before any remaining option-type specific work that is done outside of
    // the OptionValueDescription.
    OptionValueDescription entry =
        optionValues.computeIfAbsent(
            optionDefinition,
            def -> OptionValueDescription.createOptionValueDescription(def, optionsData));
    ExpansionBundle expansionBundle = entry.addOptionInstance(parsedOption, warnings);
    @Nullable String unconvertedValue = parsedOption.getUnconvertedValue();

    // There are 3 types of flags that expand to other flag values. Expansion flags are the
    // accepted way to do this, but implicit requirements also do this. We rely on the
    // OptionProcessor compile-time check's guarantee that no option sets
    // both expansion behaviors. (In Bazel, --config is another such flag, but that expansion
    // is not controlled within the options parser, so we ignore it here)

    // As much as possible, we want the behaviors of these different types of flags to be
    // identical, as this minimizes the number of edge cases, but we do not yet track these values
    // in the same way.
    if (parsedOption.getImplicitDependent() == null) {
      // Log explicit options and expanded options in the order they are parsed (can be sorted
      // later). This information is needed to correctly canonicalize flags.
      parsedOptions.add(parsedOption);

      if (aliasFlag != null && parsedOption.getCommandLineForm().startsWith(aliasFlag)) {
        List<String> alias =
            Splitter.on('=').limit(2).splitToList(parsedOption.getUnconvertedValue());

        flagAliasMappings.put(alias.get(0), alias.get(1));
      }
    }

    if (expansionBundle != null) {
      OptionsParserImplResult optionsParserImplResult =
          parse(
              OptionPriority.getChildPriority(parsedOption.getPriority()),
              o -> expansionBundle.sourceOfExpansionArgs,
              optionDefinition.hasImplicitRequirements() ? parsedOption : null,
              optionDefinition.isExpansionOption() ? parsedOption : null,
              expansionBundle.expansionArgs);
      if (!optionsParserImplResult.getResidue().isEmpty()) {

        // Throw an assertion here, because this indicates an error in the definition of this
        // option's expansion or requirements, not with the input as provided by the user.
        throw new AssertionError(
            "Unparsed options remain after processing "
                + unconvertedValue
                + ": "
                + Joiner.on(' ').join(optionsParserImplResult.getResidue()));
      }
    }
  }

  private ParsedOptionDescription identifyOptionAndPossibleArgument(
      String arg,
      Iterator<String> nextArgs,
      OptionPriority priority,
      Function<OptionDefinition, String> sourceFunction,
      ParsedOptionDescription implicitDependent,
      ParsedOptionDescription expandedFrom)
      throws OptionsParsingException {

    // Store the way this option was parsed on the command line.
    StringBuilder commandLineForm = new StringBuilder();
    commandLineForm.append(arg);
    String unconvertedValue = null;
    OptionDefinition optionDefinition;
    boolean booleanValue = true;

    if (arg.length() == 2) { // -l  (may be nullary or unary)
      optionDefinition = optionsData.getFieldForAbbrev(arg.charAt(1));
      booleanValue = true;

    } else if (arg.length() == 3 && arg.charAt(2) == '-') { // -l-  (boolean)
      optionDefinition = optionsData.getFieldForAbbrev(arg.charAt(1));
      booleanValue = false;

    } else if (arg.startsWith("--")) { // --long_option

      int equalsAt = arg.indexOf('=');
      int nameStartsAt = arg.startsWith("--") ? 2 : 1;
      String name =
          equalsAt == -1 ? arg.substring(nameStartsAt) : arg.substring(nameStartsAt, equalsAt);
      if (name.trim().isEmpty()) {
        throw new OptionsParsingException("Invalid options syntax: " + arg, arg);
      }
      unconvertedValue = equalsAt == -1 ? null : arg.substring(equalsAt + 1);
      optionDefinition = optionsData.getOptionDefinitionFromName(name);

      // Look for a "no"-prefixed option name: "no<optionName>".
      if (optionDefinition == null && name.startsWith("no")) {
        name = name.substring(2);
        optionDefinition = optionsData.getOptionDefinitionFromName(name);
        booleanValue = false;
        if (optionDefinition != null) {
          // TODO(bazel-team): Add tests for these cases.
          if (!optionDefinition.usesBooleanValueSyntax()) {
            throw new OptionsParsingException(
                "Illegal use of 'no' prefix on non-boolean option: " + arg, arg);
          }
          if (unconvertedValue != null) {
            throw new OptionsParsingException("Unexpected value after boolean option: " + arg, arg);
          }
          // "no<optionname>" signifies a boolean option w/ false value
          unconvertedValue = "0";
        }
      }
    } else {
      throw new OptionsParsingException("Invalid options syntax: " + arg, arg);
    }

    if (optionDefinition == null || shouldIgnoreOption(optionDefinition)) {
      // Do not recognize internal options, which are treated as if they did not exist.
      throw new OptionsParsingException("Unrecognized option: " + arg, arg);
    }

    if (unconvertedValue == null) {
      // Special-case boolean to supply value based on presence of "no" prefix.
      if (optionDefinition.usesBooleanValueSyntax()) {
        unconvertedValue = booleanValue ? "1" : "0";
      } else if (optionDefinition.getType().equals(Void.class)) {
        // This is expected, Void type options have no args.
      } else if (nextArgs.hasNext()) {
        // "--flag value" form
        unconvertedValue = nextArgs.next();
        commandLineForm.append(" ").append(unconvertedValue);
      } else {
        throw new OptionsParsingException("Expected value after " + arg);
      }
    }

    return ParsedOptionDescription.newParsedOptionDescription(
        optionDefinition,
        commandLineForm.toString(),
        unconvertedValue,
        new OptionInstanceOrigin(
            priority, sourceFunction.apply(optionDefinition), implicitDependent, expandedFrom));
  }

  private boolean shouldIgnoreOption(OptionDefinition optionDefinition) {
    return ignoreInternalOptions
        && ImmutableList.copyOf(optionDefinition.getOptionMetadataTags())
            .contains(OptionMetadataTag.INTERNAL);
  }

  /** Gets the result of parsing the options. */
  <O extends OptionsBase> O getParsedOptions(Class<O> optionsClass) {
    // Create the instance:
    O optionsInstance;
    try {
      Constructor<O> constructor = optionsData.getConstructor(optionsClass);
      if (constructor == null) {
        return null;
      }
      optionsInstance = constructor.newInstance();
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("Error while instantiating options class", e);
    }

    // Set the fields
    for (OptionDefinition optionDefinition :
        OptionsData.getAllOptionDefinitionsForClass(optionsClass)) {
      Object value;
      OptionValueDescription optionValue = optionValues.get(optionDefinition);
      if (optionValue == null) {
        value = optionDefinition.getDefaultValue();
      } else {
        value = optionValue.getValue();
      }
      try {
        optionDefinition.getField().set(optionsInstance, value);
      } catch (IllegalArgumentException e) {
        // May happen when a boolean option got a string value. Just ignore this error without
        // updating the field. Fixes https://github.com/bazelbuild/bazel/issues/7847
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(
            "Could not set the field due to access issues. This is impossible, as the "
                + "OptionProcessor checks that all options are non-final public fields.",
            e);
      }
    }
    return optionsInstance;
  }

  ImmutableList<String> getWarnings() {
    return ImmutableList.copyOf(warnings);
  }

  /**
   * Takes a string with a leading "-" and swaps it with the matching alias mapping. Example case
   * with --flag_alias=foo=bar mapped:
   *
   * <pre>
   *   swapShorthandAlias("-c") returns "-c"
   *   swapShorthandAlias("--foo") returns "--bar"
   *   swapShorthandAlias("--baz") returns "--baz"
   * </pre>
   *
   * This method returns immediately when aliasFlag is not set via the builder, which is an implicit
   * disabling of the aliasing functionality.
   */
  private String swapShorthandAlias(String arg) {
    if (aliasFlag == null || !arg.startsWith("--")) {
      return arg;
    }

    int equalSign = arg.indexOf("=");

    // Extracts the <arg> from '--<arg>=<value>' and '--<arg> <value>' formats on the command line
    String actualArg = (equalSign != -1) ? arg.substring(2, equalSign) : arg.substring(2);

    if (!flagAliasMappings.containsKey(actualArg)) {
      return arg;
    }

    String alias = flagAliasMappings.get(actualArg);
    actualArg = alias;

    // Converts the arg back into a command line option, accounting for both '--<arg>=<value>' and
    // '--<arg> <value>' formats
    actualArg = (equalSign != -1) ? "--" + actualArg + arg.substring(equalSign) : "--" + actualArg;

    return actualArg;
  }

  private boolean containsSkippedPrefix(String arg) {
    return skippedPrefixes.stream().anyMatch(arg::startsWith);
  }
}
