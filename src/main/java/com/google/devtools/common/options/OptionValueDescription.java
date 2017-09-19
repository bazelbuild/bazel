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

package com.google.devtools.common.options;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The value of an option.
 *
 * <p>This takes care of tracking the final value as multiple instances of an option are parsed. It
 * also tracks additional metadata describing its priority, source, whether it was set via an
 * implicit dependency, and if so, by which other option.
 */
public abstract class OptionValueDescription {

  protected final OptionDefinition optionDefinition;

  public OptionValueDescription(OptionDefinition optionDefinition) {
    this.optionDefinition = optionDefinition;
  }

  public OptionDefinition getOptionDefinition() {
    return optionDefinition;
  }

  /** Returns the current or final value of this option. */
  public abstract Object getValue();

  /** Returns the source(s) of this option, if there were multiple, duplicates are removed. */
  public abstract String getSourceString();

  // TODO(b/65540004) implicitDependant and expandedFrom are artifacts of an option instance, and
  // should be in ParsedOptionDescription.
  abstract void addOptionInstance(
      ParsedOptionDescription parsedOption,
      OptionDefinition implicitDependant,
      OptionDefinition expandedFrom,
      List<String> warnings)
      throws OptionsParsingException;

  /**
   * For the given option, returns the correct type of OptionValueDescription, to which unparsed
   * values can be added.
   *
   * <p>The categories of option types are non-overlapping, an invariant checked by the
   * OptionProcessor at compile time.
   */
  public static OptionValueDescription createOptionValueDescription(OptionDefinition option) {
    if (option.allowsMultiple()) {
      return new RepeatableOptionValueDescription(option);
    } else if (option.isExpansionOption()) {
      return new ExpansionOptionValueDescription(option);
    } else if (option.getImplicitRequirements().length > 0) {
      return new OptionWithImplicitRequirementsValueDescription(option);
    } else if (option.isWrapperOption()) {
      return new WrapperOptionValueDescription(option);
    } else {
      return new SingleOptionValueDescription(option);
    }
  }

  /**
   * For options that have not been set, this will return a correct OptionValueDescription for the
   * default value.
   */
  public static OptionValueDescription getDefaultOptionValue(OptionDefinition option) {
    return new DefaultOptionValueDescription(option);
  }

  static class DefaultOptionValueDescription extends OptionValueDescription {

    private DefaultOptionValueDescription(OptionDefinition optionDefinition) {
      super(optionDefinition);
    }

    @Override
    public Object getValue() {
      return optionDefinition.getDefaultValue();
    }

    @Override
    public String getSourceString() {
      return null;
    }

    @Override
    void addOptionInstance(
        ParsedOptionDescription parsedOption,
        OptionDefinition implicitDependant,
        OptionDefinition expandedFrom,
        List<String> warnings) {
      throw new IllegalStateException(
          "Cannot add values to the default option value. Create a modifiable "
              + "OptionValueDescription using createOptionValueDescription() instead.");
    }
  }

  /**
   * The form of a value for a default type of flag, one that does not accumulate multiple values
   * and has no expansion.
   */
  static class SingleOptionValueDescription extends OptionValueDescription {
    private ParsedOptionDescription effectiveOptionInstance;
    private Object effectiveValue;
    private OptionDefinition optionThatDependsOnEffectiveValue;
    private OptionDefinition optionThatExpandedToEffectiveValue;

    private SingleOptionValueDescription(OptionDefinition optionDefinition) {
      super(optionDefinition);
      if (optionDefinition.allowsMultiple()) {
        throw new ConstructionException("Can't have a single value for an allowMultiple option.");
      }
      if (optionDefinition.isExpansionOption()) {
        throw new ConstructionException("Can't have a single value for an expansion option.");
      }
      effectiveOptionInstance = null;
      effectiveValue = null;
      optionThatDependsOnEffectiveValue = null;
      optionThatExpandedToEffectiveValue = null;
    }

    @Override
    public Object getValue() {
      return effectiveValue;
    }

    @Override
    public String getSourceString() {
      return effectiveOptionInstance.getSource();
    }

    // Warnings should not end with a '.' because the internal reporter adds one automatically.
    @Override
    void addOptionInstance(
        ParsedOptionDescription parsedOption,
        OptionDefinition implicitDependant,
        OptionDefinition expandedFrom,
        List<String> warnings)
        throws OptionsParsingException {
      // This might be the first value, in that case, just store it!
      if (effectiveOptionInstance == null) {
        effectiveOptionInstance = parsedOption;
        optionThatDependsOnEffectiveValue = implicitDependant;
        optionThatExpandedToEffectiveValue = expandedFrom;
        effectiveValue = effectiveOptionInstance.getConvertedValue();
        return;
      }

      // If there was another value, check whether the new one will override it, and if so,
      // log warnings describing the change.
      if (parsedOption.getPriority().compareTo(effectiveOptionInstance.getPriority()) >= 0) {
        // Output warnings:
        if ((implicitDependant != null) && (optionThatDependsOnEffectiveValue != null)) {
          if (!implicitDependant.equals(optionThatDependsOnEffectiveValue)) {
            warnings.add(
                String.format(
                    "Option '%s' is implicitly defined by both option '%s' and option '%s'",
                    optionDefinition.getOptionName(),
                    optionThatDependsOnEffectiveValue.getOptionName(),
                    implicitDependant.getOptionName()));
          }
        } else if ((implicitDependant != null)
            && parsedOption.getPriority().equals(effectiveOptionInstance.getPriority())) {
          warnings.add(
              String.format(
                  "Option '%s' is implicitly defined by option '%s'; the implicitly set value "
                      + "overrides the previous one",
                  optionDefinition.getOptionName(), implicitDependant.getOptionName()));
        } else if (optionThatDependsOnEffectiveValue != null) {
          warnings.add(
              String.format(
                  "A new value for option '%s' overrides a previous implicit setting of that "
                      + "option by option '%s'",
                  optionDefinition.getOptionName(),
                  optionThatDependsOnEffectiveValue.getOptionName()));
        } else if ((parsedOption.getPriority() == effectiveOptionInstance.getPriority())
            && ((optionThatExpandedToEffectiveValue == null) && (expandedFrom != null))) {
          // Create a warning if an expansion option overrides an explicit option:
          warnings.add(
              String.format(
                  "The option '%s' was expanded and now overrides a previous explicitly specified "
                      + "option '%s'",
                  expandedFrom.getOptionName(), optionDefinition.getOptionName()));
        } else if ((optionThatExpandedToEffectiveValue != null) && (expandedFrom != null)) {
          warnings.add(
              String.format(
                  "The option '%s' was expanded to from both options '%s' and '%s'",
                  optionDefinition.getOptionName(),
                  optionThatExpandedToEffectiveValue.getOptionName(),
                  expandedFrom.getOptionName()));
        }

        // Record the new value:
        effectiveOptionInstance = parsedOption;
        optionThatDependsOnEffectiveValue = implicitDependant;
        optionThatExpandedToEffectiveValue = expandedFrom;
        effectiveValue = parsedOption.getConvertedValue();
      } else {
        // The new value does not override the old value, as it has lower priority.
        warnings.add(
            String.format(
                "The lower priority option '%s' does not override the previous value '%s'",
                parsedOption.getCommandLineForm(), effectiveOptionInstance.getCommandLineForm()));
      }
    }

    @VisibleForTesting
    ParsedOptionDescription getEffectiveOptionInstance() {
      return effectiveOptionInstance;
    }
  }

  /** The form of a value for an option that accumulates multiple values on the command line. */
  static class RepeatableOptionValueDescription extends OptionValueDescription {
    ListMultimap<OptionPriority, ParsedOptionDescription> parsedOptions;
    ListMultimap<OptionPriority, Object> optionValues;

    private RepeatableOptionValueDescription(OptionDefinition optionDefinition) {
      super(optionDefinition);
      if (!optionDefinition.allowsMultiple()) {
        throw new ConstructionException(
            "Can't have a repeated value for a non-allowMultiple option.");
      }
      parsedOptions = ArrayListMultimap.create();
      optionValues = ArrayListMultimap.create();
    }

    @Override
    public String getSourceString() {
      return parsedOptions
          .asMap()
          .values()
          .stream()
          .flatMap(Collection::stream)
          .map(ParsedOptionDescription::getSource)
          .distinct()
          .collect(Collectors.joining(", "));
    }

    @Override
    public List<Object> getValue() {
      // Sort the results by option priority and return them in a new list. The generic type of
      // the list is not known at runtime, so we can't use it here. It was already checked in
      // the constructor, so this is type-safe.
      List<Object> result = new ArrayList<>();
      for (OptionPriority priority : OptionPriority.values()) {
        // If there is no mapping for this key, this check avoids object creation (because
        // ListMultimap has to return a new object on get) and also an unnecessary addAll call.
        if (optionValues.containsKey(priority)) {
          result.addAll(optionValues.get(priority));
        }
      }
      return result;
    }

    @Override
    void addOptionInstance(
        ParsedOptionDescription parsedOption,
        OptionDefinition implicitDependant,
        OptionDefinition expandedFrom,
        List<String> warnings)
        throws OptionsParsingException {
      // For repeatable options, we allow flags that take both single values and multiple values,
      // potentially collapsing them down.
      Object convertedValue = parsedOption.getConvertedValue();
      if (convertedValue instanceof List<?>) {
        optionValues.putAll(parsedOption.getPriority(), (List<?>) convertedValue);
      } else {
        optionValues.put(parsedOption.getPriority(), convertedValue);
      }
    }
  }

  /**
   * The form of a value for an expansion option, one that does not have its own value but expands
   * in place to other options. This should be used for both flags with a static expansion defined
   * in {@link Option#expansion()} and flags with an {@link Option#expansionFunction()}.
   */
  static class ExpansionOptionValueDescription extends OptionValueDescription {

    private ExpansionOptionValueDescription(OptionDefinition optionDefinition) {
      super(optionDefinition);
      if (!optionDefinition.isExpansionOption()) {
        throw new ConstructionException(
            "Options without expansions can't be tracked using ExpansionOptionValueDescription");
      }
    }

    @Override
    public Object getValue() {
      return null;
    }

    @Override
    public String getSourceString() {
      return null;
    }

    @Override
    void addOptionInstance(
        ParsedOptionDescription parsedOption,
        OptionDefinition implicitDependant,
        OptionDefinition expandedFrom,
        List<String> warnings) {
      // TODO(b/65540004) Deal with expansion options here instead of in parse(), and track their
      // link to the options they expanded to to.
    }
  }

  /** The form of a value for a flag with implicit requirements. */
  static class OptionWithImplicitRequirementsValueDescription extends SingleOptionValueDescription {

    private OptionWithImplicitRequirementsValueDescription(OptionDefinition optionDefinition) {
      super(optionDefinition);
      if (optionDefinition.getImplicitRequirements().length == 0) {
        throw new ConstructionException(
            "Options without implicit requirements can't be tracked using "
                + "OptionWithImplicitRequirementsValueDescription");
      }
    }

    @Override
    void addOptionInstance(
        ParsedOptionDescription parsedOption,
        OptionDefinition implicitDependant,
        OptionDefinition expandedFrom,
        List<String> warnings)
        throws OptionsParsingException {
      // This is a valued flag, its value is handled the same way as a normal
      // SingleOptionValueDescription.
      super.addOptionInstance(parsedOption, implicitDependant, expandedFrom, warnings);

      // Now deal with the implicit requirements.
      // TODO(b/65540004) Deal with options with implicit requirements here instead of in parse(),
      // and track their link to the options they implicitly expanded to to.
    }
  }

  /** Form for options that contain other options in the value text to which they expand. */
  static final class WrapperOptionValueDescription extends OptionValueDescription {

    WrapperOptionValueDescription(OptionDefinition optionDefinition) {
      super(optionDefinition);
    }

    @Override
    public Object getValue() {
      return null;
    }

    @Override
    public String getSourceString() {
      return null;
    }

    @Override
    void addOptionInstance(
        ParsedOptionDescription parsedOption,
        OptionDefinition implicitDependant,
        OptionDefinition expandedFrom,
        List<String> warnings)
        throws OptionsParsingException {
      // TODO(b/65540004) Deal with options with implicit requirements here instead of in parse(),
      // and track their link to the options they implicitly expanded to to.
    }
  }
}


