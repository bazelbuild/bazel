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

package com.google.devtools.build.lib.analysis.skylark;

import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute.SplitTransitionProvider;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.lang.reflect.Field;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.Map;

/**
 * This class implements a split transition provider that takes a Skylark transition function as
 * input.  The transition function takes a settings argument, which is a dictionary containing the
 * current option values.  It either returns a dictionary mapping option name to new option value
 * (for a patch transition), or a dictionary of such dictionaries (for a split transition).
 *
 * Currently the implementation ignores the attributes provided by the containing function.
 */
public class FunctionSplitTransitionProvider implements SplitTransitionProvider {

  private static final String COMMAND_LINE_OPTION_PREFIX = "//command_line_option:";

  private final BaseFunction transitionFunction;
  private final SkylarkSemantics semantics;
  private final EventHandler eventHandler;

  public FunctionSplitTransitionProvider(BaseFunction transitionFunction,
      SkylarkSemantics semantics, EventHandler eventHandler) {
    this.transitionFunction = transitionFunction;
    this.semantics = semantics;
    this.eventHandler = eventHandler;
  }

  @Override
  public SplitTransition apply(AttributeMap attributeMap) {
    return new FunctionSplitTransition(transitionFunction, semantics, eventHandler);
  }

  private static class FunctionSplitTransition implements SplitTransition {
    private final BaseFunction transitionFunction;
    private final SkylarkSemantics semantics;
    private final EventHandler eventHandler;

    public FunctionSplitTransition(BaseFunction transitionFunction, SkylarkSemantics semantics,
        EventHandler eventHandler) {
      this.transitionFunction = transitionFunction;
      this.semantics = semantics;
      this.eventHandler = eventHandler;
    }

    @Override
    public final List<BuildOptions> split(BuildOptions buildOptions) {
      // TODO(waltl): we should be able to build this once and use it across different split
      // transitions.
      Map<String, OptionInfo> optionInfoMap = buildOptionInfo(buildOptions);
      SkylarkDict<String, Object> settings = buildSettings(buildOptions, optionInfoMap);

      ImmutableList.Builder<BuildOptions> splitBuildOptions = ImmutableList.builder();

      try {
        ImmutableList<Map<String, Object>> transitions =
            evalTransitionFunction(transitionFunction, settings);

        for (Map<String, Object> transition : transitions) {
          BuildOptions options = buildOptions.clone();
          applyTransition(options, transition, optionInfoMap);
          splitBuildOptions.add(options);
        }
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      } catch (EvalException e) {
        throw new RuntimeException(e.print());
      }

      return splitBuildOptions.build();
    }

    /**
     * Given a label-like string representing a command line option, returns the command line
     * option string that it represents. This is a temporary measure to support command line
     * options with strings that look "label-like", so that migrating users using this
     * experimental syntax is easier later.
     *
     * @throws EvalException if the given string is not a valid format to represent to
     *     a command line option
     */
    private String commandLineOptionLabelToOption(String label) throws EvalException {
      if (label.startsWith(COMMAND_LINE_OPTION_PREFIX)) {
        return label.substring(COMMAND_LINE_OPTION_PREFIX.length());
      } else {
        throw new EvalException(transitionFunction.getLocation(),
            String.format("Option key '%s' is of invalid form. "
                   + "Expected command line option to begin with %s",
                label, COMMAND_LINE_OPTION_PREFIX));
      }
    }

    /**
     * For all the options in the BuildOptions, build a map from option name to its information.
     */
    private Map<String, OptionInfo> buildOptionInfo(BuildOptions buildOptions) {
      ImmutableMap.Builder<String, OptionInfo> builder = new ImmutableMap.Builder<>();

      ImmutableSet<Class<? extends FragmentOptions>> optionClasses =
          buildOptions
          .getNativeOptions()
          .stream()
          .map(FragmentOptions::getClass)
          .collect(ImmutableSet.toImmutableSet());

      for (Class<? extends FragmentOptions> optionClass : optionClasses) {
        ImmutableList<OptionDefinition> optionDefinitions =
            OptionsParser.getOptionDefinitions(optionClass);
        for (OptionDefinition def : optionDefinitions) {
          String optionName = def.getOptionName();
          builder.put(optionName, new OptionInfo(optionClass, def));
        }
      }

      return builder.build();
    }

    /**
     * Enter the options in buildOptions into a skylark dictionary, and return the dictionary.
     *
     * @throws IllegalArgumentException If the method is unable to look up the value in buildOptions
     *     corresponding to an entry in optionInfoMap.
     * @throws RuntimeException If the field corresponding to an option value in buildOptions is
     *     inaccessible due to Java language access control, or if an option name is an invalid key
     *     to the Skylark dictionary.
     */
    private SkylarkDict<String, Object> buildSettings(BuildOptions buildOptions,
        Map<String, OptionInfo> optionInfoMap) {
      try (Mutability mutability = Mutability.create("build_settings")) {
        SkylarkDict<String, Object> dict = SkylarkDict.withMutability(mutability);

        for (Map.Entry<String, OptionInfo> entry : optionInfoMap.entrySet()) {
          String optionName = entry.getKey();
          String optionKey = COMMAND_LINE_OPTION_PREFIX + optionName;
          OptionInfo optionInfo = entry.getValue();

          try {
            Field field = optionInfo.getDefinition().getField();
            FragmentOptions options = buildOptions.get(optionInfo.getOptionClass());
            Object optionValue = field.get(options);

            dict.put(optionKey, optionValue, null, mutability);
          } catch (IllegalAccessException | EvalException e) {
            // These exceptions should not happen, but if they do, throw a RuntimeException.
            throw new RuntimeException(e);
          }
        }

        return dict;
      }
    }

    /**
     * Evaluate the input function with the given argument, and return the return value.
     */
    private Object evalFunction(BaseFunction function, Object arg)
        throws InterruptedException, EvalException {
      try (Mutability mutability = Mutability.create("eval_transition_function")) {
        Environment env =
            Environment.builder(mutability)
            .setSemantics(semantics)
            .setEventHandler(eventHandler)
            .build();

        return function.call(ImmutableList.of(arg), ImmutableMap.of(), null, env);
      }
    }

    /**
     * Evaluate the transition function, and convert the result into a list of optionName ->
     * optionValue dictionaries.
     */
    private ImmutableList<Map<String, Object>> evalTransitionFunction(BaseFunction function,
        SkylarkDict<String, Object> settings)
        throws InterruptedException, EvalException {
      Object result;
      try {
        result = evalFunction(function, settings);
      } catch (EvalException e) {
        throw new EvalException(function.getLocation(), e.getMessage());
      }

      if (!(result instanceof SkylarkDict<?, ?>)) {
        throw new EvalException(function.getLocation(),
            "Transition function must return a dictionary.");
      }

      // The result is either:
      // 1. a dictionary mapping option name to new option value (for a single transition), or
      // 2. a dictionary of such dictionaries (for a split transition).
      //
      // First try to parse the result as a dictionary of option dictionaries; then try it as an
      // option dictionary.
      SkylarkDict<?, ?> dictOrDictOfDict = (SkylarkDict<?, ?>) result;

      try {
        Map<String, SkylarkDict> dictOfDict = dictOrDictOfDict.getContents(String.class,
            SkylarkDict.class, "dictionary of option dictionaries");

        ImmutableList.Builder<Map<String, Object>> builder = ImmutableList.builder();
        for (Map.Entry<String, SkylarkDict> entry : dictOfDict.entrySet()) {
          Map<String, Object> dict =
              entry.getValue().getContents(String.class, Object.class, "an option dictionary");
          builder.add(dict);
        }
        return builder.build();
      } catch (EvalException e) {
        // Fall through.
      }

      Map<String, Object> dict;
      try {
        dict = dictOrDictOfDict.getContents(String.class, Object.class, "an option dictionary");
      } catch (EvalException e) {
        throw new EvalException(function.getLocation(), e.getMessage());
      }

      return ImmutableList.of(dict);
    }

    /**
     * Apply the transition dictionary to the build option, using optionInfoMap to look up the
     * option info.
     *
     * @throws RuntimeException If a requested option field is inaccessible.
     */
    private void applyTransition(BuildOptions buildOptions, Map<String, Object> transition,
        Map<String, OptionInfo> optionInfoMap)
        throws EvalException {
      for (Map.Entry<String, Object> entry : transition.entrySet()) {
        String optionKey = entry.getKey();

        // TODO(juliexxia): Handle keys which correspond to build_setting target labels instead
        // of assuming every key is for a command line option.
        String optionName = commandLineOptionLabelToOption(optionKey);
        Object optionValue = entry.getValue();

        try {
          if (!optionInfoMap.containsKey(optionName)) {
            throw new EvalException(transitionFunction.getLocation(),
                "Unknown option '" + optionName + "'");
          }

          OptionInfo optionInfo = optionInfoMap.get(optionName);
          OptionDefinition def = optionInfo.getDefinition();
          Field field = def.getField();
          FragmentOptions options = buildOptions.get(optionInfo.getOptionClass());
          if (optionValue == null || def.getType().isInstance(optionValue)) {
            field.set(options, optionValue);
          } else if (optionValue instanceof String) {
            field.set(options, def.getConverter().convert((String) optionValue));
          } else {
            throw new EvalException(transitionFunction.getLocation(),
                "Invalid value type for option '" + optionName + "'");
          }
        } catch (IllegalAccessException e) {
          throw new RuntimeException(
              "IllegalAccess for option " + optionName + ": " + e.getMessage());
        } catch (OptionsParsingException e) {
          throw new EvalException(transitionFunction.getLocation(),
              "OptionsParsingError for option '" + optionName + "': " + e.getMessage());
        }
      }

      BuildConfiguration.Options buildConfigOptions;
      buildConfigOptions = buildOptions.get(BuildConfiguration.Options.class);
      updateOutputDirectoryNameFragment(buildConfigOptions, transition);
    }

    /**
     * Compute the output directory name fragment corresponding to the transition, and append it to
     * the existing name fragment in buildConfigOptions.
     *
     * @throws IllegalStateException If MD5 support is not available.
     */
    private void updateOutputDirectoryNameFragment(BuildConfiguration.Options buildConfigOptions,
        Map<String, Object> transition) {
      String transitionString = new String();
      for (Map.Entry<String, Object> entry : transition.entrySet()) {
        transitionString += entry.getKey() + ":";
        if (entry.getValue() != null) {
          transitionString += entry.getValue().toString() + "@";
        }
      }

      // TODO(waltl): for transitions that don't read settings, it is possible to precompute and
      // reuse the MD5 digest and even the transition itself.
      try {
        byte[] bytes = transitionString.getBytes(US_ASCII);
        MessageDigest md = MessageDigest.getInstance("MD5");
        byte[] digest = md.digest(bytes);
        String hexDigest = BaseEncoding.base16().lowerCase().encode(digest);

        if (buildConfigOptions.transitionDirectoryNameFragment == null) {
          buildConfigOptions.transitionDirectoryNameFragment = hexDigest;
        } else {
          buildConfigOptions.transitionDirectoryNameFragment += "-" + hexDigest;
        }
      } catch (NoSuchAlgorithmException e) {
        throw new IllegalStateException("MD5 not available", e);
      }
    }

    /**
     * Stores option info useful to a FunctionSplitTransition.
     */
    private static class OptionInfo {
      private final Class<? extends FragmentOptions> optionClass;
      private final OptionDefinition definition;

      public OptionInfo(Class<? extends FragmentOptions> optionClass,
          OptionDefinition definition) {
        this.optionClass = optionClass;
        this.definition = definition;
      }

      Class<? extends FragmentOptions> getOptionClass() {
        return optionClass;
      }

      OptionDefinition getDefinition() {
        return definition;
      }
    }
  }
}
