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

package com.google.devtools.build.lib.analysis.config;

import static com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition.PATCH_TRANSITION_KEY;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.BazelStarlarkContext.Phase;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigurationTransitionApi;
import com.google.errorprone.annotations.FormatMethod;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Implementation of {@link ConfigurationTransitionApi}.
 *
 * <p>Represents a configuration transition across a dependency edge defined in Starlark.
 */
public abstract class StarlarkDefinedConfigTransition implements ConfigurationTransitionApi {

  public static final String COMMAND_LINE_OPTION_PREFIX = "//command_line_option:";

  /**
   * The two groups of build settings that are relevant for a {@link
   * StarlarkDefinedConfigTransition}
   */
  public enum Settings {
    /** Build settings that are read by a {@link StarlarkDefinedConfigTransition} */
    INPUTS,
    /** Build settings that are written by a {@link StarlarkDefinedConfigTransition} */
    OUTPUTS,
    /** Build settings that are read and/or written by a {@link StarlarkDefinedConfigTransition } */
    INPUTS_AND_OUTPUTS
  }

  private final ImmutableMap<String, String> inputsCanonicalizedToGiven;
  private final ImmutableMap<String, String> outputsCanonicalizedToGiven;
  private final Location location;

  private StarlarkDefinedConfigTransition(
      List<String> inputs,
      List<String> outputs,
      ImmutableMap<RepositoryName, RepositoryName> repoMapping,
      Label parentLabel,
      Location location)
      throws EvalException {
    this.location = location;

    this.outputsCanonicalizedToGiven =
        getCanonicalizedSettings(repoMapping, parentLabel, outputs, Settings.OUTPUTS);
    this.inputsCanonicalizedToGiven =
        getCanonicalizedSettings(repoMapping, parentLabel, inputs, Settings.INPUTS);
  }

  /**
   * Returns a build settings in canonicalized form taking into account repository remappings.
   * Native options only have one form so they are always returned unchanged (i.e.
   * //command_line_option:<option-name>).
   */
  private static String canonicalizeSetting(
      String setting, ImmutableMap<RepositoryName, RepositoryName> repoMapping, Label parentLabel)
      throws LabelSyntaxException {
    String canonicalizedString = setting;
    // native options
    if (setting.startsWith(COMMAND_LINE_OPTION_PREFIX)) {
      return canonicalizedString;
    }
    canonicalizedString =
        parentLabel.getRelativeWithRemapping(setting, repoMapping).getUnambiguousCanonicalForm();
    return canonicalizedString;
  }

  /**
   * Canonicalize the given list of settings. Return a map of their canonicalized version to the
   * form they were given in. Along the way make sure that this list of settings doesn't contain two
   * label strings that look different but canonicalize to the same target.
   *
   * @return a map of the canonicalized version of the build settings to the form the user gave
   *     them. In the case of native options, the key and value of the entry are the same -
   *     "//command_line_option:<option-name>"
   */
  private static ImmutableMap<String, String> getCanonicalizedSettings(
      ImmutableMap<RepositoryName, RepositoryName> repoMapping,
      Label parentLabel,
      List<String> settings,
      Settings inputsOrOutputs)
      throws EvalException {
    Map<String, String> canonicalizedToGiven = new HashMap<>();
    for (String setting : settings) {
      String canonicalizedSetting;
      try {
        canonicalizedSetting = canonicalizeSetting(setting, repoMapping, parentLabel);
      } catch (LabelSyntaxException unused) {
        throw Starlark.errorf(
            "Malformed label in transition %s parameter: '%s'", inputsOrOutputs, setting);
      }
      String previousSetting = canonicalizedToGiven.put(canonicalizedSetting, setting);
      if (previousSetting != null) {
        throw Starlark.errorf(
            "Transition declares duplicate build setting '%s' in %s (specified as '%s' and '%s')",
            canonicalizedSetting, inputsOrOutputs, setting, previousSetting);
      }
    }
    return ImmutableSortedMap.copyOf(canonicalizedToGiven);
  }

  /**
   * Returns true if this transition is for analysis testing. If true, then only attributes of rules
   * with {@code analysis_test=true} may use this transition object.
   */
  public abstract Boolean isForAnalysisTesting();

  /**
   * Returns the given input option keys for this transition. Only options contained in this list
   * will be provided in the 'settings' argument given to the transition implementation function.
   */
  public List<String> getInputs() {
    return inputsCanonicalizedToGiven.values().asList();
  }

  public ImmutableMap<String, String> getInputsCanonicalizedToGiven() {
    return inputsCanonicalizedToGiven;
  }

  /**
   * Returns the given output option keys for this transition. The transition implementation
   * function must return a dictionary where the options exactly match the elements of this list.
   */
  public ImmutableList<String> getOutputs() {
    return outputsCanonicalizedToGiven.values().asList();
  }

  public ImmutableMap<String, String> getOutputsCanonicalizedToGiven() {
    return outputsCanonicalizedToGiven;
  }

  /** Returns the location of the Starlark code defining the transition. */
  public Location getLocation() {
    return location;
  }

  /**
   * Given a map of a subset of the "previous" build settings, returns the changed build settings as
   * a result of applying this transition.
   *
   * @param previousSettings a map representing the previous build settings
   * @return a map of changed build setting maps; each element of the map represents a different
   *     child configuration (split transitions will have multiple elements in this map with keys
   *     provided by the transition impl, patch transitions should have a single element keyed by
   *     {@code PATCH_TRANSITION_KEY}). Each build setting map is a map from build setting to target
   *     setting value; all other build settings will remain unchanged. Returns null if errors were
   *     reported to the handler.
   * @throws InterruptedException if evaluating the transition is interrupted
   */
  @Nullable
  public abstract ImmutableMap<String, Map<String, Object>> evaluate(
      Map<String, Object> previousSettings, StructImpl attributeMap, EventHandler eventHandler)
      throws InterruptedException;

  public static StarlarkDefinedConfigTransition newRegularTransition(
      StarlarkCallable impl,
      List<String> inputs,
      List<String> outputs,
      StarlarkSemantics semantics,
      Label parentLabel,
      Location location,
      BazelStarlarkContext starlarkContext)
      throws EvalException {
    return new RegularTransition(
        impl, inputs, outputs, semantics, parentLabel, location, starlarkContext.getRepoMapping());
  }

  public static StarlarkDefinedConfigTransition newAnalysisTestTransition(
      Map<String, Object> changedSettings,
      ImmutableMap<RepositoryName, RepositoryName> repoMapping,
      Label parentLabel,
      Location location)
      throws EvalException {
    return new AnalysisTestTransition(changedSettings, repoMapping, parentLabel, location);
  }

  private static class AnalysisTestTransition extends StarlarkDefinedConfigTransition {
    private final Map<String, Object> changedSettings;

    public AnalysisTestTransition(
        Map<String, Object> changedSettings,
        ImmutableMap<RepositoryName, RepositoryName> repoMapping,
        Label parentLabel,
        Location location)
        throws EvalException {
      super(
          /*inputs=*/ ImmutableList.of(),
          ImmutableList.copyOf(changedSettings.keySet()),
          repoMapping,
          parentLabel,
          location);
      this.changedSettings = changedSettings;
    }

    @Override
    public Boolean isForAnalysisTesting() {
      return true;
    }

    @Override
    public ImmutableMap<String, Map<String, Object>> evaluate(
        Map<String, Object> previousSettings,
        StructImpl attributeMapper,
        EventHandler eventHandler) {
      return ImmutableMap.of(PATCH_TRANSITION_KEY, changedSettings);
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<analysis_test_transition object>");
    }

    @Override
    public boolean equals(Object object) {
      if (object == this) {
        return true;
      }
      if (object instanceof AnalysisTestTransition) {
        AnalysisTestTransition otherTransition = (AnalysisTestTransition) object;
        return Objects.equals(otherTransition.getInputs(), this.getInputs())
            && Objects.equals(otherTransition.getOutputs(), this.getOutputs())
            && Objects.equals(otherTransition.changedSettings, this.changedSettings);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(this.getInputs(), this.getOutputs(), this.changedSettings);
    }
  }

  /** A transition with a user-defined implementation function. */
  public static class RegularTransition extends StarlarkDefinedConfigTransition {
    private final StarlarkCallable impl;
    private final StarlarkSemantics semantics;
    private final ImmutableMap<RepositoryName, RepositoryName> repoMapping;
    private final Label parentLabel;

    RegularTransition(
        StarlarkCallable impl,
        List<String> inputs,
        List<String> outputs,
        StarlarkSemantics semantics,
        Label parentLabel,
        Location location,
        ImmutableMap<RepositoryName, RepositoryName> repoMapping)
        throws EvalException {
      super(inputs, outputs, repoMapping, parentLabel, location);
      this.impl = impl;
      this.semantics = semantics;
      this.parentLabel = parentLabel;
      this.repoMapping = repoMapping;
    }

    @Override
    public Boolean isForAnalysisTesting() {
      return false;
    }

    /**
     * This method evaluates the implementation function of the transition.
     *
     * <p>In the case of a {@link
     * com.google.devtools.build.lib.analysis.config.transitions.PatchTransition}, the impl fxn
     * returns a {@link Dict} of option name strings to option value object.
     *
     * <p>In the case of {@link
     * com.google.devtools.build.lib.analysis.config.transitions.SplitTransition}, the impl fxn can
     * return either a {@link Dict} of String keys to {@link Dict} values. Or it can return a list
     * of {@link Dict}s in cases where the consumer doesn't care about differentiating between the
     * splits (i.e. accessing later via {@code ctx.split_attrs}).
     *
     * @param previousSettings a map representing the previous build settings
     * @param attributeMapper a map of attributes
     */
    // TODO(bazel-team): integrate dict-of-dicts return type with ctx.split_attr
    @Nullable
    @Override
    public ImmutableMap<String, Map<String, Object>> evaluate(
        Map<String, Object> previousSettings, StructImpl attributeMapper, EventHandler handler)
        throws InterruptedException {
      // Call the Starlark function.
      Object result;
      try (Mutability mu = Mutability.create("eval_transition_function")) {
        StarlarkThread thread = new StarlarkThread(mu, semantics);
        thread.setPrintHandler(Event.makeDebugPrintHandler(handler));
        // TODO: If the resulting values of Starlark transitions ever evolve to be
        //  complex Starlark objects like structs as opposed to the ints, strings,
        //  etc they are today then we need a real symbol generator which is used
        //  to calculate equality between instances of Starlark objects. A candidate
        //  for transition instance uniqueness is the Rule and configuration that
        //  are used as inputs to the configuration.
        SymbolGenerator<Object> dummySymbolGenerator = new SymbolGenerator<>(new Object());

        // Create a new {@link BazelStarlarkContext} for the new thread. We need to
        // create a new context every time because {@link BazelStarlarkContext}s
        // should be confined to a single thread.
        BazelStarlarkContext starlarkContext =
            new BazelStarlarkContext(
                Phase.ANALYSIS,
                /*toolsRepository=*/ null,
                /*fragmentNameToClass=*/ null,
                repoMapping,
                /*convertedLabelsInPackage=*/ new HashMap<>(),
                dummySymbolGenerator,
                parentLabel);

        starlarkContext.storeInThread(thread);
        result =
            Starlark.fastcall(
                thread, impl, new Object[] {previousSettings, attributeMapper}, new Object[0]);
      } catch (EvalException ex) {
        handler.handle(Event.error(null, ex.getMessageWithStack()));
        return null;
      }

      if (result instanceof Dict) {
        // We need to special case empty dicts because if we don't, the error reported for rule
        // transitions (which must be 1:1) is that we're trying to return a dict of dicts, instead
        // of reporting the missing return values.
        if (((Dict) result).isEmpty()) {
          // Check if we're missing return values and this dict *shouldn't* be empty.
          try {
            validateFunctionOutputsMatchesDeclaredOutputs(ImmutableSet.of(), getOutputs());
          } catch (ValidationException ex) {
            errorf(handler, "invalid result from transition function: %s", ex.getMessage());
            return null;
          }
          // If it's properly empty, return empty dict.
          return ImmutableMap.of(PATCH_TRANSITION_KEY, ImmutableMap.of());
        }
        try {
          Map<String, ?> dictOfDict =
              Dict.cast(result, String.class, Dict.class, "dictionary of options dictionaries");
          ImmutableMap.Builder<String, Map<String, Object>> builder = ImmutableMap.builder();
          for (Map.Entry<String, ?> entry : dictOfDict.entrySet()) {
            Map<String, Object> rawDict =
                Dict.cast(entry.getValue(), String.class, Object.class, "dictionary of options");
            Map<String, Object> canonicalizedDict =
                canonicalizeTransitionOutputDict(rawDict, repoMapping, parentLabel, getOutputs());
            builder.put(entry.getKey(), canonicalizedDict);
          }
          return builder.build();
        } catch (ValidationException ex) {
          errorf(handler, "invalid result from transition function: %s", ex.getMessage());
          return null;
        } catch (EvalException ex) {
          // Fall through assuming the Dict#cast call didn't work as this is a single dictionary
          // not a dictionary of dictionaries.
        }
        try {
          // Try if this is a patch transition.
          Map<String, Object> rawDict =
              Dict.cast(result, String.class, Object.class, "dictionary of options");
          Map<String, Object> canonicalizedDict =
              canonicalizeTransitionOutputDict(rawDict, repoMapping, parentLabel, getOutputs());
          return ImmutableMap.of(PATCH_TRANSITION_KEY, canonicalizedDict);
        } catch (EvalException | ValidationException ex) {
          // TODO(adonovan): explain "want dict<string, any> or dict<string, dict<string, any>>".
          errorf(handler, "invalid result from transition function: %s", ex.getMessage());
          return null;
        }

      } else if (result instanceof Sequence) {
        ImmutableMap.Builder<String, Map<String, Object>> builder = ImmutableMap.builder();
        try {
          int i = 0;
          for (Dict<?, ?> entry :
              Sequence.cast(result, Dict.class, "dictionary of options dictionaries")) {
            // TODO(b/146347033): Document this behavior.
            Map<String, Object> rawDict =
                Dict.cast(entry, String.class, Object.class, "dictionary of options");
            Map<String, Object> canonicalizedDict =
                canonicalizeTransitionOutputDict(rawDict, repoMapping, parentLabel, getOutputs());
            builder.put(Integer.toString(i++), canonicalizedDict);
          }
        } catch (EvalException | ValidationException ex) {
          // TODO(adonovan): explain "want sequence of dict<string, any>".
          errorf(handler, "invalid result from transition function: %s", ex.getMessage());
          return null;
        }
        return builder.build();
      } else {
        errorf(
            handler,
            "transition function returned %s, want dict or list of dicts",
            Starlark.type(result));
        return null;
      }
    }

    @FormatMethod
    private void errorf(EventHandler handler, String format, Object... args) {
      handler.handle(Event.error(impl.getLocation(), String.format(format, args)));
    }

    /**
     * Validates that function outputs exactly the set of outputs it declares, as they were declared
     * (i.e. not canonicalized or in another form of the same label). More thorough checking (like
     * type checking of output values) is done elsewhere because it requires loading. see {@link
     * com.google.devtools.build.lib.analysis.starlark.StarlarkTransition#validate}
     *
     * @param returnedKeySet actual key set of dict returned by starlark transition.
     * @param declaredReturnSettings list of build settings to return as declared by the 'outputs'
     *     parameter (in their given form) to the transition definition.
     */
    private static void validateFunctionOutputsMatchesDeclaredOutputs(
        Set<String> returnedKeySet, List<String> declaredReturnSettings)
        throws ValidationException {
      if (returnedKeySet.containsAll(declaredReturnSettings)
          && returnedKeySet.size() == declaredReturnSettings.size()) {
        return;
      }

      LinkedHashSet<String> remainingOutputs = Sets.newLinkedHashSet(declaredReturnSettings);
      for (String outputKey : returnedKeySet) {
        if (!remainingOutputs.remove(outputKey)) {
          throw new ValidationException(
              String.format("transition function returned undeclared output '%s'", outputKey));
        }
      }

      if (!remainingOutputs.isEmpty()) {
        throw new ValidationException(
            String.format(
                "transition outputs [%s] were not defined by transition function",
                remainingOutputs.stream().collect(joining(","))));
      }
    }

    /**
     * Given a map of build settings to their values, return a map with the same build settings but
     * in their canonicalized string form to their values.
     *
     * <p>TODO(blaze-configurability): It would be nice if this method also returned a map of the
     * canonicalized settings to given settings so that when we throw the "unrecognized returned
     * option" warning we can show the setting as the user gave it as well as in its canonicalized
     * form.
     */
    private static ImmutableMap<String, Object> canonicalizeTransitionOutputDict(
        Map<String, Object> dict,
        ImmutableMap<RepositoryName, RepositoryName> repoMapping,
        Label parentLabel,
        List<String> outputs)
        throws EvalException, ValidationException {
      validateFunctionOutputsMatchesDeclaredOutputs(dict.keySet(), outputs);

      Map<String, String> canonicalizedToGiven = new HashMap<>();
      ImmutableSortedMap.Builder<String, Object> canonicalizedDict =
          new ImmutableSortedMap.Builder<>(Ordering.natural());
      for (Map.Entry<String, Object> entry : dict.entrySet()) {
        String returnedSetting = entry.getKey();
        String label;
        try {
          label = canonicalizeSetting(returnedSetting, repoMapping, parentLabel);
        } catch (LabelSyntaxException unused) {
          throw Starlark.errorf(
              "Malformed label in transition return dictionary: '%s'", returnedSetting);
        }
        String previousGiven = canonicalizedToGiven.put(label, returnedSetting);
        if (previousGiven != null) {
          throw Starlark.errorf(
              "Transition implementation function returns the same option '%s' in two different"
                  + " keys: '%s' and '%s'",
              label, returnedSetting, previousGiven);
        }
        canonicalizedDict.put(label, entry.getValue());
      }
      return canonicalizedDict.build();
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<transition object>");
    }

    @Override
    public boolean equals(Object object) {
      if (object == this) {
        return true;
      }
      if (object instanceof RegularTransition) {
        RegularTransition otherTransition = (RegularTransition) object;
        return Objects.equals(otherTransition.getInputs(), this.getInputs())
            && Objects.equals(otherTransition.getOutputs(), this.getOutputs())
            && Objects.equals(otherTransition.impl, this.impl);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(this.getInputs(), this.getOutputs(), this.impl);
    }
  }

  /** An exception for validating that a transition is properly constructed */
  public static final class ValidationException extends Exception {
    public ValidationException(String message) {
      super(message);
    }

    @FormatMethod
    public static ValidationException format(String format, Object... args) {
      return new ValidationException(String.format(format, args));
    }
  }
}
