// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.Immutable;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Ninja target (build statement) representation. */
public final class NinjaTarget {

  /**
   * A list of variables that pertain to input and output artifact paths of a Ninja target. For
   * memory savings, these variables should be expanded late, and the expansion results not retained
   * in memory. (Otherwise, these paths are persisted twice!)
   */
  private static final ImmutableSet<String> INPUTS_OUTPUTS_VARIABLES =
      ImmutableSet.of("in", "in_newline", "out");

  /** Builder for {@link NinjaTarget}. */
  public static class Builder {
    private String ruleName;
    private final ImmutableSortedKeyListMultimap.Builder<InputKind, PathFragment> inputsBuilder;
    private final ImmutableSortedKeyListMultimap.Builder<OutputKind, PathFragment> outputsBuilder;
    private final NinjaScope scope;
    private final long offset;

    private final ImmutableSortedMap.Builder<String, String> variablesBuilder;
    private final Interner<String> nameInterner;

    private Builder(NinjaScope scope, long offset, Interner<String> nameInterner) {
      this.scope = scope;
      this.offset = offset;
      inputsBuilder = ImmutableSortedKeyListMultimap.builder();
      outputsBuilder = ImmutableSortedKeyListMultimap.builder();
      variablesBuilder = ImmutableSortedMap.naturalOrder();
      this.nameInterner = nameInterner;
    }

    public Builder setRuleName(String ruleName) {
      this.ruleName = ruleName;
      return this;
    }

    public Builder addInputs(InputKind kind, Collection<PathFragment> inputs) {
      inputsBuilder.putAll(kind, inputs);
      return this;
    }

    public Builder addOutputs(OutputKind kind, Collection<PathFragment> outputs) {
      outputsBuilder.putAll(kind, outputs);
      return this;
    }

    public Builder addVariable(String key, String value) {
      variablesBuilder.put(key, value);
      return this;
    }

    public NinjaTarget build() throws GenericParsingException {
      Preconditions.checkNotNull(ruleName);
      String internedName = nameInterner.intern(ruleName);
      ImmutableSortedMap<String, String> variables = variablesBuilder.build();

      ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> ruleVariables;
      if (internedName.equals("phony")) {
        ruleVariables = ImmutableSortedMap.of();
      } else {
        NinjaRule ninjaRule = scope.findRule(offset, ruleName);
        if (ninjaRule == null) {
          throw new GenericParsingException(
              String.format("could not resolve rule '%s'", internedName));
        } else {
          ruleVariables =
              reduceRuleVariables(scope, offset, ninjaRule.getVariables(), variables, nameInterner);
        }
      }
      return new NinjaTarget(
          nameInterner.intern(ruleName),
          inputsBuilder.build(),
          outputsBuilder.build(),
          offset,
          ruleVariables);
    }

    /**
     * We expand the rule's variables with the following assumptions: Rule variables can refer to
     * target's variables (and file variables). Interdependence between rule variables can happen
     * only for 'command' variable, for now we ignore other possible dependencies between rule
     * variables (seems the only other variable which can meaningfully depend on sibling variables
     * is description, and currently we are ignoring it).
     *
     * <p>Also, for resolving rule's variables we are using scope+offset of target, according to
     * specification (https://ninja-build.org/manual.html#_variable_expansion).
     *
     * <p>See {@link NinjaRuleVariable} for the list.
     */
    private static ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> reduceRuleVariables(
        NinjaScope targetScope,
        long targetOffset,
        Map<NinjaRuleVariable, NinjaVariableValue> ruleVariables,
        ImmutableSortedMap<String, String> targetVariables,
        Interner<String> interner) {
      ImmutableSortedMap.Builder<String, List<Pair<Long, String>>> variablesBuilder =
          ImmutableSortedMap.naturalOrder();
      targetVariables.forEach(
          (key, value) -> variablesBuilder.put(key, ImmutableList.of(Pair.of(0L, value))));
      NinjaScope scopeWithVariables =
          targetScope.createScopeFromExpandedValues(variablesBuilder.build());

      ImmutableSortedMap.Builder<NinjaRuleVariable, NinjaVariableValue> builder =
          ImmutableSortedMap.naturalOrder();

      // Description is taken from the "build" statement (instead of the referenced rule)
      // if it's available.
      boolean targetHasDescription = false;
      String targetVariable = targetVariables.get("description");
      if (targetVariable != null) {
        builder.put(
            NinjaRuleVariable.DESCRIPTION, NinjaVariableValue.createPlainText(targetVariable));
        targetHasDescription = true;
      }

      for (Map.Entry<NinjaRuleVariable, NinjaVariableValue> entry : ruleVariables.entrySet()) {
        NinjaRuleVariable type = entry.getKey();
        if (type.equals(NinjaRuleVariable.DESCRIPTION) && targetHasDescription) {
          // Don't use the rule description, as the target defined a specific description.
          continue;
        }
        NinjaVariableValue reducedValue =
            scopeWithVariables.getReducedValue(
                targetOffset, entry.getValue(), INPUTS_OUTPUTS_VARIABLES, interner);
        builder.put(type, reducedValue);
      }
      return builder.build();
    }
  }

  /** Enum with possible kinds of inputs. */
  @Immutable
  public enum InputKind implements InputOutputKind {
    EXPLICIT,
    IMPLICIT,
    ORDER_ONLY,
    VALIDATION,
  }

  /** Enum with possible kinds of outputs. */
  @Immutable
  public enum OutputKind implements InputOutputKind {
    EXPLICIT,
    IMPLICIT
  }

  /**
   * Marker interface, so that it is possible to address {@link InputKind} and {@link OutputKind}
   * together in one map.
   */
  @Immutable
  public interface InputOutputKind {}

  private final String ruleName;
  private final ImmutableSortedKeyListMultimap<InputKind, PathFragment> inputs;
  private final ImmutableSortedKeyListMultimap<OutputKind, PathFragment> outputs;
  private final long offset;

  /**
   * A "reduced" set of ninja rule variables. All variables are expanded except for those in {@link
   * #INPUTS_OUTPUTS_VARIABLES}, as this saves memory.
   */
  private final ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> ruleVariables;

  private NinjaTarget(
      String ruleName,
      ImmutableSortedKeyListMultimap<InputKind, PathFragment> inputs,
      ImmutableSortedKeyListMultimap<OutputKind, PathFragment> outputs,
      long offset,
      ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> ruleVariables) {
    this.ruleName = ruleName;
    this.inputs = inputs;
    this.outputs = outputs;
    this.offset = offset;
    this.ruleVariables = ruleVariables;
  }

  public String getRuleName() {
    return ruleName;
  }

  public List<PathFragment> getOutputs() {
    return outputs.get(OutputKind.EXPLICIT);
  }

  public List<PathFragment> getImplicitOutputs() {
    return outputs.get(OutputKind.IMPLICIT);
  }

  public Collection<PathFragment> getAllOutputs() {
    return outputs.values();
  }

  public Collection<PathFragment> getAllInputs() {
    return inputs.values();
  }

  public Collection<Map.Entry<InputKind, PathFragment>> getAllInputsAndKind() {
    return inputs.entries();
  }

  public Collection<PathFragment> getExplicitInputs() {
    return inputs.get(InputKind.EXPLICIT);
  }

  public Collection<PathFragment> getImplicitInputs() {
    return inputs.get(InputKind.IMPLICIT);
  }

  public Collection<PathFragment> getOrderOnlyInputs() {
    return inputs.get(InputKind.ORDER_ONLY);
  }

  public Collection<PathFragment> getValidationInputs() {
    return inputs.get(InputKind.VALIDATION);
  }

  public long getOffset() {
    return offset;
  }

  public static Builder builder(NinjaScope scope, long offset, Interner<String> nameInterner) {
    return new Builder(scope, offset, nameInterner);
  }

  public String prettyPrint() {
    return "build "
        + prettyPrintPaths("\n", getOutputs())
        + prettyPrintPaths("\n| ", getImplicitOutputs())
        + "\n: "
        + this.ruleName
        + prettyPrintPaths("\n", getExplicitInputs())
        + prettyPrintPaths("\n| ", getImplicitInputs())
        + prettyPrintPaths("\n|| ", getOrderOnlyInputs());
  }

  @Override
  public int hashCode() {
    return Long.hashCode(offset);
  }

  /**
   * Returns a map from rule variable to fully-expanded value, for all rule variables defined in
   * this target.
   */
  public ImmutableSortedMap<NinjaRuleVariable, String> computeRuleVariables() {
    ImmutableSortedMap<String, String> lateExpansionVariables = computeInputOutputVariables();
    ImmutableSortedMap.Builder<String, String> fullExpansionVariablesBuilder =
        ImmutableSortedMap.<String, String>naturalOrder().putAll(lateExpansionVariables);

    ImmutableSortedMap.Builder<NinjaRuleVariable, String> builder =
        ImmutableSortedMap.naturalOrder();
    for (Map.Entry<NinjaRuleVariable, NinjaVariableValue> entry : ruleVariables.entrySet()) {
      NinjaRuleVariable type = entry.getKey();
      // Skip command for now. It may need to expand other rule variables.
      if (NinjaRuleVariable.COMMAND.equals(type)) {
        continue;
      }

      String expandedValue = entry.getValue().expandValue(lateExpansionVariables);
      builder.put(type, expandedValue);
      fullExpansionVariablesBuilder.put(Ascii.toLowerCase(type.name()), expandedValue);
    }

    // TODO(cparsons): Ensure parsing exception is thrown early if the rule has no command defined.
    // Otherwise, this throws NPE.
    String expandedCommand =
        ruleVariables
            .get(NinjaRuleVariable.COMMAND)
            .expandValue(fullExpansionVariablesBuilder.build());
    builder.put(NinjaRuleVariable.COMMAND, expandedCommand);
    return builder.build();
  }

  private ImmutableSortedMap<String, String> computeInputOutputVariables() {
    ImmutableSortedMap.Builder<String, String> builder = ImmutableSortedMap.naturalOrder();
    String inNewline =
        inputs.get(InputKind.EXPLICIT).stream()
            .map(PathFragment::getPathString)
            .collect(Collectors.joining("\n"));
    String out =
        outputs.get(OutputKind.EXPLICIT).stream()
            .map(PathFragment::getPathString)
            .collect(Collectors.joining(" "));
    builder.put("in", inNewline.replace('\n', ' '));
    builder.put("in_newline", inNewline);
    builder.put("out", out);
    return builder.build();
  }

  private static String prettyPrintPaths(String startDelimiter, Collection<PathFragment> paths) {
    if (paths.isEmpty()) {
      return "";
    }
    return startDelimiter
        + paths.stream().map(PathFragment::getPathString).collect(Collectors.joining("$\n"));
  }
}
