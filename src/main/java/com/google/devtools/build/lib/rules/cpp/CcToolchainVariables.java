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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcToolchainVariablesApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.Stack;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.Structure;

/**
 * Configured build variables usable by the toolchain configuration.
 *
 * <p>TODO(b/32655571): Investigate cleanup once implicit iteration is not needed. Variables
 * instance could serve as a top level View used to expand all flag_groups.
 */
@Immutable
public abstract class CcToolchainVariables implements CcToolchainVariablesApi {
  /**
   * A piece of a single string value.
   *
   * <p>A single value can contain a combination of text and variables (for example "-f
   * %{var1}/%{var2}"). We split the string into chunks, where each chunk represents either a text
   * snippet, or a variable that is to be replaced.
   */
  interface StringChunk {
    /**
     * Expands this chunk.
     *
     * @param variables binding of variable names to their values for a single flag expansion.
     */
    String expand(CcToolchainVariables variables, PathMapper pathMapper) throws ExpansionException;

    String getString();
  }

  /** A plain text chunk of a string (containing no variables). */
  @Immutable
  private static final class StringLiteralChunk implements StringChunk {
    private final String text;

    StringLiteralChunk(String text) {
      this.text = text;
    }

    @Override
    public String expand(CcToolchainVariables variables, PathMapper pathMapper) {
      return text;
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof StringLiteralChunk that) {
        return text.equals(that.text);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return 31 + text.hashCode();
    }

    @Override
    public String getString() {
      return text;
    }
  }

  /** A chunk of a string value into which a variable should be expanded. */
  @Immutable
  private static final class VariableChunk implements StringChunk {
    private final String variableName;

    VariableChunk(String variableName) {
      this.variableName = variableName;
    }

    @Override
    public String expand(CcToolchainVariables variables, PathMapper pathMapper)
        throws ExpansionException {
      // We check all variables in FlagGroup.expandCommandLine.
      // If we arrive here with the variable not being available, the variable was provided, but
      // the nesting level of the NestedSequence was deeper than the nesting level of the flag
      // groups.
      return variables.getStringVariable(variableName, pathMapper);
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof VariableChunk that) {
        return variableName.equals(that.variableName);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(variableName);
    }

    @Override
    public String getString() {
      return "%{" + variableName + "}";
    }
  }

  /**
   * Parser for toolchain string values.
   *
   * <p>A string value contains a snippet of text supporting variable expansion. For example, a
   * string value "-f %{var1}/%{var2}" will expand the values of the variables "var1" and "var2" in
   * the corresponding places in the string.
   *
   * <p>The {@code StringValueParser} takes a string and parses it into a list of {@link
   * StringChunk} objects, where each chunk represents either a snippet of text or a variable to be
   * expanded. In the above example, the resulting chunks would be ["-f ", var1, "/", var2].
   *
   * <p>To get a literal percent character, "%%" can be used in the string.
   */
  public static class StringValueParser {

    private final String value;

    /**
     * The current position in {@value} during parsing.
     */
    private int current = 0;

    private final ImmutableList.Builder<StringChunk> chunks = ImmutableList.builder();
    private final ImmutableSet.Builder<String> usedVariables = ImmutableSet.builder();

    public StringValueParser(String value) throws EvalException {
      this.value = value;
      parse();
    }

    /** Returns the parsed chunks for this string. */
    public ImmutableList<StringChunk> getChunks() {
      return chunks.build();
    }

    /**
     * Parses the string.
     *
     * @throws EvalException if there is a parsing error.
     */
    private void parse() throws EvalException {
      while (current < value.length()) {
        if (atVariableStart()) {
          parseVariableChunk();
        } else {
          parseStringChunk();
        }
      }
    }

    /**
     * @return whether the current position is the start of a variable.
     */
    private boolean atVariableStart() {
      // We parse a variable when value starts with '%', but not '%%'.
      return value.charAt(current) == '%'
          && (current + 1 >= value.length() || value.charAt(current + 1) != '%');
    }

    /**
     * Parses a chunk of text until the next '%', which indicates either an escaped literal '%' or a
     * variable.
     */
    private void parseStringChunk() {
      int start = current;
      // We only parse string chunks starting with '%' if they also start with '%%'.
      // In that case, we want to have a single '%' in the string, so we start at the second
      // character.
      // Note that for strings like "abc%%def" this will lead to two string chunks, the first
      // referencing the subtring "abc", and a second referencing the substring "%def".
      if (value.charAt(current) == '%') {
        current = current + 1;
        start = current;
      }
      current = value.indexOf('%', current + 1);
      if (current == -1) {
        current = value.length();
      }
      String text = value.substring(start, current);
      chunks.add(new StringLiteralChunk(text));
    }

    /**
     * Parses a variable to be expanded.
     *
     * @throws EvalException if there is a parsing error.
     */
    private void parseVariableChunk() throws EvalException {
      current = current + 1;
      if (current >= value.length() || value.charAt(current) != '{') {
        abort("expected '{'");
      }
      current = current + 1;
      if (current >= value.length() || value.charAt(current) == '}') {
        abort("expected variable name");
      }
      int end = value.indexOf('}', current);
      final String name = value.substring(current, end);
      usedVariables.add(name);
      chunks.add(new VariableChunk(name));
      current = end + 1;
    }

    /**
     * @throws EvalException with the given error text, adding information about the current
     *     position in the string.
     */
    private void abort(String error) throws EvalException {
      throw Starlark.errorf(
          "Invalid toolchain configuration: %s at position %s while parsing a flag containing '%s'",
          error, current, value);
    }
  }

  /** A flag or flag group that can be expanded under a set of variables. */
  public interface Expandable {
    /**
     * Expands the current expandable under the given {@code view}, adding new flags to {@code
     * commandLine}.
     *
     * <p>The {@code variables} controls which variables are visible during the expansion and allows
     * to recursively expand nested flag groups.
     */
    void expand(
        CcToolchainVariables variables,
        @Nullable InputMetadataProvider inputMetadataProvider,
        PathMapper pathMapper,
        List<String> commandLine)
        throws ExpansionException;
  }

  /** Returns an empty variables instance. */
  public static CcToolchainVariables empty() {
    return EmptyVariablesHolder.EMPTY;
  }

  /**
   * Avoids cyclic class initialization issues with {@link MapVariables}.
   *
   * <p>Without this holder, there would be a cycle here. {@link MapVariables} depends on its parent
   * class {@link CcToolchainVariables} and {@link CcToolchainVariables} would depend on {@link
   * MapVariables} via {@link #EMPTY}.
   *
   * <p>See <a
   * href="https://en.wikipedia.org/wiki/Initialization-on-demand_holder_idiom">Initialization on
   * demand idiom</a>.
   */
  private static class EmptyVariablesHolder {
    private static final CcToolchainVariables EMPTY = builder().build();
  }

  private static final Object NULL_MARKER = new Object();

  // Values in this cache are either VariableValue, String error message, or NULL_MARKER.
  //
  // It is initialized lazily.
  private transient volatile Map<String, Object> structuredVariableCache;

  /**
   * Retrieves a string sequence variable named {@code variableName} from {@code variables} and
   * converts it into a list of plain strings.
   *
   * <p>Throws {@link ExpansionException} when the variable is not a string sequence.
   */
  public static ImmutableList<String> toStringList(
      CcToolchainVariables variables, String variableName, PathMapper pathMapper)
      throws ExpansionException {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (VariableValue value :
        getSequenceValue(variableName, variables.getVariable(variableName, pathMapper))) {
      result.add(value.getStringValue(variableName, pathMapper));
    }
    return result.build();
  }

  /**
   * Gets a variable value named {@code name}. Supports accessing fields in structures (e.g.
   * 'libraries_to_link.interface_libraries')
   *
   * @throws ExpansionException when no such variable or no such field are present, or when
   *     accessing a field of non-structured variable
   */
  VariableValue getVariable(String name, PathMapper pathMapper) throws ExpansionException {
    return lookupVariable(
        name, /* throwOnMissingVariable= */ true, /* inputMetadataProvider= */ null, pathMapper);
  }

  VariableValue getVariable(
      String name, @Nullable InputMetadataProvider inputMetadataProvider, PathMapper pathMapper)
      throws ExpansionException {
    return lookupVariable(
        name, /* throwOnMissingVariable= */ true, inputMetadataProvider, pathMapper);
  }

  /**
   * Looks up a variable named {@code name} or return a reason why the variable was not found.
   * Supports accessing fields in structures.
   */
  @Nullable
  private VariableValue lookupVariable(
      String name,
      boolean throwOnMissingVariable,
      @Nullable InputMetadataProvider inputMetadataProvider,
      PathMapper pathMapper)
      throws ExpansionException {
    VariableValue var = getNonStructuredVariable(name);
    if (var != null) {
      return var;
    }

    if (!name.contains(".")) {
      if (throwOnMissingVariable) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot find variable named '%s'.", name));
      }
      return null;
    }

    if (structuredVariableCache == null) {
      synchronized (this) {
        if (structuredVariableCache == null) {
          structuredVariableCache = Maps.newConcurrentMap();
        }
      }
    }

    Object variableOrError = structuredVariableCache.get(name);
    if (variableOrError == null) {
      try {
        VariableValue variable =
            getStructureVariable(name, throwOnMissingVariable, inputMetadataProvider, pathMapper);
        variableOrError = variable != null ? variable : NULL_MARKER;
      } catch (ExpansionException e) {
        if (throwOnMissingVariable) {
          variableOrError = e.getMessage();
        } else {
          throw new IllegalStateException(
              "Should not happen - call to getStructuredVariable threw when asked not to.", e);
        }
      }
      structuredVariableCache.putIfAbsent(name, variableOrError);
    }

    if (variableOrError instanceof VariableValue variableValue) {
      return variableValue;
    }
    if (throwOnMissingVariable) {
      throw new ExpansionException(
          variableOrError instanceof String string
              ? string
              : String.format(
                  "Invalid toolchain configuration: Cannot find variable named '%s'.", name));
    }
    return null;
  }

  @Nullable
  private VariableValue getStructureVariable(
      String name,
      boolean throwOnMissingVariable,
      @Nullable InputMetadataProvider inputMetadataProvider,
      PathMapper pathMapper)
      throws ExpansionException {
    if (!name.contains(".")) {
      return null;
    }

    Stack<String> fieldsToAccess = new Stack<>();
    String structPath = name;
    VariableValue variable;

    do {
      fieldsToAccess.push(structPath.substring(structPath.lastIndexOf('.') + 1));
      structPath = structPath.substring(0, structPath.lastIndexOf('.'));
      variable = getNonStructuredVariable(structPath);
    } while (variable == null && structPath.contains("."));

    if (variable == null) {
      return null;
    }

    while (!fieldsToAccess.empty()) {
      String field = fieldsToAccess.pop();
      variable =
          variable.getFieldValue(
              structPath, field, inputMetadataProvider, pathMapper, throwOnMissingVariable);
      if (variable == null) {
        if (throwOnMissingVariable) {
          throw new ExpansionException(
              String.format(
                  "Invalid toolchain configuration: Cannot expand variable '%s.%s': structure %s "
                      + "doesn't have a field named '%s'",
                  structPath, field, structPath, field));
        } else {
          return null;
        }
      }
    }
    return variable;
  }

  public String getStringVariable(String variableName, PathMapper pathMapper)
      throws ExpansionException {
    return getVariable(variableName, /* inputMetadataProvider= */ null, pathMapper)
        .getStringValue(variableName, pathMapper);
  }

  /** Returns whether {@code variable} is set. */
  public boolean isAvailable(String variable) {
    return isAvailable(variable, /* inputMetadataProvider= */ null);
  }

  boolean isAvailable(String variable, @Nullable InputMetadataProvider inputMetadataProvider) {
    try {
      // Availability doesn't depend on the path mapper.
      return lookupVariable(
              variable, /* throwOnMissingVariable= */ false, inputMetadataProvider, PathMapper.NOOP)
          != null;
    } catch (ExpansionException e) {
      throw new IllegalStateException(
          "Should not happen - call to lookupVariable threw when asked not to.", e);
    }
  }

  abstract Set<String> getVariableKeys();

  abstract void addVariablesToMap(Map<String, Object> variablesMap);

  @Nullable
  abstract VariableValue getNonStructuredVariable(String name);

  /**
   * Value of a build variable exposed to the CROSSTOOL used for flag expansion.
   *
   * <p>{@link VariableValue} represent either primitive values or an arbitrarily deeply nested
   * recursive structures or sequences. Since there are builds with millions of values, some
   * implementations might exist only to optimize memory usage.
   *
   * <p>Implementations must be immutable and without any side-effects. They will be expanded and
   * queried multiple times.
   */
  interface VariableValue {
    /** Returns human-readable variable type name to be used in error messages. */
    String getVariableTypeName();

    /**
     * Returns string value of the variable, if the variable type can be converted to string (e.g.
     * StringValue), or throw exception if it cannot (e.g. Sequence).
     *
     * @param variableName name of the variable value at hand, for better exception message.
     */
    String getStringValue(String variableName, PathMapper pathMapper) throws ExpansionException;

    /**
     * Returns value of the field, if the variable is of struct type or throw exception if it is not
     * or no such field exists.
     *
     * @param variableName name of the variable value at hand, for better exception message.
     */
    VariableValue getFieldValue(
        String variableName,
        String field,
        @Nullable InputMetadataProvider inputMetadataProvider,
        PathMapper pathMapper,
        boolean throwOnMissingVariable)
        throws ExpansionException;

    @VisibleForTesting
    default VariableValue getFieldValue(String variableName, String field)
        throws ExpansionException {
      return getFieldValue(
          variableName,
          field,
          /* inputMetadataProvider= */ null,
          PathMapper.NOOP,
          /* throwOnMissingVariable= */ true);
    }

    /** Returns true if the variable is truthy */
    boolean isTruthy();
  }

  /**
   * Adapter for {@link VariableValue} predefining error handling methods. Override {@link
   * #getVariableTypeName()}, {@link #isTruthy()}, and one of {@link #getFieldValue(String,
   * String)}, or {@link VariableValue#getStringValue(String, PathMapper)}, and you'll get error
   * handling for the other methods for free.
   */
  abstract static class VariableValueAdapter implements VariableValue {

    @Override
    public abstract boolean isTruthy();

    @Nullable
    @Override
    public VariableValue getFieldValue(
        String variableName,
        String field,
        @Nullable InputMetadataProvider inputMetadataProvider,
        PathMapper pathMapper,
        boolean throwOnMissingVariable)
        throws ExpansionException {
      if (throwOnMissingVariable) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s.%s': variable '%s' is "
                    + "%s, expected structure",
                variableName, field, variableName, getVariableTypeName()));
      } else {
        return null;
      }
    }

    @Override
    public String getStringValue(String variableName, PathMapper pathMapper)
        throws ExpansionException {
      throw new ExpansionException(
          String.format(
              "Invalid toolchain configuration: Cannot expand variable '%s': expected string, "
                  + "found %s",
              variableName, getVariableTypeName()));
    }
  }

  /**
   * Returns Iterable value of the variable, if the variable type can be converted to a Iterable
   * (e.g. Sequence), or throw exception if it cannot (e.g. StringValue).
   *
   * @param variableName name of the variable value at hand, for better exception message.
   */
  static Iterable<? extends VariableValue> getSequenceValue(
      String variableName, VariableValue value) throws ExpansionException {
    if (value instanceof Sequence seq) {
      return seq.getSequenceValue();
    }
    throw new ExpansionException(
        String.format(
            "Invalid toolchain configuration: Cannot expand variable '%s': expected sequence, "
                + "found %s",
            variableName, value.getVariableTypeName()));
  }

  /** Sequence of arbitrary VariableValue objects. */
  @Immutable
  static final class Sequence extends VariableValueAdapter {
    private static final String SEQUENCE_VARIABLE_TYPE_NAME = "sequence";

    private final ImmutableList<?> values;

    Sequence(ImmutableList<?> values) {
      this.values = values;
    }

    Iterable<? extends VariableValue> getSequenceValue() {
      return Iterables.transform(values, CcToolchainVariables::asVariableValue);
    }

    @Override
    public String getVariableTypeName() {
      return SEQUENCE_VARIABLE_TYPE_NAME;
    }

    @Override
    public boolean isTruthy() {
      return !values.isEmpty();
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof Sequence)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      return Objects.equals(values, ((Sequence) other).values);
    }

    @Override
    public int hashCode() {
      return values.hashCode();
    }
  }

  /**
   * Most leaves in the variable sequence node tree are simple string values. Note that this should
   * never live outside of {@code expand}, as the object overhead is prohibitively expensive.
   */
  @Immutable
  static final class StringValue extends VariableValueAdapter {
    private static final String STRING_VARIABLE_TYPE_NAME = "string";

    private final String value;

    StringValue(String value) {
      this.value = Preconditions.checkNotNull(value, "Cannot create StringValue from null");
    }

    @Override
    public String getStringValue(String variableName, PathMapper pathMapper) {
      return pathMapper.mapHeuristically(value);
    }

    @Override
    public String getVariableTypeName() {
      return STRING_VARIABLE_TYPE_NAME;
    }

    @Override
    public boolean isTruthy() {
      return !value.isEmpty();
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof StringValue)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      return Objects.equals(value, ((StringValue) other).value);
    }

    @Override
    public int hashCode() {
      return value.hashCode();
    }
  }

  @Immutable
  private static final class BooleanValue extends VariableValueAdapter {
    private static final BooleanValue TRUE = new BooleanValue(true);
    private static final BooleanValue FALSE = new BooleanValue(false);

    private static BooleanValue of(boolean value) {
      return value ? TRUE : FALSE;
    }

    private final boolean value;

    BooleanValue(boolean value) {
      this.value = value;
    }

    @Override
    public String getStringValue(String variableName, PathMapper pathMapper) {
      return value ? "1" : "0";
    }

    @Override
    public String getVariableTypeName() {
      return "boolean";
    }

    @Override
    public boolean isTruthy() {
      return value;
    }
  }

  /**
   * Represents leaves in the variable sequence node tree that are paths of artifacts. Note that
   * this should never live outside of {@code expand}, as the object overhead is prohibitively
   * expensive.
   */
  @Immutable
  private static final class ArtifactValue extends VariableValueAdapter {
    private static final String ARTIFACT_VARIABLE_TYPE_NAME = "artifact";

    private final Artifact value;

    ArtifactValue(Artifact value) {
      this.value = value;
    }

    @Override
    public String getStringValue(String variableName, PathMapper pathMapper) {
      return pathMapper.getMappedExecPathString(value);
    }

    @Override
    public String getVariableTypeName() {
      return ARTIFACT_VARIABLE_TYPE_NAME;
    }

    @Override
    public boolean isTruthy() {
      return true;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (!(other instanceof ArtifactValue otherValue)) {
        return false;
      }
      return value.equals(otherValue.value);
    }

    @Override
    public int hashCode() {
      return value.hashCode();
    }
  }

  @Immutable
  private static final class PathFragmentValue extends VariableValueAdapter {
    private static final String PATH_FRAGMENT_VARIABLE_TYPE_NAME = "pathfragment";

    private final PathFragment value;

    PathFragmentValue(PathFragment value) {
      this.value = value;
    }

    @Override
    public String getStringValue(String variableName, PathMapper pathMapper) {
      return pathMapper.map(value).getSafePathString();
    }

    @Override
    public String getVariableTypeName() {
      return PATH_FRAGMENT_VARIABLE_TYPE_NAME;
    }

    @Override
    public boolean isTruthy() {
      return true;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (!(other instanceof PathFragmentValue otherValue)) {
        return false;
      }
      return value.equals(otherValue.value);
    }

    @Override
    public int hashCode() {
      return value.hashCode();
    }
  }

  public static Builder builder() {
    return new Builder(null);
  }

  public static Builder builder(@Nullable CcToolchainVariables parent) {
    return new Builder(parent);
  }

  /** Builder for {@code Variables}. */
  // TODO(b/65472725): Forbid sequences with empty string in them.
  public static class Builder {
    private final Map<String, Object> variablesMap = new LinkedHashMap<>();
    private final CcToolchainVariables parent;

    private Builder(@Nullable CcToolchainVariables parent) {
      // private to avoid class initialization deadlock between this class and its outer class
      this.parent = parent;
    }

    @CanIgnoreReturnValue
    public Builder overrideVariable(String name, Artifact value) {
      Preconditions.checkNotNull(value, "Cannot set null as a value for variable '%s'", name);
      variablesMap.put(name, value);
      return this;
    }

    private static final Interner<ImmutableList<String>> stringSequenceInterner =
        BlazeInterners.newWeakInterner();

    /**
     * Add a sequence variable that expands {@code name} to {@code values}.
     *
     * <p>Accepts values as ImmutableSet. As ImmutableList has smaller memory footprint, we copy the
     * values into a new list.
     */
    @CanIgnoreReturnValue
    public Builder addStringSequenceVariable(String name, ImmutableSet<String> values) {
      checkVariableNotPresentAlready(name);
      Preconditions.checkNotNull(values, "Cannot set null as a value for variable '%s'", name);
      ImmutableList.Builder<String> builder = ImmutableList.builder();
      builder.addAll(values);
      variablesMap.put(name, stringSequenceInterner.intern(builder.build()));
      return this;
    }

    /**
     * Add a sequence variable that expands {@code name} to {@code values}.
     *
     * <p>Accepts values as Iterable. The iterable is stored directly, not cloned, not iterated. Be
     * mindful of memory consumption of the particular Iterable. Prefer ImmutableList, or be sure
     * that the iterable always returns the same elements in the same order, without any side
     * effects.
     */
    @CanIgnoreReturnValue
    public Builder addStringSequenceVariable(String name, Iterable<String> values) {
      checkVariableNotPresentAlready(name);
      Preconditions.checkNotNull(values, "Cannot set null as a value for variable '%s'", name);
      variablesMap.put(name, stringSequenceInterner.intern(ImmutableList.copyOf(values)));
      return this;
    }

    /** Adds a variable that expands {@code name} to the {@code value}. */
    @CanIgnoreReturnValue
    @VisibleForTesting
    public Builder addVariable(String name, Object value) {
      checkVariableNotPresentAlready(name);
      Preconditions.checkNotNull(value, "Cannot use null value for variable '%s'", name);
      variablesMap.put(name, value);
      return this;
    }

    /** Add all string variables in a map. */
    @CanIgnoreReturnValue
    public Builder addAllStringVariables(Map<String, String> variables) {
      for (String name : variables.keySet()) {
        checkVariableNotPresentAlready(name);
      }
      variablesMap.putAll(variables);
      return this;
    }

    private void checkVariableNotPresentAlready(String name) {
      Preconditions.checkNotNull(name);
      Preconditions.checkArgument(
          !variablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
    }

    /**
     * Adds all variables to this builder. Cannot override already added variables. Does not add
     * variables defined in the {@code parent} variables.
     */
    @CanIgnoreReturnValue
    public Builder addAllNonTransitive(CcToolchainVariables variables) {
      SetView<String> intersection =
          Sets.intersection(variables.getVariableKeys(), variablesMap.keySet());
      Preconditions.checkArgument(
          intersection.isEmpty(), "Cannot overwrite existing variables: %s", intersection);
      variables.addVariablesToMap(variablesMap);
      return this;
    }

    /** @return a new {@link CcToolchainVariables} object. */
    public CcToolchainVariables build() {
      if (variablesMap.size() == 1) {
        return new SingleVariables(
            parent,
            variablesMap.keySet().iterator().next(),
            variablesMap.values().iterator().next());
      }
      return new MapVariables(parent, variablesMap);
    }
  }

  /** Wraps a raw variablesMap value into an appropriate VariableValue if necessary. */
  private static VariableValue asVariableValue(Object o) {
    return switch (o) {
      case NoneType ignored -> null; // null has the same behavior as omitted field
      case Boolean b -> BooleanValue.of(b);
      case String s -> new StringValue(s);
      case Artifact artifact -> new ArtifactValue(artifact);
      case PathFragment pathFragment -> new PathFragmentValue(pathFragment);
      case Iterable<?> iterable -> new Sequence(ImmutableList.copyOf(iterable));
      case NestedSet<?> nestedSet -> new Sequence(nestedSet.toList());
      case Depset depset -> new Sequence(depset.toList());
      case Structure val -> new StarlarkStructureAdapter(val);
      default -> (VariableValue) o;
    };
  }

  /**
   * Adapts Starlark structures.
   *
   * <p>It's used to support NamedLibraryInfo, ObjectFileGroupInfo and VersionedLibraryInfo
   * structures create in {@code create_libraries_to_link_values.bzl}
   */
  public static class StarlarkStructureAdapter extends VariableValueAdapter {
    private final Structure val;

    StarlarkStructureAdapter(Structure val) {
      this.val = val;
    }

    @Nullable
    @Override
    public VariableValue getFieldValue(
        String variableName,
        String field,
        @Nullable InputMetadataProvider inputMetadataProvider,
        PathMapper pathMapper,
        boolean throwOnMissingVariable)
        throws ExpansionException {
      try {
        Object fieldValue = val.getValue(field);

        // Special handling for tree artifacts. Needed for ObjectFileGroupInfo containing a tree
        // artifact. When this code is migrated to Starlark, the expansion should happen on Starlark
        // command lines.
        if (fieldValue instanceof Iterable<?> iterable) {
          ImmutableList.Builder<Object> expandedIterable = ImmutableList.builder();
          for (Object element : iterable) {
            if (element instanceof Artifact artifact
                && artifact.isTreeArtifact()
                && inputMetadataProvider != null) {
              TreeArtifactValue treeArtifactValue = inputMetadataProvider.getTreeMetadata(artifact);
              if (treeArtifactValue != null) {
                expandedIterable.addAll(
                    Collections2.transform(
                        treeArtifactValue.getChildren(), pathMapper::getMappedExecPathString));
              }
            } else {
              expandedIterable.add(element);
            }
          }
          return new Sequence(expandedIterable.build());
        }
        return asVariableValue(fieldValue);
      } catch (EvalException e) {
        if (throwOnMissingVariable) {
          throw new ExpansionException(
              String.format(
                  "Invalid toolchain configuration: Cannot expand variable '%s.%s': variable '%s'"
                      + " is %s, expected structure",
                  variableName, field, variableName, getVariableTypeName()),
              e);
        } else {
          return null;
        }
      }
    }

    @Override
    public String getVariableTypeName() {
      return Starlark.type(val);
    }

    @Override
    public boolean isTruthy() {
      return val.truth();
    }
  }

  /**
   * A group of extra {@code Variable} instances, packaged as logic for adding to a {@code Builder}
   */
  public interface VariablesExtension {
    void addVariables(Builder builder);
  }

  static final class MapVariables extends CcToolchainVariables {
    private static final Interner<ImmutableMap<String, Integer>> keyInterner =
        BlazeInterners.newWeakInterner();

    @Nullable private final CcToolchainVariables parent;

    /**
     * This is a slightly interesting data structure that's necessary to optimize for memory
     * consumption. The premise is that a lot of compilations use the exact same variable keys, just
     * with different values. Thus, it is important to store the keys separately so that they can be
     * interned while storing the values in a compact way. keyToIndex maps from a variable name to
     * the index of the corresponding value in values.
     */
    private final ImmutableMap<String, Integer> keyToIndex;

    /** The values belonging to the keys stored in keyToIndex. */
    private final ImmutableList<Object> values;

    MapVariables(CcToolchainVariables parent, Map<String, Object> variablesMap) {
      this.parent = parent;
      ImmutableMap.Builder<String, Integer> keyBuilder = ImmutableMap.builder();
      ImmutableList.Builder<Object> valuesBuilder = ImmutableList.builder();
      int index = 0;
      for (String key : ImmutableList.sortedCopyOf(variablesMap.keySet())) {
        keyBuilder.put(key, index++);
        valuesBuilder.add(variablesMap.get(key));
      }
      this.keyToIndex = keyInterner.intern(keyBuilder.buildOrThrow());
      this.values = valuesBuilder.build();
    }

    @Override
    public boolean isImmutable() {
      return true; // immutable and Starlark-hashable
    }

    @Override
    ImmutableSet<String> getVariableKeys() {
      return keyToIndex.keySet();
    }

    @Override
    void addVariablesToMap(Map<String, Object> variablesMap) {
      for (Map.Entry<String, Integer> entry : keyToIndex.entrySet()) {
        variablesMap.put(entry.getKey(), values.get(entry.getValue()));
      }
    }

    @Nullable
    @Override
    VariableValue getNonStructuredVariable(String name) {
      if (keyToIndex.containsKey(name)) {
        return CcToolchainVariables.asVariableValue(values.get(keyToIndex.get(name)));
      }

      if (parent != null) {
        return parent.getNonStructuredVariable(name);
      }

      return null;
    }

    /**
     * NB: this compares parents using reference equality instead of logical equality.
     *
     * <p>This is a performance optimization to avoid possibly expensive recursive equality
     * expansions and suitable for comparisons needed by interning deserialized values. If full
     * logical equality is desired, it's possible to either enable full interning (at a modest CPU
     * cost) or change the parent comparison to use deep equality.
     *
     * <p>This same comment applies to {@link SingleVariables#equals}.
     */
    @Override
    public boolean equals(Object other) {
      if (!(other instanceof MapVariables that)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      if (this.parent != that.parent) {
        return false;
      }
      return Objects.equals(this.keyToIndex, that.keyToIndex)
          && Objects.equals(this.values, that.values);
    }

    @Override
    public int hashCode() {
      return 31 * Objects.hash(keyToIndex, values) + System.identityHashCode(parent);
    }
  }

  static final class SingleVariables extends CcToolchainVariables {
    @Nullable private final CcToolchainVariables parent;
    private final String name;
    private final Object variableValue;

    SingleVariables(CcToolchainVariables parent, String name, Object variableValue) {
      this.parent = parent;
      this.name = name;
      this.variableValue = variableValue;
    }

    @Override
    ImmutableSet<String> getVariableKeys() {
      return ImmutableSet.of(name);
    }

    @Override
    void addVariablesToMap(Map<String, Object> variablesMap) {
      variablesMap.put(name, variableValue);
    }

    @Nullable
    @Override
    VariableValue getNonStructuredVariable(String name) {
      if (this.name.equals(name)) {
        return CcToolchainVariables.asVariableValue(variableValue);
      }
      return parent == null ? null : parent.getNonStructuredVariable(name);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof SingleVariables that)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      if (this.parent != that.parent) {
        return false;
      }
      return Objects.equals(this.name, that.name)
          && Objects.equals(this.variableValue, that.variableValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(parent, name, variableValue);
    }
  }
}
