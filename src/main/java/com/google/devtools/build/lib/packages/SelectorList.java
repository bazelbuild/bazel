// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.docgen.annot.GlobalMethods;
import com.google.devtools.build.docgen.annot.GlobalMethods.Environment;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.HasBinary;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.TokenKind;

/**
 * An attribute value consisting of a concatenation (via the {@code +} operator for lists or the
 * {@code |} operator for dicts) of native types and selects, e.g:
 *
 * <pre>
 *   rule(
 *       name = 'myrule',
 *       deps =
 *           [':defaultdep']
 *           + select({
 *               'a': [':adep'],
 *               'b': [':bdep'],})
 *           + select({
 *               'c': [':cdep'],
 *               'd': [':ddep'],})
 *   )
 * </pre>
 */
@StarlarkBuiltin(
    name = "select",
    doc = "A selector between configuration-dependent entities.",
    documented = false)
public final class SelectorList implements StarlarkValue, HasBinary {

  // TODO(adonovan): combine Selector{List,Value} and BuildType.SelectorList.
  // We don't need three classes for the same concept

  private final Class<?> type;
  private final List<Object> elements;

  private SelectorList(Class<?> type, List<Object> elements) {
    this.type = type;
    this.elements = elements;
  }

  /**
   * Returns an ordered list of the elements in this expression. Each element may be a native type
   * or a select.
   */
  List<Object> getElements() {
    return elements;
  }

  /** Returns the native type contained by this expression. */
  Class<?> getType() {
    return type;
  }

  /** Implementation of the Starlark {@code select()} function exposed to BUILD and .bzl files. */
  private static Object select(
      Dict<?, ?> dict,
      String noMatchError,
      @Nullable Label.PackageContext packageContext,
      @Nullable Label.RepoMappingRecorder repoMappingRecorder)
      throws EvalException, LabelSyntaxException {
    if (dict.isEmpty()) {
      throw Starlark.errorf(
          "select({}) with an empty dictionary can never resolve because it includes no conditions"
              + " to match");
    }
    var selectDict = ImmutableMap.builderWithExpectedSize(dict.size());
    for (var entry : dict.entrySet()) {
      switch (entry.getKey()) {
        case Label label -> selectDict.put(label, entry.getValue());
        case String labelString ->
            selectDict.put(
                packageContext != null
                    ? Label.parseWithPackageContext(
                        labelString, packageContext, repoMappingRecorder)
                    : labelString,
                entry.getValue());
        default ->
            throw Starlark.errorf(
                "select: got %s for dict key, want a Label or label string",
                Starlark.type(entry.getKey()));
      }
    }
    return SelectorList.of(new SelectorValue(selectDict.buildOrThrow(), noMatchError));
  }

  /** Creates a "wrapper" list that consists of a single select. */
  static SelectorList of(SelectorValue selector) {
    return new SelectorList(selector.getType(), ImmutableList.of(selector));
  }

  /**
   * Creates a list from the given sequence of values, which must be non-empty. Each value may be a
   * native type, a select over that type, or a selector list over that type.
   *
   * @throws EvalException if all values don't have the same underlying type
   */
  static SelectorList of(Iterable<?> values) throws EvalException {
    Preconditions.checkArgument(!Iterables.isEmpty(values));
    ImmutableList.Builder<Object> elements = ImmutableList.builder();
    Object firstValue = null;

    for (Object value : values) {
      if (value instanceof SelectorList) {
        elements.addAll(((SelectorList) value).elements);
      } else {
        elements.add(value);
      }
      if (firstValue == null) {
        firstValue = value;
      }
      if (!canConcatenate(getNativeType(firstValue), getNativeType(value))) {
        throw Starlark.errorf(
            "Cannot combine incompatible types (%s, %s)",
            getTypeName(firstValue), getTypeName(value));
      }
    }

    return new SelectorList(getNativeType(firstValue), elements.build());
  }

  /**
   * Wraps a single value in a {@code select()} where the default condition maps to the given value
   */
  public static SelectorList wrapSingleValue(Object obj) {
    return SelectorList.of(
        new SelectorValue(ImmutableMap.of(BuildType.Selector.DEFAULT_CONDITION_KEY, obj), ""));
  }

  /**
   * Creates a list that concatenates two values, where each value may be a native type, a select
   * over that type, or a selector list over that type.
   *
   * @throws EvalException if the values don't have the same underlying type
   */
  static SelectorList concat(Object x, Object y) throws EvalException {
    return of(Arrays.asList(x, y));
  }

  private static TokenKind binaryOpToken(Object value) {
    return getNativeType(value).equals(Dict.class) ? TokenKind.PIPE : TokenKind.PLUS;
  }

  @Override
  @Nullable
  public SelectorList binaryOp(TokenKind op, Object that, boolean thisLeft) throws EvalException {
    if (op == binaryOpToken(that)) {
      return thisLeft ? concat(this, that) : concat(that, this);
    }
    return null;
  }

  private static final Class<?> NATIVE_LIST_TYPE = List.class;

  private static String getTypeName(Object x) {
    if (x instanceof SelectorList) {
      return "select of " + Depset.ElementType.of(((SelectorList) x).type);
    } else if (x instanceof SelectorValue) {
      return "select of " + Depset.ElementType.of(((SelectorValue) x).getType());
    } else {
      return Starlark.type(x);
    }
  }

  private static Class<?> getNativeType(Object value) {
    if (value instanceof SelectorList) {
      return ((SelectorList) value).type;
    } else if (value instanceof SelectorValue) {
      return ((SelectorValue) value).getType();
    } else {
      return value.getClass();
    }
  }

  private static boolean isListType(Class<?> type) {
    return NATIVE_LIST_TYPE.isAssignableFrom(type);
  }

  private static boolean canConcatenate(Class<?> type1, Class<?> type2) {
    if (type1 == type2) {
      return true;
    }
    return isListType(type1) && isListType(type2);
  }

  @Override
  public String toString() {
    return Starlark.repr(this);
  }

  @Override
  public void repr(Printer printer) {
    printer.printList(elements, "", String.format(" %s ", binaryOpToken(this)), "");
  }

  @Override
  public int hashCode() {
    return Objects.hash(type, elements);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof SelectorList)) {
      return false;
    }
    SelectorList that = (SelectorList) other;
    return Objects.equals(this.type, that.type) && Objects.equals(this.elements, that.elements);
  }

  /** The user-facing API to the {@code select()} callable. */
  @GlobalMethods(environment = {Environment.BUILD, Environment.BZL})
  public static final class SelectLibrary {

    private SelectLibrary() {}

    public static final SelectLibrary INSTANCE = new SelectLibrary();

    @StarlarkMethod(
        name = "select",
        doc =
            "<code>select()</code> is the helper function that makes a rule attribute "
                + "<a href=\"${link common-definitions#configurable-attributes}\">"
                + "configurable</a>. See "
                + "<a href=\"${link functions#select}\">build encyclopedia</a> for details.",
        parameters = {
          @Param(
              name = "x",
              positional = true,
              doc =
                  "A dict that maps configuration conditions to values. Each key is a "
                      + "<a href=\"../builtins/Label.html\">Label</a> or a label string"
                      + " that identifies a config_setting or constraint_value instance. See the"
                      + " <a href=\"https://bazel.build/rules/macros#label-resolution\">"
                      + "documentation on macros</a> for when to use a Label instead of a string."),
          @Param(
              name = "no_match_error",
              defaultValue = "''",
              doc = "Optional custom error to report if no condition matches.",
              named = true),
        },
        useStarlarkThread = true)
    public Object select(Dict<?, ?> dict, String noMatchError, StarlarkThread thread)
        throws EvalException {
      // If this is not null, string keys in the dict will be resolved to Labels eagerly using the
      // given context. This is unnecessary for BUILD files and packageContext will remain null in
      // this case.
      Label.PackageContext packageContext = null;
      if (thread
          .getSemantics()
          .getBool(BuildLanguageOptions.INCOMPATIBLE_RESOLVE_SELECT_KEYS_EAGERLY)) {
        var module = Module.ofInnermostEnclosingStarlarkFunction(thread);
        if (module != null) {
          var ctx = BazelModuleContext.of(module);
          if (ctx != null) {
            packageContext = ctx.packageContext();
          }
        }
      }
      try {
        return SelectorList.select(
            dict,
            noMatchError,
            packageContext,
            // select() is not usually called in a repository rule or a module extension, but its
            // string representation does provide a way to observe the repo mapping.
            thread.getThreadLocal(Label.RepoMappingRecorder.class));
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf("invalid label in select(): %s", e.getMessage());
      }
    }
  }
}
