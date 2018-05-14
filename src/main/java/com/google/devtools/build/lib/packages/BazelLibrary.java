// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment.GlobalFrame;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.MethodLibrary;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SelectorList;
import com.google.devtools.build.lib.syntax.SelectorValue;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.syntax.SkylarkType;

/**
 * A helper class containing additional built in functions for Bazel (BUILD files and .bzl files).
 */
public class BazelLibrary {

  // TODO(bazel-team): Move to MethodLibrary alongside other pure-Skylark builtins.
  @SkylarkSignature(
    name = "type",
    returnType = String.class,
    doc =
        "Returns the type name of its argument. This is useful for debugging and "
            + "type-checking. Examples:"
            + "<pre class=\"language-python\">"
            + "type(2) == \"int\"\n"
            + "type([1]) == \"list\"\n"
            + "type(struct(a = 2)) == \"struct\""
            + "</pre>"
            + "This function might change in the future. To write Python-compatible code and "
            + "be future-proof, use it only to compare return values: "
            + "<pre class=\"language-python\">"
            + "if type(x) == type([]):  # if x is a list"
            + "</pre>",
    parameters = {@Param(name = "x", doc = "The object to check type of.")}
  )
  private static final BuiltinFunction type =
      new BuiltinFunction("type") {
        public String invoke(Object object) {
          // There is no 'type' type in Skylark, so we return a string with the type name.
          return EvalUtils.getDataTypeName(object, false);
        }
      };

  @SkylarkSignature(
    name = "depset",
    returnType = SkylarkNestedSet.class,
    doc =
        "Creates a <a href=\"depset.html\">depset</a>. The <code>direct</code> parameter is a list "
            + "of direct elements of the depset, and <code>transitive</code> parameter is "
            + "a list of depsets whose elements become indirect elements of the created depset. "
            + "The order in which elements are returned when the depset is converted to a list "
            + "is specified by the <code>order</code> parameter. "
            + "See the <a href=\"../depsets.md\">Depsets overview</a> for more information. "
            + "<p> All elements (direct and indirect) of a depset must be of the same type. "
            + "<p> The order of the created depset should be <i>compatible</i> with the order of "
            + "its <code>transitive</code> depsets. <code>\"default\"</code> order is compatible "
            + "with any other order, all other orders are only compatible with themselves."
            + "<p> Note on backward/forward compatibility. This function currently accepts a "
            + "positional <code>items</code> parameter. It is deprecated and will be removed "
            + "in the future, and after its removal <code>direct</code> will become a sole "
            + "positional parameter of the <code>depset</code> function. Thus, both of the "
            + "following calls are equivalent and future-proof:<br>"
            + "<pre class=language-python>"
            + "depset(['a', 'b'], transitive = [...])\n"
            + "depset(direct = ['a', 'b'], transitive = [...])\n"
            + "</pre>",
    parameters = {
      @Param(
        name = "items",
        type = Object.class,
        defaultValue = "[]",
        doc =
            "Deprecated: Either an iterable whose items become the direct elements of "
                + "the new depset, in left-to-right order, or else a depset that becomes "
                + "a transitive element of the new depset. In the latter case, "
                + "<code>transitive</code> cannot be specified."
      ),
      @Param(
        name = "order",
        type = String.class,
        defaultValue = "\"default\"",
        doc =
            "The traversal strategy for the new depset. See <a href=\"depset.html\">here</a> for "
                + "the possible values."
      ),
      @Param(
        name = "direct",
        type = SkylarkList.class,
        defaultValue = "None",
        positional = false,
        named = true,
        noneable = true,
        doc = "A list of <i>direct</i> elements of a depset."
      ),
      @Param(
        name = "transitive",
        named = true,
        positional = false,
        type = SkylarkList.class,
        generic1 = SkylarkNestedSet.class,
        noneable = true,
        doc = "A list of depsets whose elements will become indirect elements of the depset.",
        defaultValue = "None"
      )
    },
    useLocation = true
  )
  private static final BuiltinFunction depset =
      new BuiltinFunction("depset") {
        public SkylarkNestedSet invoke(
            Object items, String orderString, Object direct, Object transitive, Location loc)
            throws EvalException {
          Order order;
          try {
            order = Order.parse(orderString);
          } catch (IllegalArgumentException ex) {
            throw new EvalException(loc, ex);
          }

          if (transitive == Runtime.NONE && direct == Runtime.NONE) {
            // Legacy behavior.
            return SkylarkNestedSet.of(order, items, loc);
          }

          if (direct != Runtime.NONE && !isEmptySkylarkList(items)) {
            throw new EvalException(
                loc, "Do not pass both 'direct' and 'items' argument to depset constructor.");
          }

          // Non-legacy behavior: either 'transitive' or 'direct' were specified.
          Iterable<Object> directElements;
          if (direct != Runtime.NONE) {
            directElements = ((SkylarkList<?>) direct).getContents(Object.class, "direct");
          } else {
            SkylarkType.checkType(items, SkylarkList.class, "items");
            directElements = ((SkylarkList<?>) items).getContents(Object.class, "items");
          }

          Iterable<SkylarkNestedSet> transitiveList;
          if (transitive != Runtime.NONE) {
            SkylarkType.checkType(transitive, SkylarkList.class, "transitive");
            transitiveList =
                ((SkylarkList<?>) transitive).getContents(SkylarkNestedSet.class, "transitive");
          } else {
            transitiveList = ImmutableList.of();
          }
          SkylarkNestedSet.Builder builder = SkylarkNestedSet.builder(order, loc);
          for (Object directElement : directElements) {
            builder.addDirect(directElement);
          }
          for (SkylarkNestedSet transitiveSet : transitiveList) {
            builder.addTransitive(transitiveSet);
          }
          return builder.build();
        }
      };

  private static boolean isEmptySkylarkList(Object o) {
    return o instanceof SkylarkList && ((SkylarkList) o).isEmpty();
  }

  /**
   * Returns a function-value implementing "select" (i.e. configurable attributes) in the specified
   * package context.
   */
  @SkylarkSignature(
    name = "select",
    doc =
        "<code>select()</code> is the helper function that makes a rule attribute "
            + "<a href=\"$BE_ROOT/common-definitions.html#configurable-attributes\">configurable</a>. "
            + "See <a href=\"$BE_ROOT/functions.html#select\">build encyclopedia</a> for details.",
    parameters = {
      @Param(name = "x", type = SkylarkDict.class, doc = "The parameter to convert."),
      @Param(
        name = "no_match_error",
        type = String.class,
        defaultValue = "''",
        doc = "Optional custom error to report if no condition matches."
      )
    },
    useLocation = true
  )
  private static final BuiltinFunction select =
      new BuiltinFunction("select") {
        public Object invoke(SkylarkDict<?, ?> dict, String noMatchError, Location loc)
            throws EvalException {
          for (Object key : dict.keySet()) {
            if (!(key instanceof String)) {
              throw new EvalException(
                  loc, String.format("Invalid key: %s. select keys must be label references", key));
            }
          }
          return SelectorList.of(new SelectorValue(dict, noMatchError));
        }
      };

  /** Adds bindings for all the builtin functions of this class to the given map builder. */
  public static void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    for (BaseFunction function : allFunctions) {
      builder.put(function.getName(), function);
    }
  }

  private static final ImmutableList<BaseFunction> allFunctions =
      ImmutableList.of(select, depset, type);

  /** A global frame containing pure Skylark builtins and some Bazel builtins. */
  public static final GlobalFrame GLOBALS = createGlobals();

  private static GlobalFrame createGlobals() {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
    Runtime.addConstantsToBuilder(builder);
    MethodLibrary.addBindingsToBuilder(builder);
    BazelLibrary.addBindingsToBuilder(builder);
    return GlobalFrame.createForBuiltins(builder.build());
  }

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(BazelLibrary.class);
  }
}
