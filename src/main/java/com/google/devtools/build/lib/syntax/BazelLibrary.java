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

package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import java.util.List;

/**
 * A helper class containing additional built in functions for Bazel (BUILD files and .bzl files).
 */
public class BazelLibrary {

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
        "Creates a <a href=\"depset.html\">depset</a>. In the case that <code>items</code> is an "
            + "iterable, its contents become the direct elements of the depset, with their left-to-"
            + "right order preserved, and the depset has no transitive elements. In the case that "
            + "<code>items</code> is a depset, it is made the sole transitive element of the new "
            + "depset, and no direct elements are added. In the second case the given depset's "
            + "order must match the <code>order</code> param or else one of the two must be <code>"
            + "\"default\"</code>. See the <a href=\"../depsets.md\">Depsets overview</a> for more "
            + "information.",
    parameters = {
      @Param(
        name = "items",
        type = Object.class,
        defaultValue = "[]",
        doc =
            "An iterable whose items become the direct elements of the new depset, in left-to-"
                + "right order; or alternatively, a depset that becomes the transitive element of "
                + "the new depset."
      ),
      @Param(
        name = "order",
        type = String.class,
        defaultValue = "\"default\"",
        doc =
            "The traversal strategy for the new depset. See <a href=\"depset.html\">here</a> for "
                + "the possible values."
      )
    },
    useLocation = true
  )
  private static final BuiltinFunction depset =
      new BuiltinFunction("depset") {
        public SkylarkNestedSet invoke(Object items, String order, Location loc)
            throws EvalException {
          try {
            return new SkylarkNestedSet(Order.parse(order), items, loc);
          } catch (IllegalArgumentException ex) {
            throw new EvalException(loc, ex);
          }
        }
      };

  @SkylarkSignature(
    name = "set",
    returnType = SkylarkNestedSet.class,
    documentationReturnType = SkylarkNestedSet.LegacySet.class,
    doc =
        "A temporary alias for <a href=\"#depset\">depset</a>. "
            + "Deprecated in favor of <code>depset</code>.",
    parameters = {
      @Param(
        name = "items",
        type = Object.class,
        defaultValue = "[]",
        doc = "Same as for <a href=\"#depset\">depset</a>."
      ),
      @Param(
        name = "order",
        type = String.class,
        defaultValue = "\"default\"",
        doc = "Same as for <a href=\"#depset\">depset</a>."
      )
    },
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction set =
      new BuiltinFunction("set") {
        public SkylarkNestedSet invoke(Object items, String order, Location loc, Environment env)
            throws EvalException {
          if (env.getSemantics().incompatibleDepsetConstructor) {
            throw new EvalException(
                loc,
                "The `set` constructor for depsets is deprecated and will be removed. Please use "
                    + "the `depset` constructor instead. You can temporarily enable the "
                    + "deprecated `set` constructor by passing the flag "
                    + "--incompatible_depset_constructor=false");
          }
          try {
            return new SkylarkNestedSet(Order.parse(order), items, loc);
          } catch (IllegalArgumentException ex) {
            throw new EvalException(loc, ex);
          }
        }
      };

  @SkylarkSignature(
    name = "union",
    objectType = SkylarkNestedSet.class,
    returnType = SkylarkNestedSet.class,
    doc =
        "<i>(Deprecated)</i> Returns a new <a href=\"depset.html\">depset</a> that is the merge "
            + "of the given depset and <code>new_elements</code>. This is the same as the <code>+"
            + "</code> operator.",
    parameters = {
      @Param(name = "input", type = SkylarkNestedSet.class, doc = "The input depset."),
      @Param(name = "new_elements", type = Object.class, doc = "The elements to be added.")
    },
    useLocation = true
  )
  private static final BuiltinFunction union =
      new BuiltinFunction("union") {
        @SuppressWarnings("unused")
        public SkylarkNestedSet invoke(SkylarkNestedSet input, Object newElements, Location loc)
            throws EvalException {
          // newElements' type is Object because of the polymorphism on unioning two
          // SkylarkNestedSets versus a set and another kind of iterable.
          // Can't use EvalUtils#toIterable since that would discard this information.
          return new SkylarkNestedSet(input, newElements, loc);
        }
      };

  @SkylarkSignature(
    name = "to_list",
    objectType = SkylarkNestedSet.class,
    returnType = MutableList.class,
    doc =
        "Returns a list of the elements, without duplicates, in the depset's traversal order. "
            + "Note that order is unspecified (but deterministic) for elements that were added "
            + "more than once to the depset. Order is also unspecified for <code>\"default\""
            + "</code>-ordered depsets, and for elements of child depsets whose order differs "
            + "from that of the parent depset. The list is a copy; modifying it has no effect "
            + "on the depset and vice versa.",
    parameters = {@Param(name = "input", type = SkylarkNestedSet.class, doc = "The input depset.")},
    useEnvironment = true
  )
  private static final BuiltinFunction toList =
      new BuiltinFunction("to_list") {
        @SuppressWarnings("unused")
        public MutableList<Object> invoke(SkylarkNestedSet input, Environment env) {
          return new MutableList<>(input.toCollection(), env);
        }
      };

  /**
   * Returns a function-value implementing "select" (i.e. configurable attributes) in the specified
   * package context.
   */
  @SkylarkSignature(
    name = "select",
    doc = "Creates a SelectorValue from the dict parameter.",
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

  private static Environment.Frame createGlobals() {
    List<BaseFunction> bazelGlobalFunctions =
        ImmutableList.<BaseFunction>of(select, depset, set, type);

    try (Mutability mutability = Mutability.create("BUILD")) {
      Environment env = Environment.builder(mutability).build();
      Runtime.setupConstants(env);
      Runtime.setupMethodEnvironment(env, MethodLibrary.defaultGlobalFunctions);
      Runtime.setupMethodEnvironment(env, bazelGlobalFunctions);
      return env.getGlobals();
    }
  }

  public static final Environment.Frame GLOBALS = createGlobals();

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(BazelLibrary.class);
  }
}
