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
        "Creates a <a href=\"depset.html\">depset</a> from the <code>items</code>. "
            + "The depset supports nesting other depsets of the same element type in it. "
            + "A desired <a href=\"depset.html\">iteration order</a> can also be specified.<br>"
            + "Examples:<br><pre class=\"language-python\">depset([\"a\", \"b\"])\n"
            + "depset([1, 2, 3], order=\"postorder\")</pre>",
    parameters = {
      @Param(
        name = "items",
        type = Object.class,
        defaultValue = "[]",
        doc =
            "The items to initialize the depset with. May contain both standalone items "
                + "and other depsets."
      ),
      @Param(
        name = "order",
        type = String.class,
        defaultValue = "\"default\"",
        doc =
            "The ordering strategy for the depset. Possible values are: <code>default</code> "
                + "(default), <code>postorder</code>, <code>topological</code>, and "
                + "<code>preorder</code>. These are also known by the deprecated names "
                + "<code>stable</code>, <code>compile</code>, <code>link</code> and "
                + "<code>naive_link</code> respectively. An explanation of the values can be found "
                + "<a href=\"depset.html\">here</a>."
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
    useLocation = true
  )
  private static final BuiltinFunction set =
      new BuiltinFunction("set") {
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
    name = "union",
    objectType = SkylarkNestedSet.class,
    returnType = SkylarkNestedSet.class,
    doc =
        "Creates a new <a href=\"depset.html\">depset</a> that contains both "
            + "the input depset as well as all additional elements.",
    parameters = {
      @Param(name = "input", type = SkylarkNestedSet.class, doc = "The input depset."),
      @Param(name = "new_elements", type = Object.class, doc = "The elements to be added.")
    },
    useLocation = true
  )
  private static final BuiltinFunction union =
      new BuiltinFunction("union") {
        @SuppressWarnings("unused")
        public SkylarkNestedSet invoke(
            SkylarkNestedSet input, Object newElements, Location loc)
            throws EvalException {
          // newElements' type is Object because of the polymorphism on unioning two
          // SkylarkNestedSets versus a set and another kind of iterable.
          // Can't use EvalUtils#toIterable since that would discard this information.
          return new SkylarkNestedSet(input, newElements, loc);
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
              throw new EvalException(loc,
                  String.format("Invalid key: %s. select keys must be label references", key));
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
