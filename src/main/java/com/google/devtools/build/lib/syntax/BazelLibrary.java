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
    name = "set",
    returnType = SkylarkNestedSet.class,
    doc =
        "Creates a <a href=\"set.html\">set</a> from the <code>items</code>. "
            + "The set supports nesting other sets of the same element type in it. "
            + "A desired <a href=\"set.html\">iteration order</a> can also be specified.<br>"
            + "Examples:<br><pre class=\"language-python\">set([\"a\", \"b\"])\n"
            + "set([1, 2, 3], order=\"compile\")</pre>",
    parameters = {
      @Param(
        name = "items",
        type = Object.class,
        defaultValue = "[]",
        doc =
            "The items to initialize the set with. May contain both standalone items "
                + "and other sets."
      ),
      @Param(
        name = "order",
        type = String.class,
        defaultValue = "\"stable\"",
        doc =
            "The ordering strategy for the set if it's nested, "
                + "possible values are: <code>stable</code> (default), <code>compile</code>, "
                + "<code>link</code> or <code>naive_link</code>. An explanation of the "
                + "values can be found <a href=\"set.html\">here</a>."
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
        "Creates a new <a href=\"set.html\">set</a> that contains both "
            + "the input set as well as all additional elements.",
    parameters = {
      @Param(name = "input", type = SkylarkNestedSet.class, doc = "The input set"),
      @Param(name = "new_elements", type = Iterable.class, doc = "The elements to be added")
    },
    useLocation = true
  )
  private static final BuiltinFunction union =
      new BuiltinFunction("union") {
        @SuppressWarnings("unused")
        public SkylarkNestedSet invoke(
            SkylarkNestedSet input, Iterable<Object> newElements, Location loc)
            throws EvalException {
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
    }
  )
  private static final BuiltinFunction select =
      new BuiltinFunction("select") {
        public Object invoke(SkylarkDict<?, ?> dict, String noMatchError) throws EvalException {
          return SelectorList.of(new SelectorValue(dict, noMatchError));
        }
      };

  private static Environment.Frame createGlobals() {
    List<BaseFunction> bazelGlobalFunctions = ImmutableList.<BaseFunction>of(select, set, type);

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
