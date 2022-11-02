// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Template expansion dict module. */
@StarlarkBuiltin(
    name = "TemplateDict",
    category = DocCategory.TOP_LEVEL_TYPE,
    doc =
        "An Args-like structure for use in ctx.actions.expand_template(), which allows for"
            + " deferring evaluation of values till the execution phase.")
public interface TemplateDictApi extends StarlarkValue {

  @StarlarkMethod(
      name = "add",
      doc = "Add a String value",
      parameters = {
        @Param(name = "key", doc = "A String key"),
        @Param(name = "value", doc = "A String value")
      },
      useStarlarkThread = true)
  TemplateDictApi addArgument(String key, String value, StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "add_joined",
      doc = "Add depset of values",
      parameters = {
        @Param(name = "key", doc = "A String key"),
        @Param(
            name = "values",
            allowedTypes = {
              @ParamType(type = Depset.class),
            },
            doc = "The depset whose items will be joined."),
        @Param(
            name = "join_with",
            named = true,
            positional = false,
            doc =
                "A delimiter string used to join together the strings obtained from applying "
                    + "<code>map_each</code>, in the same manner as "
                    + "<a href='string.html#join'><code>string.join()</code></a>."),
        @Param(
            name = "map_each",
            allowedTypes = {
              @ParamType(type = StarlarkCallable.class),
            },
            named = true,
            positional = false,
            doc =
                "A Starlark function accepting a single argument and returning a String. This"
                    + " function is applied to each item of the depset specified in the"
                    + " <code>values</code> parameter"),
        @Param(
            name = "uniquify",
            named = true,
            positional = false,
            defaultValue = "False",
            doc =
                "If true, duplicate strings derived from <code>values</code> will be omitted. Only "
                    + "the first occurrence of each string will remain. Usually this feature is "
                    + "not needed because depsets already omit duplicates, but it can be useful "
                    + "if <code>map_each</code> emits the same string for multiple items."),
      },
      useStarlarkThread = true)
  TemplateDictApi addJoined(
      String key,
      Depset values,
      String joinWith,
      StarlarkCallable mapEach,
      Boolean uniquify,
      StarlarkThread thread)
      throws EvalException;
}
