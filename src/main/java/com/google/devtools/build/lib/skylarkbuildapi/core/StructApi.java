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

package com.google.devtools.build.lib.skylarkbuildapi.core;

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Interface for the "struct" object in the build API. */
@SkylarkModule(
    name = "struct",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "A generic object with fields."
            + "<p>Structs fields cannot be reassigned once the struct is created. Two structs are "
            + "equal if they have the same fields and if corresponding field values are equal.")
public interface StructApi extends StarlarkValue {

  @SkylarkCallable(
      name = "to_proto",
      doc =
          "Creates a text message from the struct parameter. This method only works if all "
              + "struct elements (recursively) are strings, ints, booleans, "
              + "other structs or dicts or lists of these types. "
              + "Quotes and new lines in strings are escaped. "
              + "Struct keys are iterated in the sorted order. "
              + "Examples:<br><pre class=language-python>"
              + "struct(key=123).to_proto()\n# key: 123\n\n"
              + "struct(key=True).to_proto()\n# key: true\n\n"
              + "struct(key=[1, 2, 3]).to_proto()\n# key: 1\n# key: 2\n# key: 3\n\n"
              + "struct(key='text').to_proto()\n# key: \"text\"\n\n"
              + "struct(key=struct(inner_key='text')).to_proto()\n"
              + "# key {\n#   inner_key: \"text\"\n# }\n\n"
              + "struct(key=[struct(inner_key=1), struct(inner_key=2)]).to_proto()\n"
              + "# key {\n#   inner_key: 1\n# }\n# key {\n#   inner_key: 2\n# }\n\n"
              + "struct(key=struct(inner_key=struct(inner_inner_key='text'))).to_proto()\n"
              + "# key {\n#    inner_key {\n#     inner_inner_key: \"text\"\n#   }\n# }\n\n"
              + "struct(foo={4: 3, 2: 1}).to_proto()\n"
              + "# foo: {\n"
              + "#   key: 4\n"
              + "#   value: 3\n"
              + "# }\n"
              + "# foo: {\n"
              + "#   key: 2\n"
              + "#   value: 1\n"
              + "# }\n"
              + "</pre>")
  String toProto() throws EvalException;

  @SkylarkCallable(
      name = "to_json",
      doc =
          "Creates a JSON string from the struct parameter. This method only works if all "
              + "struct elements (recursively) are strings, ints, booleans, other structs, a "
              + "list of these types or a dictionary with string keys and values of these types. "
              + "Quotes and new lines in strings are escaped. "
              + "Examples:<br><pre class=language-python>"
              + "struct(key=123).to_json()\n# {\"key\":123}\n\n"
              + "struct(key=True).to_json()\n# {\"key\":true}\n\n"
              + "struct(key=[1, 2, 3]).to_json()\n# {\"key\":[1,2,3]}\n\n"
              + "struct(key='text').to_json()\n# {\"key\":\"text\"}\n\n"
              + "struct(key=struct(inner_key='text')).to_json()\n"
              + "# {\"key\":{\"inner_key\":\"text\"}}\n\n"
              + "struct(key=[struct(inner_key=1), struct(inner_key=2)]).to_json()\n"
              + "# {\"key\":[{\"inner_key\":1},{\"inner_key\":2}]}\n\n"
              + "struct(key=struct(inner_key=struct(inner_inner_key='text'))).to_json()\n"
              + "# {\"key\":{\"inner_key\":{\"inner_inner_key\":\"text\"}}}\n</pre>")
  String toJson() throws EvalException;

  /** Callable Provider for new struct objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  interface StructProviderApi extends ProviderApi {

    @SkylarkCallable(
        name = "struct",
        doc =
            "Creates an immutable struct using the keyword arguments as attributes. It is used to "
                + "group multiple values together. Example:<br>"
                + "<pre class=\"language-python\">s = struct(x = 2, y = 3)\n"
                + "return s.x + getattr(s, \"y\")  # returns 5</pre>",
        extraKeywords =
            @Param(
                name = "kwargs",
                type = Dict.class,
                defaultValue = "{}",
                doc = "Dictionary of arguments."),
        useStarlarkThread = true,
        selfCall = true)
    @SkylarkConstructor(objectType = StructApi.class, receiverNameForDoc = "struct")
    StructApi createStruct(Dict<String, Object> kwargs, StarlarkThread thread) throws EvalException;
  }
}
