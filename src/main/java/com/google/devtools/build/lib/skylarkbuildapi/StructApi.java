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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.EvalException;

/** Interface for the "struct" object in the build API. */
@SkylarkModule(
    name = "struct",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "A generic object with fields. See the global <a href=\"globals.html#struct\"><code>struct"
            + "</code></a> function for more details."
            + "<p>Structs fields cannot be reassigned once the struct is created. Two structs are "
            + "equal if they have the same fields and if corresponding field values are equal.")
public interface StructApi extends SkylarkValue {

  @SkylarkCallable(
      name = "to_proto",
      doc =
          "Creates a text message from the struct parameter. This method only works if all "
              + "struct elements (recursively) are strings, ints, booleans, other structs or a "
              + "list of these types. Quotes and new lines in strings are escaped. "
              + "Keys are iterated in the sorted order. "
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
              + "# key {\n#    inner_key {\n#     inner_inner_key: \"text\"\n#   }\n# }\n</pre>",
      useLocation = true)
  public String toProto(Location loc) throws EvalException;

  @SkylarkCallable(
      name = "to_json",
      doc =
          "Creates a JSON string from the struct parameter. This method only works if all "
              + "struct elements (recursively) are strings, ints, booleans, other structs or a "
              + "list of these types. Quotes and new lines in strings are escaped. "
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
              + "# {\"key\":{\"inner_key\":{\"inner_inner_key\":\"text\"}}}\n</pre>",
      useLocation = true)
  public String toJson(Location loc) throws EvalException;
}
