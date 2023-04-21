// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlarkbuildapi.core;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/** Interface for provider objects (constructors for {@link StructApi} objects). */
@StarlarkBuiltin(
    name = "Provider",
    category = DocCategory.BUILTIN,
    doc =
        "A constructor for simple value objects, known as provider instances."
            + "<br>"
            + "This value has a dual purpose:"
            + "  <ul>"
            + "     <li>It is a function that can be called to construct 'struct'-like values:"
            + "<pre class=\"language-python\">DataInfo = provider()\n"
            + "d = DataInfo(x = 2, y = 3)\n"
            + "print(d.x + d.y) # prints 5</pre>"
            + "     Note: Some providers, defined internally, do not allow instance creation"
            + "     </li>"
            + "     <li>It is a <i>key</i> to access a provider instance on a"
            + "        <a href=\"../builtins/Target.html\">Target</a>"
            + "<pre class=\"language-python\">DataInfo = provider()\n"
            + "def _rule_impl(ctx)\n"
            + "  ... ctx.attr.dep[DataInfo]</pre>"
            + "     </li>"
            + "  </ul>"
            + "Create a new <code>Provider</code> using the "
            + "<a href=\"../globals/bzl.html#provider\">provider</a> function.")
public interface ProviderApi extends StarlarkValue {}
