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

import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkDocumentationCategory;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Represents a collection of configuration fragments in Starlark. */
@StarlarkBuiltin(
    name = "fragments",
    category = StarlarkDocumentationCategory.BUILTIN,
    doc =
        "A collection of configuration fragments available in the current rule implementation "
            + "context. Access a specific fragment by its field name. For example, "
            + "<code>ctx.fragments.java</code> "
            + "<p>Only configuration fragments which are declared in the rule definition "
            + "may be accessed in this collection.</p>"
            + "<p>See <a href=\"../rules.DOC_EXT#configuration-fragments\">rules documentation</a> "
            + "for details.")
public interface FragmentCollectionApi extends ClassObject, StarlarkValue {}
