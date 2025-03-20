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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/** The interface for late bound defaults in Starlark. */
@StarlarkBuiltin(
    name = "LateBoundDefault",
    category = DocCategory.BUILTIN,
    doc =
        "Represents a late-bound default attribute value of type 'Label'. The value of a"
            + " LateBoundDefault is only resolvable in the context of a rule implementation"
            + " function, and depends on the current build configuration. For example, a"
            + " LateBoundDefault might represent the Label of the java toolchain in the current"
            + " build configuration. <p>See <a"
            + " href=\"../globals/bzl.html#configuration_field\">configuration_field</a> for"
            + " example usage.")
public interface LateBoundDefaultApi extends StarlarkValue {}
