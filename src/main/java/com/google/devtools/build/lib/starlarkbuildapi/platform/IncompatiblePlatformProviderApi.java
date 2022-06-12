// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi.platform;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/** Targets that are incompatible with the target platform. */
@StarlarkBuiltin(
    name = "IncompatiblePlatformProvider",
    doc =
        "A provider for targets that are incompatible with the target platform. See "
            + "<a href='${link platforms#detecting-incompatible-targets-using-bazel-cquery}'>"
            + "Detecting incompatible targets using <code>bazel cquery</code></a> for more "
            + "information.",
    category = DocCategory.PROVIDER)
public interface IncompatiblePlatformProviderApi extends StarlarkValue {}
