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

package com.google.devtools.build.lib.skylarkbuildapi.java;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDeprecated;
import net.starlark.java.annot.StarlarkDocumentationCategory;

/**
 * Provides access to information about Java rules. Every Java-related target provides this struct,
 * accessible as a java field on a Target.
 */
@StarlarkBuiltin(
    name = "JavaStarlarkApiProvider",
    title = "java",
    category = StarlarkDocumentationCategory.PROVIDER,
    doc =
        "Deprecated. Use <a"
            + " href=\"https://docs.bazel.build/versions/master/skylark/lib/JavaInfo.html\">JavaInfo</a>"
            + " instead.")
@StarlarkDeprecated
public interface JavaStarlarkApiProviderApi<FileT extends FileApi> extends StarlarkValue {}
