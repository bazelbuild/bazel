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

package com.google.devtools.build.lib.skylarkbuildapi.apple;

import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;

/**
 * Interface for an enum describing the bitcode mode to use when compiling Objective-C and Swift
 * code on Apple platforms.
 */
@StarlarkBuiltin(
    name = "apple_bitcode_mode",
    category = StarlarkDocumentationCategory.BUILTIN,
    doc =
        "The Bitcode mode to use when compiling Objective-C and Swift code on Apple platforms. "
            + "Possible values are:<br><ul>"
            + "<li><code>'none'</code></li>"
            + "<li><code>'embedded'</code></li>"
            + "<li><code>'embedded_markers'</code></li>"
            + "</ul>")
public interface AppleBitcodeModeApi extends StarlarkValue {}
