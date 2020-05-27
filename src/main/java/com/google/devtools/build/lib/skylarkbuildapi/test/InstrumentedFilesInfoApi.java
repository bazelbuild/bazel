// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi.test;

import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;

/** Contains information about instrumented files sources and instrumentation metadata. */
@StarlarkBuiltin(
    name = "InstrumentedFilesInfo",
    category = StarlarkDocumentationCategory.PROVIDER,
    doc =
        "Contains information about instrumented source files and instrumentation metadata files "
            + "for purposes of <a href=\"../rules.html#code-coverage-instrumentation\">code "
            + "coverage data collection</a>.")
public interface InstrumentedFilesInfoApi extends StarlarkValue {}
