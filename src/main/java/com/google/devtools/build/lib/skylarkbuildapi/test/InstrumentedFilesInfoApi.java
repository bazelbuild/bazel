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

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** Contains information about instrumented files sources and instrumentation metadata. */
@StarlarkBuiltin(
    name = "InstrumentedFilesInfo",
    category = StarlarkDocumentationCategory.PROVIDER,
    doc =
        "Contains information about source files and instrumentation metadata files for rule "
            + "targets matched by <a href=\"../../command-line-reference.html"
            + "#flag--instrumentation_filter\"><code>--instrumentation_filter</code></a> "
            + "for purposes of <a href=\"../rules.html#code-coverage-instrumentation\">code "
            + "coverage data collection</a>. When coverage data collection is enabled, a manifest "
            + "containing the combined paths in <a href=\"#instrumented_files\">"
            + "<code>instrumented_files</code></a> and <a href=\"#metadata_files\">"
            + "<code>metadata_files</code></a> are passed to the test action as inputs, with the "
            + "manifest's path noted in the environment variable <code>COVERAGE_MANIFEST</code>. "
            + "The metadata files, but not the source files, are also passed to the test action "
            + "as inputs.")
public interface InstrumentedFilesInfoApi extends StructApi {

  @StarlarkMethod(
      name = "instrumented_files",
      doc =
          "<a href=\"depset.html\"><code>depset</code></a> of <a href=\"File.html\"><code>File"
              + "</code></a> objects representing instrumented source files for this target and "
              + "its dependencies.",
      structField = true)
  Depset getInstrumentedFilesForStarlark();

  @StarlarkMethod(
      name = "metadata_files",
      doc =
          "<a href=\"depset.html\"><code>depset</code></a> of <a href=\"File.html\"><code>File"
              + "</code></a> objects representing coverage metadata files for this target and its "
              + "dependencies. These files contain additional information required to generate "
              + "LCOV-format coverage output after the code is executed, e.g. the "
              + "<code>.gcno</code> files generated when <code>gcc</code> is run with "
              + "<code>-ftest-coverage</code>.",
      structField = true)
  Depset getInstrumentationMetadataFilesForStarlark();
}
