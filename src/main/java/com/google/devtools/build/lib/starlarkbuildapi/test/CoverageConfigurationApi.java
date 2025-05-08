// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi.test;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.cmdline.Label;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Coverage configuration fragment API. */
@StarlarkBuiltin(
    name = "coverage",
    category = DocCategory.CONFIGURATION_FRAGMENT,
    doc = "A configuration fragment representing the coverage configuration.")
public interface CoverageConfigurationApi extends StarlarkValue {

  @StarlarkMethod(
      name = "output_generator",
      allowReturnNones = true,
      structField = true,
      doc =
          "Returns the label pointed to by the"
              + " <a href=\"https://bazel.build/reference/command-line-reference"
              + "#flag--coverage_output_generator\">"
              + "<code>--coverage_output_generator</code></a> option if coverage collection is"
              + " enabled, otherwise returns <code>None</code>. Can be accessed with"
              + " <a href=\"../globals/bzl.html#configuration_field\"><code>configuration_field"
              + "</code></a>:<br/>"
              + "<pre>attr.label(<br/>"
              + "    default = configuration_field(<br/>"
              + "        fragment = \"coverage\",<br/>"
              + "        name = \"output_generator\"<br/>"
              + "    )<br/>"
              + ")</pre>")
  @Nullable
  Label outputGenerator();
}
