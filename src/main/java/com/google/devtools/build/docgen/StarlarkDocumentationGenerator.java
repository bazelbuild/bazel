// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import com.google.devtools.common.options.OptionsParser;
import java.util.Collections;

/** The main class for the Starlark documentation generator. */
public class StarlarkDocumentationGenerator {

  private static void printUsage(OptionsParser parser) {
    System.err.println(
"""
Usage: skydoc_bin output_dir --link_map_path=link_map.json [other options]

Generates Starlark API documentation, including Starlark and BUILD language
built-in functions and data types, providers, configuration fragments, etc.
""");
    System.err.println(
        parser.describeOptionsWithDeprecatedCategories(
            Collections.emptyMap(), OptionsParser.HelpVerbosity.LONG));
  }

  private static void fail(Throwable e, boolean printStackTrace) {
    System.err.println("ERROR: " + e.getMessage());
    if (printStackTrace) {
      e.printStackTrace();
    }
    Runtime.getRuntime().exit(1);
  }

  public static void main(String[] args) {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(StarlarkDocumentationOptions.class)
            .allowResidue(true)
            .build();
    parser.parseAndExitUponError(args);
    StarlarkDocumentationOptions options = parser.getOptions(StarlarkDocumentationOptions.class);

    if (options.help) {
      printUsage(parser);
      Runtime.getRuntime().exit(0);
    }

    if (parser.getResidue().size() != 1 || options.linkMapPath.isEmpty()) {
      printUsage(parser);
      Runtime.getRuntime().exit(1);
    }
    String outputDir = parser.getResidue().getFirst();

    System.out.println("Generating Starlark documentation...");
    try {
      StarlarkDocumentationProcessor.generateDocumentation(outputDir, options);
    } catch (Throwable e) {
      fail(e, true);
    }
    System.out.println("Finished.");
  }
}
