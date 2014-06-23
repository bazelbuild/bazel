// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider;

/**
 * The main class for the docgen project. The class checks the input arguments
 * and uses the BuildEncyclopeidaPRocessor for the actual documentation generation.
 */
public class BuildEncyclopediaGenerator {

  private static boolean checkArgs(String[] args) {
    if (args.length < 1) {
      System.err.println("There has to be one or two input parameters\n"
          + " - a comma separated list for input directories\n"
          + " - an output directory (optional).");
      return false;
    }
    return true;
  }

  private static void fail(Throwable e, boolean printStackTrace) {
    System.err.println("ERROR: " + e.getMessage());
    if (printStackTrace) {
      e.printStackTrace();
    }
    Runtime.getRuntime().exit(1);
  }

  public static void main(String[] args) {
    if (checkArgs(args)) {
      // TODO(bazel-team): use flags
      System.out.println("Generating Build Encyclopedia...");
      BuildEncyclopediaProcessor processor = new BuildEncyclopediaProcessor(
          BazelRuleClassProvider.create());
      try {
        processor.generateDocumentation(args[0].split(","), args.length > 1 ? args[1] : null);
      } catch (BuildEncyclopediaDocException e) {
        fail(e, false);
      } catch (Throwable e) {
        fail(e, true);
      }
      System.out.println("Finished.");
    }
  }
}
