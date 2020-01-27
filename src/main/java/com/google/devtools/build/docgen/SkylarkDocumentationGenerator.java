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


import java.util.Arrays;

/**
 * The main class for the skylark documentation generator.
 */
public class SkylarkDocumentationGenerator {

  private static boolean checkArgs(String[] args) {
    if (args.length < 1) {
      System.err.println("There has to be at least one input parameter\n"
          + " - an output file.");
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
      System.out.println("Generating Starlark documentation...");
      try {
        SkylarkDocumentationProcessor.generateDocumentation(
            args[0],
            Arrays.copyOfRange(args, 1, args.length));
      } catch (Throwable e) {
        fail(e, true);
      }
      System.out.println("Finished.");
    }
  }
}
