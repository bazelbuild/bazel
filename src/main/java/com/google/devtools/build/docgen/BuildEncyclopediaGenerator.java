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

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * The main class for the docgen project. The class checks the input arguments
 * and uses the BuildEncyclopediaProcessor for the actual documentation generation.
 */
public class BuildEncyclopediaGenerator {

  private static boolean checkArgs(String[] args) {
    if (args.length < 2) {
      System.err.println("There has to be two to four input parameters\n"
          + " - a comma separated list for input directories\n"
          + " - the name of the rule class provider\n"
          + " - an output directory (optional)."
          + " - a path to a file listing rules to not document (optional)");
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

  private static ConfiguredRuleClassProvider createRuleClassProvider(String classProvider)
      throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException,
      IllegalAccessException {
    Class<?> providerClass = Class.forName(classProvider);
    Method createMethod = providerClass.getMethod("create");
    return (ConfiguredRuleClassProvider) createMethod.invoke(null);
  }

  public static void main(String[] args) {
    if (checkArgs(args)) {
      // TODO(bazel-team): use flags
      try {
        BuildEncyclopediaProcessor processor = new BuildEncyclopediaProcessor(
            createRuleClassProvider(args[1]));
        processor.generateDocumentation(
            args[0].split(","), getArgsOrNull(args, 2), getArgsOrNull(args, 3));
      } catch (BuildEncyclopediaDocException e) {
        fail(e, false);
      } catch (Throwable e) {
        fail(e, true);
      }
    }
  }

  private static String getArgsOrNull(String[] args, int idx) {
    return args.length > idx ? args[idx] : null;
  }
}
