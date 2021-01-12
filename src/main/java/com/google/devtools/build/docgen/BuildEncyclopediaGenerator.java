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
import com.google.devtools.common.options.OptionsParser;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collections;

/**
 * The main class for the docgen project. The class checks the input arguments
 * and uses the BuildEncyclopediaProcessor for the actual documentation generation.
 */
public class BuildEncyclopediaGenerator {
  private static void printUsage(OptionsParser parser) {
    System.err.println(
        "Usage: docgen_bin -n product_name -p rule_class_provider (-i input_dir)+\n"
            + "    [-o outputdir] [-b denylist] [-1] [-h]\n\n"
            + "Generates the Build Encyclopedia from embedded native rule documentation.\n"
            + "The product name (-n), rule class provider (-p) and at least one input_dir\n"
            + "(-i) must be specified.\n");
    System.err.println(
        parser.describeOptionsWithDeprecatedCategories(
            Collections.<String, String>emptyMap(), OptionsParser.HelpVerbosity.LONG));
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
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(BuildEncyclopediaOptions.class)
            .allowResidue(false)
            .build();
    parser.parseAndExitUponError(args);
    BuildEncyclopediaOptions options = parser.getOptions(BuildEncyclopediaOptions.class);

    if (options.help) {
      printUsage(parser);
      Runtime.getRuntime().exit(0);
    }

    if (options.productName.isEmpty()
        || options.inputDirs.isEmpty()
        || options.provider.isEmpty()) {
      printUsage(parser);
      Runtime.getRuntime().exit(1);
    }

    try {
      BuildEncyclopediaProcessor processor = null;
      if (options.singlePage) {
        processor =
            new SinglePageBuildEncyclopediaProcessor(
                options.productName, createRuleClassProvider(options.provider));
      } else {
        processor =
            new MultiPageBuildEncyclopediaProcessor(
                options.productName, createRuleClassProvider(options.provider));
      }
      processor.generateDocumentation(options.inputDirs, options.outputDir, options.denylist);
    } catch (BuildEncyclopediaDocException e) {
      fail(e, false);
    } catch (Throwable e) {
      fail(e, true);
    }
  }
}
