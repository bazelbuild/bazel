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
        "Usage: docgen_bin -m link_map_file -p rule_class_provider\n"
            + "    [-r input_root] (-i input_dir)+ (--input_stardoc_proto binproto)+\n"
            + "    [-o outputdir] [-b denylist] [-1 | -t] [-h]\n\n"
            + "Generates the Build Encyclopedia from embedded native rule documentation.\n"
            + "The link map file (-m), rule class provider (-p), and at least one input_dir\n"
            + "(-i) or binproto (--input_stardoc_proto) must be specified.\n"
            + "Single page (-1) and table-of-contents creation (-t) are mutually exclusive.\n");
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

    if (options.linkMapPath.isEmpty()
        || (options.inputJavaDirs.isEmpty() && options.inputStardocProtos.isEmpty())
        || options.provider.isEmpty()
        || (options.singlePage && options.createToc)) {
      printUsage(parser);
      Runtime.getRuntime().exit(1);
    }

    try {
      DocLinkMap linkMap = DocLinkMap.createFromFile(options.linkMapPath);
      RuleLinkExpander linkExpander = new RuleLinkExpander(options.singlePage, linkMap);
      SourceUrlMapper urlMapper = new SourceUrlMapper(linkMap, options.inputRoot);

      BuildEncyclopediaProcessor processor = null;
      if (options.singlePage) {
        processor =
            new SinglePageBuildEncyclopediaProcessor(
                linkExpander, urlMapper, createRuleClassProvider(options.provider));
      } else {
        processor =
            new MultiPageBuildEncyclopediaProcessor(
                linkExpander,
                urlMapper,
                createRuleClassProvider(options.provider),
                options.createToc);
      }
      processor.generateDocumentation(
          options.inputJavaDirs, options.inputStardocProtos, options.outputDir, options.denylist);
    } catch (BuildEncyclopediaDocException e) {
      fail(e, false);
    } catch (Throwable e) {
      fail(e, true);
    }
  }
}
