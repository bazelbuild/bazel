// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.docgen.starlark.*;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.build.docgen.rulesrcdoc.RuleSrcDoc.RuleSrcDocs;
import com.google.devtools.build.docgen.rulesrcdoc.RuleSrcDoc.RuleDocumentationProto;
import com.google.devtools.build.docgen.rulesrcdoc.RuleSrcDoc.RuleDocumentationAttributeProto;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;

/** The main class for exporting rule docs collected from source files. */
public class RuleSrcDocExporter {

  private static void printUsage(OptionsParser parser) {
//    System.err.println(
//        "Usage: api_exporter_bin -m link_map_path -p rule_class_provider\n"
//            + "    [-r input_root] (-i input_dir)+ (--input_stardoc_proto binproto)+\n"
//            + "    -f outputFile [-b denylist] [-h]\n\n"
//            + "Exports all Starlark builtins to a file including the embedded native rules.\n"
//            + "The link map path (-m), rule class provider (-p), output file (-f), and at least\n"
//            + " one input_dir (-i) or binproto (--input_stardoc_proto) must be specified.\n");
//    System.err.println(
//        parser.describeOptionsWithDeprecatedCategories(
//            Collections.<String, String>emptyMap(), OptionsParser.HelpVerbosity.LONG));
  }

  private static void fail(Throwable e, boolean printStackTrace) {
    System.err.println("ERROR: " + e.getMessage());
    if (printStackTrace) {
      e.printStackTrace();
    }
    Runtime.getRuntime().exit(1);
  }

  private static ConfiguredRuleClassProvider createRuleClassProvider(String classProvider)
          throws NoSuchMethodException, InvocationTargetException, IllegalAccessException,
          ClassNotFoundException {
    Class<?> providerClass = Class.forName(classProvider);
    Method createMethod = providerClass.getMethod("create");
    return (ConfiguredRuleClassProvider) createMethod.invoke(null);
  }

  private static RuleSrcDocs ruleDocsToProto(Collection<RuleDocumentation> ruleDocs) throws BuildEncyclopediaDocException {
    RuleSrcDocs.Builder builder = RuleSrcDocs.newBuilder();

    for (RuleDocumentation ruleDoc : ruleDocs.stream().sorted().toList()) {
        RuleDocumentationProto.Builder ruleProto = builder.addRuleBuilder();
        ruleProto.setRuleName(ruleDoc.getRuleName());
        ruleProto.setHtmlDocumentation(ruleDoc.getHtmlDocumentation());

        for (RuleDocumentationAttribute ruleAttr : ruleDoc.getAttributes()) {
          RuleDocumentationAttributeProto.Builder attrProto = ruleProto.addAttributeBuilder();
          attrProto.setAttributeName(ruleAttr.getAttributeName());
          attrProto.setHtmlDocumentation(ruleAttr.getHtmlDocumentation());
          String defaultValue = ruleAttr.getDefaultValue();
          if (defaultValue != null) {
            attrProto.setDefaultValue(defaultValue);
          }
          attrProto.setIsMandatory(ruleAttr.isMandatory());
          attrProto.setIsDeprecated(ruleAttr.isDeprecated());
        }
    }
    return builder.build();
  }


  private static void writeSrcDocs(String filename, RuleSrcDocs ruleSrcDocs) throws IOException {
    try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(filename))) {
      ruleSrcDocs.writeTo(out);
    }
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
            || options.provider.isEmpty()) {
      printUsage(parser);
      Runtime.getRuntime().exit(1);
    }

    try {
      DocLinkMap linkMap = DocLinkMap.createFromFile(options.linkMapPath);
      RuleLinkExpander linkExpander = new RuleLinkExpander(
              /* singlePage */ false, linkMap);
      SourceUrlMapper urlMapper = new SourceUrlMapper(linkMap, options.inputRoot);
      // TODO: Describe why RuleClassProvider is needed even when getting docs from source files.
      ConfiguredRuleClassProvider ruleClassProvider = createRuleClassProvider(options.provider);

      BuildDocCollector collector = new BuildDocCollector(linkExpander, urlMapper, ruleClassProvider);
      Map<String, RuleDocumentation> ruleDocEntries =
              collector.collect(options.inputJavaDirs, options.inputStardocProtos, options.denylist);

      RuleSrcDocs rulesProto = ruleDocsToProto(ruleDocEntries.values());
      writeSrcDocs(options.outputFile, rulesProto);

    } catch (BuildEncyclopediaDocException e) {
      fail(e, false);
    } catch (Throwable e) {
      fail(e, true);
    }
  }

}
