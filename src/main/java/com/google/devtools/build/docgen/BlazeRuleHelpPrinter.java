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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * A helper class to load and store printable build rule documentation. The doc
 * printed here doesn't contain attribute and implicit output definitions, just
 * the general rule documentation and examples.
 */
public class BlazeRuleHelpPrinter {

  private static Map<String, RuleDocumentation> ruleDocMap = null;

  /**
   * Returns the documentation of the given rule to be printed on the console.
   */
  public static String getRuleDoc(String ruleName, ConfiguredRuleClassProvider provider) {
    if (ruleDocMap == null) {
      try {
        BuildDocCollector collector = new BuildDocCollector(provider, false);
        Map<String, RuleDocumentation> ruleDocs = collector.collect(
            ImmutableList.of("java/com/google/devtools/build/lib/view",
                             "java/com/google/devtools/build/lib/rules"), null);
        ruleDocMap = new HashMap<>();
        for (RuleDocumentation ruleDoc : ruleDocs.values()) {
          ruleDocMap.put(ruleDoc.getRuleName(), ruleDoc);
        }
      } catch (BuildEncyclopediaDocException e) {
        return e.getErrorMsg();
      } catch (IOException e) {
        return e.getMessage();
      }
    }
    // TODO(fwe): generate the rule documentation when building the Bazel binary and ship
    // the text documentation with the binary.
    // The Bazel binary may not contain the rule sources, which means that the documentation
    // retrieval step would fail with an exception.
    if (!ruleDocMap.containsKey(ruleName)) {
      return String.format("Documentation for rule %s is not part of the binary.\n", ruleName);
    }
    return "Rule " + ruleName + ":"
        + ruleDocMap.get(ruleName).getCommandLineDocumentation() + "\n";
  }
}
