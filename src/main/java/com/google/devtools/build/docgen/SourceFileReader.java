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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.regex.Matcher;

/**
 * A helper class to read and process documentations for rule classes and attributes
 * from exactly one java source file.
 */
public class SourceFileReader {

  private Collection<RuleDocumentation> ruleDocEntries;
  private ListMultimap<String, RuleDocumentationAttribute> attributeDocEntries;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final String javaSourceFilePath;

  public SourceFileReader(
      ConfiguredRuleClassProvider ruleClassProvider, String javaSourceFilePath) {
    this.ruleClassProvider = ruleClassProvider;
    this.javaSourceFilePath = javaSourceFilePath;
  }

  /**
   * The handler class of the line read from the text file.
   */
  public abstract static class ReadAction {

    // Text file line indexing starts from 1
    private int lineCnt = 1;

    protected abstract void readLineImpl(String line)
        throws BuildEncyclopediaDocException, IOException;

    protected int getLineCnt() {
      return lineCnt;
    }

    public void readLine(String line)
        throws BuildEncyclopediaDocException, IOException {
      readLineImpl(line);
      lineCnt++;
    }
  }

  private static final String LS = DocgenConsts.LS;

  /**
   * Reads the attribute and rule documentation present in the file represented by
   * SourceFileReader.javaSourceFilePath. The rule doc variables are added to the rule
   * documentation (which therefore must be defined in the same file). The attribute docs are
   * stored in a different class member, so they need to be handled outside this method.
   */
  public void readDocsFromComments() throws BuildEncyclopediaDocException, IOException {
    final Map<String, RuleDocumentation> docMap = new HashMap<>();
    final List<RuleDocumentationVariable> docVariables = new LinkedList<>();
    final ListMultimap<String, RuleDocumentationAttribute> docAttributes =
        LinkedListMultimap.create();
    readTextFile(javaSourceFilePath, new ReadAction() {

      private boolean inBlazeRuleDocs = false;
      private boolean inBlazeRuleVarDocs = false;
      private boolean inBlazeAttributeDocs = false;
      private StringBuilder sb = new StringBuilder();
      private String ruleName;
      private String ruleType;
      private String ruleFamily;
      private String variableName;
      private String attributeName;
      private ImmutableSet<String> flags;
      private int startLineCnt;

      @Override
      public void readLineImpl(String line) throws BuildEncyclopediaDocException {
        // TODO(bazel-team): check if copy paste code can be reduced using inner classes
        if (inBlazeRuleDocs) {
          if (DocgenConsts.BLAZE_RULE_END.matcher(line).matches()) {
            endBlazeRuleDoc(docMap);
          } else {
            appendLine(line);
          }
        } else if (inBlazeRuleVarDocs) {
          if (DocgenConsts.BLAZE_RULE_VAR_END.matcher(line).matches()) {
            endBlazeRuleVarDoc(docVariables);
          } else {
            appendLine(line);
          }
        } else if (inBlazeAttributeDocs) {
          if (DocgenConsts.BLAZE_RULE_ATTR_END.matcher(line).matches()) {
            endBlazeAttributeDoc(docAttributes);
          } else {
            appendLine(line);
          }
        }
        Matcher ruleStartMatcher = DocgenConsts.BLAZE_RULE_START.matcher(line);
        Matcher ruleVarStartMatcher = DocgenConsts.BLAZE_RULE_VAR_START.matcher(line);
        Matcher ruleAttrStartMatcher = DocgenConsts.BLAZE_RULE_ATTR_START.matcher(line);
        if (ruleStartMatcher.find()) {
          startBlazeRuleDoc(line, ruleStartMatcher);
        } else if (ruleVarStartMatcher.find()) {
          startBlazeRuleVarDoc(ruleVarStartMatcher);
        } else if (ruleAttrStartMatcher.find()) {
          startBlazeAttributeDoc(line, ruleAttrStartMatcher);
        }
      }

      private void appendLine(String line) {
        // Add another line of html code to the building rule documentation
        // Removing whitespace and java comment asterisk from the beginning of the line
        sb.append(line.replaceAll("^[\\s]*\\*", "") + LS);
      }

      private void startBlazeRuleDoc(String line, Matcher matcher)
          throws BuildEncyclopediaDocException {
        checkDocValidity();
        // Start of a new rule.
        // e.g.: matcher.group(1) = "NAME = cc_binary, TYPE = BINARY, FAMILY = C / C++"
        for (String group : Splitter.on(",").split(matcher.group(1))) {
          List<String> parts = Splitter.on("=").limit(2).splitToList(group);
          boolean good = false;
          if (parts.size() == 2) {
            String key = parts.get(0).trim();
            String value = parts.get(1).trim();
            good = true;
            if (DocgenConsts.META_KEY_NAME.equals(key)) {
              ruleName = value;
            } else if (DocgenConsts.META_KEY_TYPE.equals(key)) {
              ruleType = value;
            } else if (DocgenConsts.META_KEY_FAMILY.equals(key)) {
              ruleFamily = value;
            } else {
              good = false;
            }
          }
          if (!good) {
            System.err.printf("WARNING: bad rule definition in line %d: '%s'", getLineCnt(), line);
          }
        }

        startLineCnt = getLineCnt();
        addFlags(line);
        inBlazeRuleDocs = true;
      }

      private void endBlazeRuleDoc(final Map<String, RuleDocumentation> documentations)
          throws BuildEncyclopediaDocException {
        // End of a rule, create RuleDocumentation object
        documentations.put(ruleName, new RuleDocumentation(ruleName, ruleType,
            ruleFamily, sb.toString(), getLineCnt(), javaSourceFilePath, flags,
            ruleClassProvider));
        sb = new StringBuilder();
        inBlazeRuleDocs = false;
      }

      private void startBlazeRuleVarDoc(Matcher matcher) throws BuildEncyclopediaDocException {
        checkDocValidity();
        // Start of a new rule variable
        ruleName = matcher.group(1).replaceAll("[\\s]", "");
        variableName = matcher.group(2).replaceAll("[\\s]", "");
        startLineCnt = getLineCnt();
        inBlazeRuleVarDocs = true;
      }

      private void endBlazeRuleVarDoc(final List<RuleDocumentationVariable> docVariables) {
        // End of a rule, create RuleDocumentationVariable object
        docVariables.add(
            new RuleDocumentationVariable(ruleName, variableName, sb.toString(), startLineCnt));
        sb = new StringBuilder();
        inBlazeRuleVarDocs = false;
      }

      private void startBlazeAttributeDoc(String line, Matcher matcher)
          throws BuildEncyclopediaDocException {
        checkDocValidity();
        // Start of a new attribute
        ruleName = matcher.group(1).replaceAll("[\\s]", "");
        attributeName = matcher.group(2).replaceAll("[\\s]", "");
        startLineCnt = getLineCnt();
        addFlags(line);
        inBlazeAttributeDocs = true;
      }

      private void endBlazeAttributeDoc(
          final ListMultimap<String, RuleDocumentationAttribute> docAttributes) {
        // End of a attribute, create RuleDocumentationAttribute object
        docAttributes.put(attributeName, RuleDocumentationAttribute.create(
            ruleClassProvider.getRuleClassDefinition(ruleName),
            attributeName, sb.toString(), startLineCnt, javaSourceFilePath, flags));
        sb = new StringBuilder();
        inBlazeAttributeDocs = false;
      }

      private void addFlags(String line) {
        // Add flags if there's any
        Matcher matcher = DocgenConsts.BLAZE_RULE_FLAGS.matcher(line);
        if (matcher.find()) {
          flags = ImmutableSet.<String>copyOf(matcher.group(1).split(","));
        } else {
          flags = ImmutableSet.<String>of();
        }
      }

      private void checkDocValidity() throws BuildEncyclopediaDocException {
        if (inBlazeRuleDocs || inBlazeRuleVarDocs || inBlazeAttributeDocs) {
          throw new BuildEncyclopediaDocException(javaSourceFilePath, getLineCnt(),
              "Malformed documentation, #BLAZE_RULE started after another #BLAZE_RULE.");
        }
      }
    });

    // Adding rule doc variables to the corresponding rules
    for (RuleDocumentationVariable docVariable : docVariables) {
      if (docMap.containsKey(docVariable.getRuleName())) {
        docMap.get(docVariable.getRuleName()).addDocVariable(
          docVariable.getVariableName(), docVariable.getValue());
      } else {
        throw new BuildEncyclopediaDocException(javaSourceFilePath, docVariable.getStartLineCnt(),
            String.format("Malformed rule variable #BLAZE_RULE(%s).%s, rule %s not found in file.",
                docVariable.getRuleName(), docVariable.getVariableName(),
                docVariable.getRuleName()));
      }
    }
    ruleDocEntries = docMap.values();
    attributeDocEntries = docAttributes;
  }

  public Collection<RuleDocumentation> getRuleDocEntries() {
    return ruleDocEntries;
  }

  public ListMultimap<String, RuleDocumentationAttribute> getAttributeDocEntries() {
    return attributeDocEntries;
  }

  /**
   * Reads the template file without variable substitution.
   */
  public static String readTemplateContents(String templateFilePath)
      throws BuildEncyclopediaDocException, IOException {
    return readTemplateContents(templateFilePath, null);
  }

  /**
   * Reads a template file and substitutes variables of the format ${FOO}.
   *
   * @param variables keys are the possible variable names, e.g. "FOO", values are the substitutions
   *     (can be null)
   */
  public static String readTemplateContents(String templateFilePath,
      final Map<String, String> variables) throws BuildEncyclopediaDocException, IOException {
    final StringBuilder sb = new StringBuilder();
    readTextFile(templateFilePath, new ReadAction() {
      @Override
      public void readLineImpl(String line) {
        sb.append(expandVariables(line, variables)).append(LS);
      }
    });
    return sb.toString();
  }

  private static String expandVariables(String line, Map<String, String> variables) {
    if (variables == null || line.indexOf("${") == -1) {
      return line;
    }

    for (Entry<String, String> variable : variables.entrySet()) {
      line = line.replace("${" + variable.getKey() + "}", variable.getValue());
    }
    return line;
  }

  public static void readTextFile(String filePath, ReadAction action)
      throws BuildEncyclopediaDocException, IOException {
    BufferedReader br = null;
    try {
      File file = new File(filePath);
      if (file.exists()) {
        br = new BufferedReader(new FileReader(file));
      } else {
        InputStream is = SourceFileReader.class.getResourceAsStream(filePath);
        if (is != null) {
          br = new BufferedReader(new InputStreamReader(is));
        }
      }
      if (br != null) {
        String line = null;
        while ((line = br.readLine()) != null) {
          action.readLine(line);
        }
      } else {
        System.out.println("Couldn't find file or resource: " + filePath);
      }
    } finally {
      if (br != null) {
        br.close();
      }
    }
  }
}
