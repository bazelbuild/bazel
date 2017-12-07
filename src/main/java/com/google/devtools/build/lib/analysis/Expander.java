// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.stringtemplate.ExpansionException;
import com.google.devtools.build.lib.analysis.stringtemplate.TemplateContext;
import com.google.devtools.build.lib.analysis.stringtemplate.TemplateExpander;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.syntax.Type;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Expansion of strings and string lists by replacing make variables and $(location) functions.
 */
public final class Expander {

  private final RuleContext ruleContext;
  private final TemplateContext templateContext;

  Expander(RuleContext ruleContext, TemplateContext templateContext) {
    this.ruleContext = ruleContext;
    this.templateContext = templateContext;
  }

  /**
   * Returns a new instance that also expands locations using the default configuration of {@link
   * LocationTemplateContext}.
   */
  private Expander withLocations(boolean execPaths, boolean allowData) {
    TemplateContext newTemplateContext =
        new LocationTemplateContext(templateContext, ruleContext, null, execPaths, allowData);
    return new Expander(ruleContext, newTemplateContext);
  }

  /**
   * Returns a new instance that also expands locations, passing {@code allowData} to the underlying
   * {@link LocationTemplateContext}.
   */
  public Expander withDataLocations() {
    return withLocations(false, true);
  }

  /**
   * Returns a new instance that also expands locations, passing {@code allowData} and {@code
   * execPaths} to the underlying {@link LocationTemplateContext}.
   */
  public Expander withDataExecLocations() {
    return withLocations(true, true);
  }

  /**
   * Returns a new instance that also expands locations, passing the given location map, as well as
   * {@code execPaths} to the underlying {@link LocationTemplateContext}.
   */
  public Expander withExecLocations(ImmutableMap<Label, ImmutableCollection<Artifact>> locations) {
    TemplateContext newTemplateContext =
        new LocationTemplateContext(templateContext, ruleContext, locations, true, false);
    return new Expander(ruleContext, newTemplateContext);
  }

  /**
   * Expands the given value string, tokenizes it, and then adds it to the given list. The attribute
   * name is only used for error reporting.
   */
  public void tokenizeAndExpandMakeVars(
      List<String> result,
      String attributeName,
      String value) {
    expandValue(result, attributeName, value, /* shouldTokenize */ true);
  }

  /** Expands make variables and $(location) tags in value, and optionally tokenizes the result. */
  private void expandValue(
      List<String> tokens, String attributeName, String value, boolean shouldTokenize) {
    value = expand(attributeName, value);
    if (shouldTokenize) {
      try {
        ShellUtils.tokenize(tokens, value);
      } catch (ShellUtils.TokenizationException e) {
        ruleContext.attributeError(attributeName, e.getMessage());
      }
    } else {
      tokens.add(value);
    }
  }

  /**
   * Returns the string "expression" after expanding all embedded references to
   * "Make" variables.  If any errors are encountered, they are reported, and
   * "expression" is returned unchanged.
   *
   * @param attributeName the name of the attribute
   * @return the expansion of "expression".
   */
  public String expand(String attributeName) {
    return expand(attributeName, ruleContext.attributes().get(attributeName, Type.STRING));
  }

  /**
   * Returns the string "expression" after expanding all embedded references to
   * "Make" variables.  If any errors are encountered, they are reported, and
   * "expression" is returned unchanged.
   *
   * @param attributeName the name of the attribute from which "expression" comes;
   *     used for error reporting.
   * @param expression the string to expand.
   * @return the expansion of "expression".
   */
  public String expand(@Nullable String attributeName, String expression) {
    try {
      return TemplateExpander.expand(expression, templateContext);
    } catch (ExpansionException e) {
      if (attributeName == null) {
        ruleContext.ruleError(e.getMessage());
      } else {
        ruleContext.attributeError(attributeName, e.getMessage());
      }
      return expression;
    }
  }

  /**
   * Expands all the strings in the given list, optionally tokenizing them after expansion. The
   * attribute name is only used for error reporting.
   */
  private ImmutableList<String> expandAndTokenizeList(
      String attrName, List<String> values, boolean shouldTokenize) {
    List<String> variables = new ArrayList<>();
    for (String variable : values) {
      expandValue(variables, attrName, variable, shouldTokenize);
    }
    return ImmutableList.copyOf(variables);
  }

  /**
   * Obtains the value of the attribute, expands all values, and returns the resulting list. If the
   * attribute does not exist or is not of type {@link Type#STRING_LIST}, then this method throws
   * an error.
   */
  public ImmutableList<String> list(String attrName) {
    return list(attrName, ruleContext.attributes().get(attrName, Type.STRING_LIST));
  }

  /**
   * Expands all the strings in the given list. The attribute name is only used for error reporting.
   */
  public ImmutableList<String> list(String attrName, List<String> values) {
    return expandAndTokenizeList(attrName, values, /* shouldTokenize */ false);
  }

  /**
   * Obtains the value of the attribute, expands, and tokenizes all values. If the attribute does
   * not exist or is not of type {@link Type#STRING_LIST}, then this method throws an error.
   */
  public ImmutableList<String> tokenized(String attrName) {
    return tokenized(attrName, ruleContext.attributes().get(attrName, Type.STRING_LIST));
  }

  /**
   * Expands all the strings in the given list, and tokenizes them after expansion. The attribute
   * name is only used for error reporting.
   */
  public ImmutableList<String> tokenized(String attrName, List<String> values) {
    return expandAndTokenizeList(attrName, values, /* shouldTokenize */ true);
  }

  /**
   * If the string consists of a single variable, returns the expansion of that variable. Otherwise,
   * returns null. Syntax errors are reported.
   *
   * @param attrName the name of the attribute from which "expression" comes; used for error
   *     reporting.
   * @param expression the string to expand.
   * @return the expansion of "expression", or null.
   */
  @Nullable
  public String expandSingleMakeVariable(String attrName, String expression) {
    try {
      return TemplateExpander.expandSingleVariable(expression, templateContext);
    } catch (ExpansionException e) {
      ruleContext.attributeError(attrName, e.getMessage());
      return expression;
    }
  }
}
