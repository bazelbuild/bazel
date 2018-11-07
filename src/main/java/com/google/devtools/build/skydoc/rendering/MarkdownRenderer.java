// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.rendering;

import com.google.common.base.Joiner;
import java.io.IOException;
import java.io.StringWriter;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.velocity.VelocityContext;
import org.apache.velocity.app.VelocityEngine;
import org.apache.velocity.exception.MethodInvocationException;
import org.apache.velocity.exception.ParseErrorException;
import org.apache.velocity.exception.ResourceNotFoundException;
import org.apache.velocity.runtime.resource.loader.ClasspathResourceLoader;
import org.apache.velocity.runtime.resource.loader.JarResourceLoader;

/**
 * Produces skydoc output in markdown form.
 */
public class MarkdownRenderer {

  private static final String RULE_TEMPLATE_FILENAME =
      "com/google/devtools/build/skydoc/rendering/templates/rule.vm";
  private static final String PROVIDER_TEMPLATE_FILENAME =
      "com/google/devtools/build/skydoc/rendering/templates/provider.vm";
  private static final String FUNCTION_TEMPLATE_FILENAME =
      "com/google/devtools/build/skydoc/rendering/templates/func.vm";

  private final VelocityEngine velocityEngine;

  public MarkdownRenderer() {
    this.velocityEngine = new VelocityEngine();
    velocityEngine.setProperty("resource.loader", "classpath, jar");
    velocityEngine.setProperty("classpath.resource.loader.class",
        ClasspathResourceLoader.class.getName());
    velocityEngine.setProperty("jar.resource.loader.class", JarResourceLoader.class.getName());
    velocityEngine.setProperty("input.encoding", "UTF-8");
    velocityEngine.setProperty("output.encoding", "UTF-8");
    velocityEngine.setProperty("runtime.references.strict", true);
  }

  /**
   * Returns a markdown rendering of rule documentation for the given rule information object with
   * the given rule name.
   */
  public String render(String ruleName, RuleInfo ruleInfo) throws IOException {
    VelocityContext context = new VelocityContext();
    // TODO(cparsons): Attributes in summary form should have links.
    context.put("summaryform", getSummaryForm(ruleName, ruleInfo));
    context.put("ruleName", ruleName);
    context.put("ruleInfo", ruleInfo);

    StringWriter stringWriter = new StringWriter();
    try {
      velocityEngine.mergeTemplate(RULE_TEMPLATE_FILENAME, "UTF-8", context, stringWriter);
    } catch (ResourceNotFoundException | ParseErrorException | MethodInvocationException e) {
      throw new IOException(e);
    }
    return stringWriter.toString();
  }

  /**
   * Returns a markdown rendering of provider documentation for the given provider information
   * object with the given name.
   */
  public String render(String providerName, ProviderInfo providerInfo) throws IOException {
    VelocityContext context = new VelocityContext();
    context.put("providerName", providerName);
    context.put("providerInfo", providerInfo);

    StringWriter stringWriter = new StringWriter();
    try {
      velocityEngine.mergeTemplate(PROVIDER_TEMPLATE_FILENAME, "UTF-8", context, stringWriter);
    } catch (ResourceNotFoundException | ParseErrorException | MethodInvocationException e) {
      throw new IOException(e);
    }
    return stringWriter.toString();
  }

  /**
   * Returns a markdown rendering of a user-defined function's documentation for the function info
   * object.
   */
  public String render(UserDefinedFunctionInfo functionInfo) throws IOException {
    VelocityContext context = new VelocityContext();
    context.put("funcInfo", functionInfo);

    StringWriter stringWriter = new StringWriter();
    try {
      velocityEngine.mergeTemplate(FUNCTION_TEMPLATE_FILENAME, "UTF-8", context, stringWriter);
    } catch (ResourceNotFoundException | ParseErrorException | MethodInvocationException e) {
      throw new IOException(e);
    }
    return stringWriter.toString();
  }

  private static String getSummaryForm(String ruleName, RuleInfo ruleInfo) {
    List<String> attributeNames = ruleInfo.getAttributes().stream()
        .map(attr -> attr.getName())
        .collect(Collectors.toList());
    return String.format("%s(%s)", ruleName, Joiner.on(", ").join(attributeNames));
  }
}
