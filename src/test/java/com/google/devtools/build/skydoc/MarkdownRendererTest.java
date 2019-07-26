// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.skydoc.rendering.MarkdownRenderer;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.FunctionParamInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderFieldInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.UserDefinedFunctionInfo;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Java tests for MarkdownRenderer */
@RunWith(JUnit4.class)
public final class MarkdownRendererTest {

  private final String headerTemplatePath =
      "com/google/devtools/build/skydoc/test_templates/header.vm";
  private final String ruleTemplatePath = "com/google/devtools/build/skydoc/test_templates/rule.vm";
  private final String providerTemplatePath =
      "com/google/devtools/build/skydoc/test_templates/provider.vm";
  private final String funcTemplatePath = "com/google/devtools/build/skydoc/test_templates/func.vm";
  private final String aspectTemplatePath =
      "com/google/devtools/build/skydoc/test_templates/aspect.vm";
  private final MarkdownRenderer renderer =
      new MarkdownRenderer(
          headerTemplatePath,
          ruleTemplatePath,
          providerTemplatePath,
          funcTemplatePath,
          aspectTemplatePath);

  @Test
  public void testHeaderStrings() throws IOException {
    assertThat(renderer.renderMarkdownHeader())
        .isEqualTo("<!-- Generated with Stardoc: http://skydoc.bazel.build -->\n");
  }

  @Test
  public void testRuleStrings() throws IOException {
    AttributeInfo attrInfo =
        AttributeInfo.newBuilder()
            .setName("first")
            .setDocString("the first attribute")
            .setTypeValue(AttributeType.STRING.getNumber())
            .build();
    RuleInfo ruleInfo =
        RuleInfo.newBuilder()
            .setRuleName("my_rule")
            .setDocString("This rule does things.")
            .addAttribute(attrInfo)
            .build();

    assertThat(renderer.render(ruleInfo.getRuleName(), ruleInfo))
        .isEqualTo(
            "<a name=\"#my_rule\"></a>\n"
                + "\n"
                + "## my_rule\n"
                + "\n"
                + "<pre>\n"
                + "my_rule(<a href=\"#my_rule-first\">first</a>)\n"
                + "</pre>\n"
                + "\n"
                + "This rule does things.\n"
                + "\n"
                + "### Attributes\n"
                + "\n"
                + "    <code>first</code>\n"
                + "    String; optional\n"
                + "        <p>\n"
                + "          the first attribute\n"
                + "        </p>\n");
  }

  @Test
  public void testProviderStrings() throws IOException {
    ProviderFieldInfo fieldInfo =
        ProviderFieldInfo.newBuilder().setName("one").setDocString("the first field").build();
    ProviderInfo providerInfo =
        ProviderInfo.newBuilder()
            .setProviderName("my_provider")
            .setDocString("This provider does things.")
            .addFieldInfo(fieldInfo)
            .build();

    assertThat(renderer.render(providerInfo.getProviderName(), providerInfo))
        .isEqualTo(
            "<a name=\"#my_provider\"></a>\n"
                + "\n"
                + "## my_provider\n"
                + "\n"
                + "<pre>\n"
                + "my_provider(<a href=\"#my_provider-one\">one</a>)\n"
                + "</pre>\n"
                + "\n"
                + "This provider does things.\n"
                + "\n"
                + "### Fields\n"
                + "\n"
                + "<code>one</code><\n"
                + "<p>the first field</p>\n"
                + "\n");
  }

  @Test
  public void testFunctionStrings() throws IOException {
    FunctionParamInfo paramInfo =
        FunctionParamInfo.newBuilder()
            .setName("param1")
            .setDocString("the first parameter")
            .setDefaultValue("32")
            .build();
    UserDefinedFunctionInfo funcInfo =
        UserDefinedFunctionInfo.newBuilder()
            .setFunctionName("my_function")
            .setDocString("This function does something.")
            .addParameter(paramInfo)
            .build();

    assertThat(renderer.render(funcInfo))
        .isEqualTo(
            "<a name=\"#my_function\"></a>\n"
                + "\n"
                + "## my_function\n"
                + "\n"
                + "<pre>\n"
                + "my_function(<a href=\"#my_function-param1\">param1</a>)\n"
                + "</pre>\n"
                + "\n"
                + "This function does something.\n"
                + "\n"
                + "### Parameters\n"
                + "\n"
                + "  <code>param1</code>\n"
                + "  optional. default is <code>32</code>\n"
                + "        <p>\n"
                + "          the first parameter\n"
                + "        </p>\n");
  }

  @Test
  public void testAspectStrings() throws IOException {
    AttributeInfo attrInfo =
        AttributeInfo.newBuilder()
            .setName("first")
            .setDocString("the first attribute")
            .setTypeValue(AttributeType.STRING.getNumber())
            .build();
    AspectInfo aspectInfo =
        AspectInfo.newBuilder()
            .setAspectName("my_aspect")
            .setDocString("This aspect does things.")
            .addAttribute(attrInfo)
            .addAspectAttribute("deps")
            .build();

    assertThat(renderer.render(aspectInfo.getAspectName(), aspectInfo))
        .isEqualTo(
            "<a name=\"#my_aspect\"></a>\n"
                + "\n"
                + "## my_aspect\n"
                + "\n"
                + "<pre>\n"
                + "null(<a href=\"#null-first\">first</a>)\n"
                + "</pre>\n"
                + "\n"
                + "This aspect does things.\n"
                + "\n"
                + "### Aspect Attributes\n"
                + "\n"
                + "        <code>deps</code><\n"
                + "        String; required.\n"
                + "\n"
                + "### Attributes\n"
                + "\n"
                + "      <code>first</code>\n"
                + "      String; optional\n"
                + "        <p>\n"
                + "          the first attribute\n"
                + "        </p>\n");
  }
}
