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

package com.google.devtools.build.skydoc.renderer;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

/** Contains options for running {@link RendererMain}. */
public class RendererOptions extends OptionsBase {

  @Option(
      name = "input",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help = "The path of the proto file that will be converted to markdown")
  public String inputPath;

  @Option(
      name = "output",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help = "The path of the file to output documentation into")
  public String outputFilePath;

  @Option(
      name = "header_template",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help =
          "The template for the header string. If the option is unspecified,"
              + " a default markdown output template will be used.")
  public String headerTemplateFilePath;

  @Option(
      name = "rule_template",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help =
          "The template for the documentation of a rule. If the option is unspecified, a"
              + " default markdown output template will be used.")
  public String ruleTemplateFilePath;

  @Option(
      name = "provider_template",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help =
          "The template for the documentation of a provider. If the option is"
              + " unspecified, a default markdown output template will be used.")
  public String providerTemplateFilePath;

  @Option(
      name = "func_template",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help =
          "The template for the documentation of a function. If the option is"
              + " unspecified, a default markdown output template will be used.")
  public String funcTemplateFilePath;

  @Option(
      name = "aspect_template",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help =
          "The template for the documentation of a aspect. If the option is unspecified, a"
              + " default markdown output template will be used.")
  public String aspectTemplateFilePath;
}
