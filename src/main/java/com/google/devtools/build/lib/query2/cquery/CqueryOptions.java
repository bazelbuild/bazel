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
package com.google.devtools.build.lib.query2.cquery;

import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnum;
import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnumConverter;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;

/** Options class for cquery specific query options. */
public class CqueryOptions extends CommonQueryOptions {

  /** Converter for {@link CqueryOptions.Transitions} enum. */
  public static class TransitionsConverter extends EnumConverter<Transitions> {
    public TransitionsConverter() {
      super(Transitions.class, "transition verbosity");
    }
  }

  /** How much information to output about transitions. */
  public enum Transitions {
    FULL, /** includes everything in LITE plus transition's effect on options. */
    LITE, /** includes which attribute the transition is applied on and class name of transition */
    NONE /** default value, no transition information */
  }

  @Option(
      name = "output",
      defaultValue = "label",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "The format in which the cquery results should be printed. Allowed values for cquery "
              + "are: label, label_kind, textproto, transitions, proto, streamed_proto, jsonproto. "
              + "If you select 'transitions', you also have to specify the "
              + "--transitions=(lite|full) option.")
  public String outputFormat;

  @Option(
      name = "transitions",
      converter = TransitionsConverter.class,
      defaultValue = "none",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "The format in which cquery will print transition information."
  )
  public Transitions transitions;

  @Option(
      name = "show_config_fragments",
      defaultValue = "off",
      converter = IncludeConfigFragmentsEnumConverter.class,
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Shows the configuration fragments required by a rule and its transitive "
              + "dependencies. This can be useful for evaluating how much a configured target "
              + "graph can be trimmed.")
  // This implicitly sets the option --include_config_fragments_provider (see CoreOptions), which
  // makes configured targets compute the data cquery needs to enable this feature.
  public IncludeConfigFragmentsEnum showRequiredConfigFragments;

  @Option(
      name = "proto:include_configurations",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "if enabled, proto output will include information about configurations. When disabled,"
              + "cquery proto output format resembles query output format.")
  public boolean protoIncludeConfigurations;

  @Option(
      name = "starlark:expr",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "A Starlark expression to format each configured target in cquery's"
              + " --output=starlark mode. The configured target is bound to 'target'."
              + " If neither --starlark:expr nor --starlark:file is specified, this option will"
              + " default to 'str(target.label)'. It is an error to specify both --starlark:expr"
              + " and --starlark:file.")
  public String expr;

  @Option(
      name = "starlark:file",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "The name of a file that defines a Starlark function called 'format', of one argument,"
              + " that is applied to each configured target to format it as a string. It is an"
              + " error to specify both --starlark:expr and --starlark:file. See help for"
              + " --output=starlark for additional detail.")
  public String file;
}
