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
package com.google.devtools.build.lib.query2.output;

import com.google.devtools.build.lib.query2.CommonQueryOptions;
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
              + "are: label, textproto, transitions, proto. If you select 'transitions', you also "
              + "have to specify the --transitions=(lite|full) option.")
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
    name = "proto:include_configurations",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "if enabled, proto output will include information about configurations. When disabled,"
            + "cquery proto output format resembles query output format"
  )
  public boolean protoIncludeConfigurations;
}
