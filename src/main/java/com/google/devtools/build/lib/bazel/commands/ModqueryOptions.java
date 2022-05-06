// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.commands;

import com.google.common.base.Splitter;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.commands.ModqueryCommand.TargetModule;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;

/** Options for ModqueryCommand */
public class ModqueryOptions extends OptionsBase {

  @Option(
      name = "modq_unused",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.MODQUERY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Takes into account the modules that were initially part of the dependency graph, but"
              + " were removed during module resolution. The exact effect depends on each query"
              + " type.")
  public boolean unused;

  @Option(
      name = "output",
      defaultValue = "text",
      documentationCategory = OptionDocumentationCategory.MODQUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "The format in which the query results should be printed. Allowed values for query are: "
              + "text, graph")
  public String outputFormat;

  static class TargetModuleConverter implements Converter<TargetModule> {

    @Override
    public TargetModule convert(String input) throws OptionsParsingException {
      String errorMessage = String.format("Cannot parse the given module argument: %s.", input);
      if (input == null) {
        throw new OptionsParsingException(errorMessage);
      } else if (input.equalsIgnoreCase("root")) {
        return TargetModule.create("", Version.EMPTY);
      } else {
        List<String> splits = Splitter.on('@').splitToList(input);
        if (splits.size() == 0 || splits.get(0).isEmpty()) {
          throw new OptionsParsingException(errorMessage);
        }

        if (splits.size() == 2) {
          if (splits.get(1).equals("_")) {
            return TargetModule.create(splits.get(0), Version.EMPTY);
          }
          try {
            return TargetModule.create(splits.get(0), Version.parse(splits.get(1)));
          } catch (ParseException e) {
            throw new OptionsParsingException(errorMessage, e);
          }

        } else if (splits.size() == 1) {
          return TargetModule.create(splits.get(0), null);
        } else {
          throw new OptionsParsingException(errorMessage);
        }
      }
    }

    @Override
    public String getTypeDescription() {
      return "root, <module>@<version> or <module>";
    }
  }

  enum QueryType {
    DEPS,
    TRANSITIVE_DEPS,
    ALL_PATHS,
    PATH,
    EXPLAIN
  }
}
