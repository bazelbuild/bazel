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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.Map;

/**
 * Options that the Bazel client passes to server, which are then incorporated into the environment.
 *
 * <p>The rc file options are parsed in their own right and appear, if applicable, in the final
 * value of the parsed options. The environment variables update the stored values in the
 * CommandEnvironment. These options should never be accessed directly from this class after command
 * environment initialization.
 */
public class ClientOptions extends OptionsBase {
  /**
   * A class representing a blazerc option. blazeRc is serial number of the rc file this option came
   * from, option is the name of the option and value is its value (or null if not specified).
   */
  public static class OptionOverride {
    final int blazeRc;
    final String command;
    final String option;

    public OptionOverride(int blazeRc, String command, String option) {
      this.blazeRc = blazeRc;
      this.command = command;
      this.option = option;
    }

    @Override
    public String toString() {
      return String.format("%d:%s=%s", blazeRc, command, option);
    }
  }

  /** Converter for --default_override. The format is: --default_override=blazerc:command=option. */
  public static class OptionOverrideConverter implements Converter<OptionOverride> {
    static final String ERROR_MESSAGE =
        "option overrides must be in form rcfile:command=option, where rcfile is a nonzero integer";

    public OptionOverrideConverter() {}

    @Override
    public OptionOverride convert(String input) throws OptionsParsingException {
      int colonPos = input.indexOf(':');
      int assignmentPos = input.indexOf('=');

      if (colonPos < 0) {
        throw new OptionsParsingException(ERROR_MESSAGE);
      }

      if (assignmentPos <= colonPos + 1) {
        throw new OptionsParsingException(ERROR_MESSAGE);
      }

      int blazeRc;
      try {
        blazeRc = Integer.valueOf(input.substring(0, colonPos));
      } catch (NumberFormatException e) {
        throw new OptionsParsingException(ERROR_MESSAGE, e);
      }

      if (blazeRc < 0) {
        throw new OptionsParsingException(ERROR_MESSAGE);
      }

      String command = input.substring(colonPos + 1, assignmentPos);
      String option = input.substring(assignmentPos + 1);

      return new OptionOverride(blazeRc, command, option);
    }

    @Override
    public String getTypeDescription() {
      return "blazerc option override";
    }
  }

  @Option(
      name = "client_env",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      metadataTags = {OptionMetadataTag.HIDDEN},
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      converter = Converters.AssignmentConverter.class,
      allowMultiple = true,
      help = "A system-generated parameter which specifies the client's environment")
  public List<Map.Entry<String, String>> clientEnv;

  /**
   * These are the actual default overrides. Each value is a tuple of (bazelrc index, command name,
   * value). The blazerc index is a number used to find the blazerc in --rc_source's values.
   *
   * <p>For example: "--default_override=rc:build=--cpu=piii"
   */
  @Option(
      name = "default_override",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      metadataTags = {OptionMetadataTag.HIDDEN},
      converter = OptionOverrideConverter.class,
      help = "")
  public List<OptionOverride> optionsOverrides;

  /** This is the filename that the Blaze client parsed. */
  @Option(
      name = "rc_source",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "")
  public List<String> rcSource;
}
