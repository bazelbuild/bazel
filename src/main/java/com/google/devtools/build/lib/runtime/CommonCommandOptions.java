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

import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;

/**
 * Options common to all commands.
 */
public class CommonCommandOptions extends OptionsBase {
  /**
   * A class representing a blazerc option. blazeRc is serial number of the rc
   * file this option came from, option is the name of the option and value is
   * its value (or null if not specified).
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

  /**
   * Converter for --default_override. The format is:
   * --default_override=blazerc:command=option.
   */
  public static class OptionOverrideConverter implements Converter<OptionOverride> {
    static final String ERROR_MESSAGE = "option overrides must be in form "
      + " rcfile:command=option, where rcfile is a nonzero integer";

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
        throw new OptionsParsingException(ERROR_MESSAGE);
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


  // To create a new incompatible change, see the javadoc for AllIncompatibleChangesExpansion.
  @Option(
    name = "all_incompatible_changes",
    defaultValue = "null",
    category = "misc",
    expansionFunction = AllIncompatibleChangesExpansion.class,
    help =
        "Enables all options of the form --incompatible_*. Use this option to find places where "
            + "your build may break in the future due to deprecations or other changes."
  )
  public Void allIncompatibleChanges;

  @Option(
    name = "config",
    defaultValue = "",
    category = "misc",
    allowMultiple = true,
    help =
        "Selects additional config sections from the rc files; for every <command>, it "
            + "also pulls in the options from <command>:<config> if such a section exists; "
            + "if the section does not exist, this flag is ignored. "
            + "Note that it is currently only possible to provide these options on the "
            + "command line, not in the rc files. The config sections and flag combinations "
            + "they are equivalent to are located in the tools/*.blazerc config files."
  )
  public List<String> configs;

  @Option(
    name = "logging",
    defaultValue = "3", // Level.INFO
    category = "verbosity",
    converter = Converters.LogLevelConverter.class,
    help = "The logging level."
  )
  public Level verbosity;

  @Option(
    name = "client_env",
    defaultValue = "",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    converter = Converters.AssignmentConverter.class,
    allowMultiple = true,
    help = "A system-generated parameter which specifies the client's environment"
  )
  public List<Map.Entry<String, String>> clientEnv;

  @Option(
    name = "ignore_client_env",
    defaultValue = "false",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    deprecationWarning = "Deprecated, no-op.",
    help = "Deprecated, no-op."
  )
  // TODO(laszlocsomor, dslomov) 2017-03-07: remove this flag after 2017-06-01 (~3 months from now)
  // and all of its occurrences.
  public boolean ignoreClientEnv;

  @Option(
    name = "client_cwd",
    defaultValue = "",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    converter = OptionsUtils.PathFragmentConverter.class,
    help = "A system-generated parameter which specifies the client's working directory"
  )
  public PathFragment clientCwd;

  @Option(
    name = "announce_rc",
    defaultValue = "false",
    category = "verbosity",
    help = "Whether to announce rc options."
  )
  public boolean announceRcOptions;

  /**
   * These are the actual default overrides. Each value is a pair of (command name, value).
   *
   * <p>For example: "--default_override=build=--cpu=piii"
   */
  @Option(
    name = "default_override",
    defaultValue = "",
    allowMultiple = true,
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    converter = OptionOverrideConverter.class,
    help = ""
  )
  public List<OptionOverride> optionsOverrides;

  /** This is the filename that the Blaze client parsed. */
  @Option(
    name = "rc_source",
    defaultValue = "",
    allowMultiple = true,
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    help = ""
  )
  public List<String> rcSource;

  @Option(
    name = "always_profile_slow_operations",
    defaultValue = "true",
    optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
    help = "Whether profiling slow operations is always turned on"
  )
  public boolean alwaysProfileSlowOperations;

  @Option(
    name = "allow_undefined_configs",
    defaultValue = "true",
    category = "flags",
    help = "Do not throw an error when the config is not defined."
  )
  public boolean allowUndefinedConfigs;

  @Option(
    name = "profile",
    defaultValue = "null",
    category = "misc",
    converter = OptionsUtils.PathFragmentConverter.class,
    help =
        "If set, profile Blaze and write data to the specified "
            + "file. Use blaze analyze-profile to analyze the profile."
  )
  public PathFragment profilePath;

  @Option(
    name = "record_full_profiler_data",
    defaultValue = "false",
    optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
    help =
        "By default, Blaze profiler will record only aggregated data for fast but numerous "
            + "events (such as statting the file). If this option is enabled, profiler will record "
            + "each event - resulting in more precise profiling data but LARGE performance "
            + "hit. Option only has effect if --profile used as well."
  )
  public boolean recordFullProfilerData;

  @Option(
    name = "memory_profile",
    defaultValue = "null",
    optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
    converter = OptionsUtils.PathFragmentConverter.class,
    help = "If set, write memory usage data to the specified " + "file at phase ends."
  )
  public PathFragment memoryProfilePath;

  @Option(
    name = "gc_watchdog",
    defaultValue = "false",
    optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
    deprecationWarning = "Ignoring: this option is no longer supported",
    help = "Deprecated."
  )
  public boolean gcWatchdog;

  @Option(
    name = "startup_time",
    defaultValue = "0",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    help = "The time in ms the launcher spends before sending the request to the blaze server."
  )
  public long startupTime;

  @Option(
    name = "extract_data_time",
    defaultValue = "0",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    help = "The time in ms spent on extracting the new blaze version."
  )
  public long extractDataTime;

  @Option(
    name = "command_wait_time",
    defaultValue = "0",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    help = "The time in ms a command had to wait on a busy Blaze server process."
  )
  public long waitTime;

  @Option(
    name = "tool_tag",
    defaultValue = "",
    category = "misc",
    help = "A tool name to attribute this Blaze invocation to."
  )
  public String toolTag;

  @Option(
    name = "restart_reason",
    defaultValue = "no_restart",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    help = "The reason for the server restart."
  )
  public String restartReason;

  @Option(
    name = "binary_path",
    defaultValue = "",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    help = "The absolute path of the blaze binary."
  )
  public String binaryPath;

  @Option(
    name = "experimental_allow_project_files",
    defaultValue = "false",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    help = "Enable processing of +<file> parameters."
  )
  public boolean allowProjectFiles;

  @Option(
    name = "block_for_lock",
    defaultValue = "true",
    optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
    help =
        "If set (the default), a command will block if there is another one running. If "
            + "unset, these commands will immediately return with an error."
  )
  public boolean blockForLock;

}
