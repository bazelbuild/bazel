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

import com.google.devtools.build.lib.runtime.CommandLineEvent.ToolCommandLineEvent;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;
import java.util.logging.Level;

/**
 * Options common to all commands.
 */
public class CommonCommandOptions extends OptionsBase {

  // To create a new incompatible change, see the javadoc for AllIncompatibleChangesExpansion.
  @Option(
    name = "all_incompatible_changes",
    defaultValue = "null",
    category = "misc",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
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
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
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
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    converter = Converters.LogLevelConverter.class,
    help = "The logging level."
  )
  public Level verbosity;

  @Option(
    name = "client_cwd",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    metadataTags = {OptionMetadataTag.HIDDEN},
    effectTags = {OptionEffectTag.CHANGES_INPUTS},
    converter = OptionsUtils.PathFragmentConverter.class,
    help = "A system-generated parameter which specifies the client's working directory"
  )
  public PathFragment clientCwd;

  @Option(
    name = "announce_rc",
    defaultValue = "false",
    category = "verbosity",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Whether to announce rc options."
  )
  public boolean announceRcOptions;

  @Option(
    name = "always_profile_slow_operations",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    help = "Whether profiling slow operations is always turned on"
  )
  public boolean alwaysProfileSlowOperations;

  @Option(
    name = "allow_undefined_configs",
    defaultValue = "true",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.EAGERNESS_TO_EXIT},
    help = "Do not throw an error when the config is not defined."
  )
  public boolean allowUndefinedConfigs;

  @Option(
    name = "profile",
    defaultValue = "null",
    category = "misc",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
    converter = OptionsUtils.PathFragmentConverter.class,
    help =
        "If set, profile Blaze and write data to the specified "
            + "file. Use blaze analyze-profile to analyze the profile."
  )
  public PathFragment profilePath;

  @Option(
    name = "record_full_profiler_data",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
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
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
    converter = OptionsUtils.PathFragmentConverter.class,
    help = "If set, write memory usage data to the specified file at phase ends."
  )
  public PathFragment memoryProfilePath;

  @Deprecated
  @Option(
    name = "gc_watchdog",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.NO_OP},
    metadataTags = {OptionMetadataTag.DEPRECATED},
    deprecationWarning = "Ignoring: this option is no longer supported",
    help = "Deprecated."
  )
  public boolean gcWatchdog;

  @Option(
    name = "experimental_oom_more_eagerly_threshold",
    defaultValue = "100",
    documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
    effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
    help =
        "If this flag is set to a value less than 100, Blaze will OOM if, after two full GC's, more"
            + "than this percentage of the (old gen) heap is still occupied."
  )
  public int oomMoreEagerlyThreshold;

  @Option(
    name = "startup_time",
    defaultValue = "0",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
    metadataTags = {OptionMetadataTag.HIDDEN},
    help = "The time in ms the launcher spends before sending the request to the blaze server."
  )
  public long startupTime;

  @Option(
    name = "extract_data_time",
    defaultValue = "0",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
    metadataTags = {OptionMetadataTag.HIDDEN},
    help = "The time in ms spent on extracting the new blaze version."
  )
  public long extractDataTime;

  @Option(
    name = "command_wait_time",
    defaultValue = "0",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
    metadataTags = {OptionMetadataTag.HIDDEN},
    help = "The time in ms a command had to wait on a busy Blaze server process."
  )
  public long waitTime;

  @Option(
    name = "tool_tag",
    defaultValue = "",
    category = "misc",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
    help = "A tool name to attribute this Blaze invocation to."
  )
  public String toolTag;

  @Option(
    name = "restart_reason",
    defaultValue = "no_restart",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
    metadataTags = {OptionMetadataTag.HIDDEN},
    help = "The reason for the server restart."
  )
  public String restartReason;

  @Option(
    name = "binary_path",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
    metadataTags = {OptionMetadataTag.HIDDEN},
    help = "The absolute path of the blaze binary."
  )
  public String binaryPath;

  @Option(
    name = "experimental_allow_project_files",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.CHANGES_INPUTS},
    metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.HIDDEN},
    help = "Enable processing of +<file> parameters."
  )
  public boolean allowProjectFiles;

  @Option(
    name = "block_for_lock",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    metadataTags = {OptionMetadataTag.HIDDEN},
    help =
        "If set (the default), a command will block if there is another one running. If "
            + "unset, these commands will immediately return with an error."
  )
  public boolean blockForLock;

  // We could accept multiple of these, in the event where there's a chain of tools that led to a
  // Bazel invocation. We would not want to expect anything from the order of these, and would need
  // to guarantee that the "label" for each command line is unique. Unless a need is demonstrated,
  // though, logs are a better place to track this information than flags, so let's try to avoid it.
  @Option(
    // In May 2018, this feature will have been out for 6 months. If the format we accept has not
    // changed in that time, we can remove the "experimental" prefix and tag.
    name = "experimental_tool_command_line",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    // Keep this flag HIDDEN so that it is not listed with our reported command lines, it being
    // reported separately.
    metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.HIDDEN},
    converter = ToolCommandLineEvent.Converter.class,
    help =
        "An extra command line to report with this invocation's command line. Useful for tools "
            + "that invoke Bazel and want the original information that the tool received to be "
            + "logged with the rest of the Bazel invocation."
  )
  public ToolCommandLineEvent toolCommandLine;

  @Option(
    name = "unconditional_warning",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    allowMultiple = true,
    help =
        "A warning that will unconditionally get printed with build warnings and errors. This is "
            + "useful to deprecate bazelrc files or --config definitions. If the intent is to "
            + "effectively deprecate some flag or combination of flags, this is NOT sufficient. "
            + "The flag or flags should use the deprecationWarning field in the option definition, "
            + "or the bad combination should be checked for programmatically."
  )
  public List<String> deprecationWarnings;
}
