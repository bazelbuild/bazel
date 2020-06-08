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

import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.devtools.build.lib.profiler.MemoryProfiler.MemoryProfileStableHeapParameters;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.runtime.CommandLineEvent.ToolCommandLineEvent;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Converters.AssignmentConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.logging.Level;

/** Options common to all commands. */
public class CommonCommandOptions extends OptionsBase {

  /**
   * To create a new incompatible change, see the javadoc for {@link
   * AllIncompatibleChangesExpansion}.
   */
  @Option(
      name = "all_incompatible_changes",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      expansionFunction = AllIncompatibleChangesExpansion.class,
      help =
          "Enables all options of the form --incompatible_*. Use this option to find places where "
              + "your build may break in the future due to deprecations or other changes.")
  public Void allIncompatibleChanges;

  @Option(
      name = "enable_platform_specific_config",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "If true, Bazel picks up host-OS-specific config lines from bazelrc files. For example, "
              + "if the host OS is Linux and you run bazel build, Bazel picks up lines starting "
              + "with build:linux. Supported OS identifiers are linux, macos, windows, freebsd, "
              + "and openbsd. Enabling this flag is equivalent to using --config=linux on Linux, "
              + "--config=windows on Windows, etc.")
  public boolean enablePlatformSpecificConfig;

  @Option(
      name = "config",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = true,
      help =
          "Selects additional config sections from the rc files; for every <command>, it "
              + "also pulls in the options from <command>:<config> if such a section exists; "
              + "if this section doesn't exist in any .rc file, Blaze fails with an error. "
              + "The config sections and flag combinations they are equivalent to are "
              + "located in the tools/*.blazerc config files.")
  public List<String> configs;

  @Option(
      name = "logging",
      defaultValue = "3", // Level.INFO
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      converter = Converters.LogLevelConverter.class,
      help = "The logging level.")
  public Level verbosity;

  @Option(
      name = "client_cwd",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      metadataTags = {OptionMetadataTag.HIDDEN},
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      converter = OptionsUtils.PathFragmentConverter.class,
      help = "A system-generated parameter which specifies the client's working directory")
  public PathFragment clientCwd;

  @Option(
      name = "announce_rc",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Whether to announce rc options.")
  public boolean announceRcOptions;

  @Option(
      name = "always_profile_slow_operations",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "Whether profiling slow operations is always turned on")
  public boolean alwaysProfileSlowOperations;

  /** Converter for UUID. Accepts values as specified by {@link UUID#fromString(String)}. */
  public static class UUIDConverter implements Converter<UUID> {

    @Override
    public UUID convert(String input) throws OptionsParsingException {
      if (isNullOrEmpty(input)) {
        return null;
      }
      try {
        return UUID.fromString(input);
      } catch (IllegalArgumentException e) {
        throw new OptionsParsingException(
            String.format("Value '%s' is not a value UUID.", input), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a UUID";
    }
  }

  /**
   * Converter for options (--build_request_id) that accept prefixed UUIDs. Since we do not care
   * about the structure of this value after validation, we store it as a string.
   */
  public static class PrefixedUUIDConverter implements Converter<String> {

    @Override
    public String convert(String input) throws OptionsParsingException {
      if (isNullOrEmpty(input)) {
        return null;
      }
      // UUIDs that are accepted by UUID#fromString have 36 characters. Interpret the last 36
      // characters as an UUID and the rest as a prefix. We do not check anything about the contents
      // of the prefix.
      try {
        int uuidStartIndex = input.length() - 36;
        UUID.fromString(input.substring(uuidStartIndex));
      } catch (IllegalArgumentException | IndexOutOfBoundsException e) {
        throw new OptionsParsingException(
            String.format("Value '%s' does not end in a valid UUID.", input), e);
      }
      return input;
    }

    @Override
    public String getTypeDescription() {
      return "An optionally prefixed UUID. The last 36 characters will be verified as a UUID.";
    }
  }

  // Command ID and build request ID can be set either by flag or environment variable. In most
  // cases, the internally generated ids should be sufficient, but we allow these to be set
  // externally if required. Option wins over environment variable, if both are set.
  // TODO(b/67895628) Stop reading ids from the environment after the compatibility window has
  // passed.
  @Option(
      name = "invocation_id",
      defaultValue = "",
      converter = UUIDConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_MONITORING, OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "Unique identifier, in UUID format, for the command being run. If explicitly specified"
              + " uniqueness must be ensured by the caller. The UUID is printed to stderr, the BEP"
              + " and remote execution protocol.")
  public UUID invocationId;

  @Option(
      name = "build_request_id",
      defaultValue = "",
      converter = PrefixedUUIDConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_MONITORING, OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "Unique string identifier for the build being run.")
  public String buildRequestId;

  @Option(
      name = "build_metadata",
      converter = AssignmentConverter.class,
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help = "Custom key-value string pairs to supply in a build event.")
  public List<Map.Entry<String, String>> buildMetadata;

  @Option(
      name = "oom_message",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_MONITORING, OptionEffectTag.TERMINAL_OUTPUT},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "Custom message to be emitted on an out of memory failure.")
  public String oomMessage;

  @Option(
      name = "incompatible_enable_profile_by_default",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If enabled, Bazel will generate a JSON profile by default.")
  public boolean enableProfileByDefault;

  @Option(
      name = "experimental_generate_json_trace_profile",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help =
          "If enabled, Bazel profiles the build and writes a JSON-format profile into a file in"
              + " the output base. View profile by loading into chrome://tracing.")
  public boolean enableTracer;

  @Option(
      name = "json_trace_compression",
      oldName = "experimental_json_trace_compression",
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help =
          "If enabled, Bazel compresses the JSON-format profile with gzip. "
              + "By default, this is decided based on the extension of the file specified in "
              + "--profile.")
  public TriState enableTracerCompression;

  @Option(
      name = "experimental_profile_cpu_usage",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help = "If set, Bazel will measure cpu usage and add it to the JSON profile.")
  public boolean enableCpuUsageProfiling;

  @Option(
      name = "experimental_profile_additional_tasks",
      converter = ProfilerTaskConverter.class,
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help = "Specifies additional profile tasks to be included in the profile.")
  public List<ProfilerTask> additionalProfileTasks;

  @Option(
      name = "slim_profile",
      oldName = "experimental_slim_json_profile",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help =
          "Slims down the size of the JSON profile by merging events if the profile gets "
              + " too large.")
  public boolean slimProfile;

  @Option(
      name = "experimental_profile_include_primary_output",
      oldName = "experimental_include_primary_output",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help =
          "Includes the extra \"out\" attribute in action events that contains the exec path "
              + "to the action's primary output.")
  public boolean includePrimaryOutput;

  @Option(
      name = "experimental_profile_include_target_label",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help = "Includes target label in action events' JSON profile data.")
  public boolean profileIncludeTargetLabel;

  @Option(
      name = "experimental_announce_profile_path",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help = "If enabled, adds the JSON profile path to the log.")
  public boolean announceProfilePath;

  @Option(
      name = "profile",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          "If set, profile Bazel and write data to the specified "
              + "file. Use bazel analyze-profile to analyze the profile.")
  public PathFragment profilePath;

  @Option(
      name = "starlark_cpu_profile",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.BAZEL_MONITORING},
      help = "Writes into the specified file a pprof profile of CPU usage by all Starlark threads.")
  public String starlarkCpuProfile;

  @Option(
      name = "record_full_profiler_data",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help =
          "By default, Bazel profiler will record only aggregated data for fast but numerous "
              + "events (such as statting the file). If this option is enabled, profiler will "
              + "record each event - resulting in more precise profiling data but LARGE "
              + "performance hit. Option only has effect if --profile used as well.")
  public boolean recordFullProfilerData;

  @Option(
      name = "memory_profile",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          "If set, write memory usage data to the specified file at phase ends and stable heap to"
              + " master log at end of build.")
  public PathFragment memoryProfilePath;

  @Option(
      name = "memory_profile_stable_heap_parameters",
      defaultValue = "1,0",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.BAZEL_MONITORING},
      converter = MemoryProfileStableHeapParameters.Converter.class,
      help =
          "Tune memory profile's computation of stable heap at end of build. Should be two"
              + " integers separated by a comma. First parameter is the number of GCs to perform."
              + " Second parameter is the number of seconds to wait between GCs.")
  public MemoryProfileStableHeapParameters memoryProfileStableHeapParameters;

  @Option(
      name = "experimental_oom_more_eagerly_threshold",
      defaultValue = "100",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      help =
          "If this flag is set to a value less than 100, Bazel will OOM if, after two full GC's, "
              + "more than this percentage of the (old gen) heap is still occupied.")
  public int oomMoreEagerlyThreshold;

  @Option(
      name = "startup_time",
      defaultValue = "0",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "The time in ms the launcher spends before sending the request to the bazel server.")
  public long startupTime;

  @Option(
      name = "extract_data_time",
      defaultValue = "0",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "The time in ms spent on extracting the new bazel version.")
  public long extractDataTime;

  @Option(
      name = "command_wait_time",
      defaultValue = "0",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "The time in ms a command had to wait on a busy Bazel server process.")
  public long waitTime;

  @Option(
      name = "tool_tag",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help = "A tool name to attribute this Bazel invocation to.")
  public String toolTag;

  @Option(
      name = "restart_reason",
      defaultValue = "no_restart",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "The reason for the server restart.")
  public String restartReason;

  @Option(
      name = "binary_path",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "The absolute path of the bazel binary.")
  public String binaryPath;

  @Option(
      name = "experimental_allow_project_files",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.HIDDEN},
      help = "Enable processing of +<file> parameters.")
  public boolean allowProjectFiles;

  @Option(
      name = "block_for_lock",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help =
          "If set (the default), a command will block if there is another one running. If "
              + "unset, these commands will immediately return with an error.")
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
              + "logged with the rest of the Bazel invocation.")
  public ToolCommandLineEvent toolCommandLine;

  @Option(
      name = "unconditional_warning",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      allowMultiple = true,
      help =
          "A warning that will unconditionally get printed with build warnings and errors. This is"
              + " useful to deprecate bazelrc files or --config definitions. If the intent is to"
              + " effectively deprecate some flag or combination of flags, this is NOT sufficient."
              + " The flag or flags should use the deprecationWarning field in the option"
              + " definition, or the bad combination should be checked for programmatically.")
  public List<String> deprecationWarnings;

  @Option(
      name = "track_incremental_state",
      oldName = "keep_incrementality_data",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "If false, Blaze will not persist data that allows for invalidation and re-evaluation "
              + "on incremental builds in order to save memory on this build. Subsequent builds "
              + "will not have any incrementality with respect to this one. Usually you will want "
              + "to specify --batch when setting this to false.")
  public boolean trackIncrementalState;

  @Option(
      name = "keep_state_after_build",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "If false, Blaze will discard the inmemory state from this build when the build "
              + "finishes. Subsequent builds will not have any incrementality with respect to this "
              + "one.")
  public boolean keepStateAfterBuild;

  /** The option converter to check that the user can only specify legal profiler tasks. */
  public static class ProfilerTaskConverter extends EnumConverter<ProfilerTask> {
    public ProfilerTaskConverter() {
      super(ProfilerTask.class, "profiler task");
    }
  }
}
