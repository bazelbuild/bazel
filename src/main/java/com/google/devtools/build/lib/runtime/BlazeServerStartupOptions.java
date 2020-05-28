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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.Map;

/**
 * Options that will be evaluated by the blaze client startup code and passed to the blaze server
 * upon startup.
 *
 * <h4>IMPORTANT</h4>
 *
 * These options and their defaults must be kept in sync with those in the source of the launcher.
 * The latter define the actual default values, most startup options are passed every time,
 * regardless of whether a value was set explicitly or if the default was used. Some options are
 * omitted by default, though this should only be true for options where "omitted" is a distinct
 * value.
 *
 * <p>The same relationship holds between {@link HostJvmStartupOptions} and the launcher.
 */
public class BlazeServerStartupOptions extends OptionsBase {
  /**
   * Converter for the <code>option_sources</code> option. Takes a string in the form of
   * "option_name1:source1:option_name2:source2:.." and converts it into an option name to source
   * map.
   */
  public static class OptionSourcesConverter implements Converter<Map<String, String>> {
    private String unescape(String input) {
      return input.replace("_C", ":").replace("_U", "_");
    }

    @Override
    public Map<String, String> convert(String input) {
      ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
      if (input.isEmpty()) {
        return builder.build();
      }

      String[] elements = input.split(":");
      for (int i = 0; i < (elements.length + 1) / 2; i++) {
        String name = elements[i * 2];
        String value = "";
        if (elements.length > i * 2 + 1) {
          value = elements[i * 2 + 1];
        }
        builder.put(unescape(name), unescape(value));
      }
      return builder.build();
    }

    @Override
    public String getTypeDescription() {
      return "a list of option-source pairs";
    }
  }

  /* Passed from the client to the server, specifies the installation
   * location. The location should be of the form:
   * $OUTPUT_BASE/_blaze_${USER}/install/${MD5_OF_INSTALL_MANIFEST}.
   * The server code will only accept a non-empty path; it's the
   * responsibility of the client to compute a proper default if
   * necessary.
   */
  @Option(
      name = "install_base",
      defaultValue = "", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      metadataTags = {OptionMetadataTag.HIDDEN},
      converter = OptionsUtils.PathFragmentConverter.class,
      help = "This launcher option is intended for use only by tests.")
  public PathFragment installBase;

  /*
   * The installation MD5 - a content hash of the blaze binary (includes the Blaze deploy JAR and
   * any other embedded binaries - anything that ends up in the install_base).
   */
  @Option(
      name = "install_md5",
      defaultValue = "", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.BAZEL_MONITORING},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "This launcher option is intended for use only by tests.")
  public String installMD5;

  /* Note: The help string in this option applies to the client code; not
   * the server code. The server code will only accept a non-empty path; it's
   * the responsibility of the client to compute a proper default if
   * necessary.
   */
  @Option(
      name = "output_base",
      defaultValue = "null", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      converter = OptionsUtils.PathFragmentConverter.class,
      valueHelp = "<path>",
      help =
          "If set, specifies the output location to which all build output will be written. "
              + "Otherwise, the location will be "
              + "${OUTPUT_ROOT}/_blaze_${USER}/${MD5_OF_WORKSPACE_ROOT}. Note: If you specify a "
              + "different option from one to the next Bazel invocation for this value, you'll "
              + "likely start up a new, additional Bazel server. Bazel starts exactly one server "
              + "per specified output base. Typically there is one output base per workspace - "
              + "however, with this option you may have multiple output bases per workspace and "
              + "thereby run multiple builds for the same client on the same machine concurrently. "
              + "See 'bazel help shutdown' on how to shutdown a Bazel server.")
  public PathFragment outputBase;

  @Option(
      name = "output_user_root",
      defaultValue = "null", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      converter = OptionsUtils.PathFragmentConverter.class,
      valueHelp = "<path>",
      help =
          "The user-specific directory beneath which all build outputs are written; by default, "
              + "this is a function of $USER, but by specifying a constant, build outputs can be "
              + "shared between collaborating users.")
  public PathFragment outputUserRoot;

  /**
   * Note: This option is only used by the C++ client, never by the Java server. It is included here
   * to make sure that the option is documented in the help output, which is auto-generated by Java
   * code.
   */
  @Option(
      name = "server_jvm_out",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      converter = OptionsUtils.PathFragmentConverter.class,
      valueHelp = "<path>",
      help =
          "The location to write the server's JVM's output. If unset then defaults to a location "
              + "in output_base.")
  public PathFragment serverJvmOut;

  // Note: The help string in this option applies to the client code; not the server code. The
  // server code will only accept a non-empty path; it's the responsibility of the client to compute
  // a proper default if necessary.
  @Option(
      name = "failure_detail_out",
      defaultValue = "null", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      converter = OptionsUtils.PathFragmentConverter.class,
      valueHelp = "<path>",
      help =
          "If set, specifies a location to write a failure_detail protobuf message if the server"
              + " experiences a failure and cannot report it via gRPC, as normal. Otherwise, the"
              + " location will be ${OUTPUT_BASE}/failure_detail.rawproto.")
  public PathFragment failureDetailOut;

  @Option(
      name = "workspace_directory",
      defaultValue = "", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      metadataTags = {OptionMetadataTag.HIDDEN},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          "The root of the workspace, that is, the directory that Bazel uses as the root of the "
              + "build. This flag is only to be set by the bazel client.")
  public PathFragment workspaceDirectory;

  @Option(
      name = "default_system_javabase",
      defaultValue = "", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      metadataTags = {OptionMetadataTag.HIDDEN},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          "The root of the user's local JDK install, to be used as the default target javabase"
              + " and as a fall-back host_javabase. This is not the embedded JDK.")
  public PathFragment defaultSystemJavabase;

  @Option(
      name = "max_idle_secs",
      // NOTE: default value only used for documentation, value is always passed by the client when
      // not in --batch mode.
      defaultValue = "" + (3 * 3600),
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.EAGERNESS_TO_EXIT, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      valueHelp = "<integer>",
      help =
          "The number of seconds the build server will wait idling before shutting down. Zero "
              + "means that the server will never shutdown.")
  public int maxIdleSeconds;

  @Option(
      name = "shutdown_on_low_sys_mem",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.EAGERNESS_TO_EXIT, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "If max_idle_secs is set and the build server has been idle for a while, shut down the "
              + "server when the system is low on free RAM. Linux only.")
  public boolean shutdownOnLowSysMem;

  @Option(
      name = "batch",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION
      },
      metadataTags = {OptionMetadataTag.DEPRECATED},
      help =
          "If set, Bazel will be run as just a client process without a server, instead of in "
              + "the standard client/server mode. This is deprecated and will be removed, please "
              + "prefer shutting down the server explicitly if you wish to avoid lingering "
              + "servers.")
  public boolean batch;

  @Option(
      name = "deep_execroot",
      defaultValue = "true", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.EXECUTION},
      help =
          "If set, the execution root will be under $OUTPUT_BASE/execroot instead of "
              + "$OUTPUT_BASE.")
  public boolean deepExecRoot;

  @Option(
      name = "block_for_lock",
      defaultValue = "true", // NOTE: only for documentation, value never passed to the server.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.EAGERNESS_TO_EXIT},
      help =
          "When --noblock_for_lock is passed, Bazel does not wait for a running command to "
              + "complete, but instead exits immediately.")
  public boolean blockForLock;

  @Option(
      name = "io_nice_level",
      defaultValue = "-1", // NOTE: only for documentation, value never passed to the server.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      valueHelp = "{-1,0,1,2,3,4,5,6,7}",
      help =
          "Only on Linux; set a level from 0-7 for best-effort IO scheduling using the "
              + "sys_ioprio_set system call. 0 is highest priority, 7 is lowest. The anticipatory "
              + "scheduler may only honor up to priority 4. If set to a negative value, then Bazel "
              + "does not perform a system call.")
  public int ioNiceLevel;

  @Option(
      name = "batch_cpu_scheduling",
      defaultValue = "false", // NOTE: only for documentation, value never passed to the server.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      help =
          "Only on Linux; use 'batch' CPU scheduling for Blaze. This policy is useful for "
              + "workloads that are non-interactive, but do not want to lower their nice value. "
              + "See 'man 2 sched_setscheduler'. If false, then Bazel does not perform a system "
              + "call.")
  public boolean batchCpuScheduling;

  @Option(
      name = "ignore_all_rc_files",
      defaultValue = "false", // NOTE: purely decorative, rc files are read by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          "Disables all rc files, regardless of the values of other rc-modifying flags, even if "
              + "these flags come later in the list of startup options.")
  public boolean ignoreAllRcFiles;

  @Option(
      name = "fatal_event_bus_exceptions",
      defaultValue = "false", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.EAGERNESS_TO_EXIT, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      deprecationWarning = "Will be enabled by default and removed soon",
      help = "No-op: please use --fatal_event_bus_exceptions_exclusions instead")
  public boolean fatalEventBusExceptions;

  @Option(
      name = "option_sources",
      converter = OptionSourcesConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "")
  public Map<String, String> optionSources;

  // TODO(bazel-team): In order to make it easier to have local watchers in open source Bazel,
  // turn this into a non-startup option.
  @Option(
      name = "watchfs",
      defaultValue = "false", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = OptionMetadataTag.DEPRECATED,
      help =
          "If true, %{product} tries to use the operating system's file watch service for local "
              + "changes instead of scanning every file for a change.")
  public boolean watchFS;

  // This option is only passed in --batch mode. The value is otherwise passed as part of the
  // server request.
  @Option(
      name = "invocation_policy",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          "A base64-encoded-binary-serialized or text-formated "
              + "invocation_policy.InvocationPolicy proto. Unlike other options, it is an error to "
              + "specify --invocation_policy multiple times.")
  public String invocationPolicy;

  @Option(
      name = "command_port",
      defaultValue = "0",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION
      },
      help = "Port to start up the gRPC command server on. If 0, let the kernel choose.")
  public int commandPort;

  @Option(
      name = "product_name",
      defaultValue = "bazel", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.BAZEL_MONITORING
      },
      metadataTags = {OptionMetadataTag.HIDDEN},
      help =
          "The name of the build system. It is used as part of the name of the generated "
              + "directories (e.g. productName-bin for binaries) as well as for printing error "
              + "messages and logging")
  public String productName;

  // TODO(ulfjack): Make this a command option.
  @Option(
      name = "write_command_log",
      defaultValue = "true", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help = "Whether or not to write the command.log file")
  public boolean writeCommandLog;

  @Option(
      name = "client_debug",
      defaultValue = "false", // NOTE: only for documentation, value is set and used by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BAZEL_MONITORING},
      help =
          "If true, log debug information from the client to stderr. Changing this option will not "
              + "cause the server to restart.")
  public boolean clientDebug;

  @Option(
      name = "connect_timeout_secs",
      defaultValue = "30", // NOTE: only for documentation, value is set and used by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "The amount of time the client waits for each attempt to connect to the server")
  public int connectTimeoutSecs;

  // TODO(b/109764197): Add OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS & remove the
  // experimental tag once this has been tested and is ready for use.
  @Option(
      name = "digest_function",
      defaultValue = "null",
      converter = DigestHashFunction.DigestFunctionConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION
      },
      metadataTags = OptionMetadataTag.EXPERIMENTAL,
      help = "The hash function to use when computing file digests.")
  public DigestHashFunction digestHashFunction;

  @Deprecated
  @Option(
      name = "expand_configs_in_place",
      defaultValue = "true", // NOTE: only for documentation, value is always passed by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.NO_OP},
      metadataTags = {OptionMetadataTag.DEPRECATED},
      deprecationWarning = "This option is now a no-op and will soon be deleted.",
      help =
          "Changed the expansion of --config flags to be done in-place, as opposed to in a fixed "
              + "point expansion between normal rc options and command-line specified options.")
  public boolean expandConfigsInPlace;

  @Option(
      name = "idle_server_tasks",
      defaultValue = "true", // NOTE: only for documentation, value is set and used by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS,
      },
      help = "Run System.gc() when the server is idle")
  public boolean idleServerTasks;

  @Option(
      name = "unlimit_coredumps",
      defaultValue = "false", // NOTE: purely decorative, rc files are read by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
      },
      help =
          "Raises the soft coredump limit to the hard limit to make coredumps of the server"
              + " (including the JVM) and the client possible under common conditions. Stick this"
              + " flag in your bazelrc once and forget about it so that you get coredumps when you"
              + " actually encounter a condition that triggers them.")
  public boolean unlimitCoredumps;

  @Option(
      name = "macos_qos_class",
      defaultValue = "default", // Only for documentation; value is set and used by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {
        OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS,
      },
      help =
          "Sets the QoS service class of the %{product} server when running on macOS. This "
              + "flag has no effect on all other platforms but is supported to ensure rc files "
              + "can be shared among them without changes. Possible values are: user-interactive, "
              + "user-initiated, default, utility, and background.")
  public String macosQosClass;

  @Option(
      name = "incompatible_enable_execution_transition",
      defaultValue = "false", // Only for documentation; value is set by the client.
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If false, the execution transition behaves like the host transition.")
  public boolean enableExecutionTransition;

  @Option(
      name = "windows_enable_symlinks",
      defaultValue = "false", // Only for documentation; value is set by the client.
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "If true, real symbolic links will be created on Windows instead of file copying. "
              + "Requires Windows developer mode to be enabled and Windows 10 version 1703 or "
              + "greater.")
  public boolean enableWindowsSymlinks;
}
