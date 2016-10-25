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
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import java.util.Map;

/**
 * Options that will be evaluated by the blaze client startup code and passed
 * to the blaze server upon startup.
 *
 * <h4>IMPORTANT</h4> These options and their defaults must be kept in sync with those in the
 * source of the launcher.  The latter define the actual default values; this class exists only to
 * provide the help message, which displays the default values.
 *
 * The same relationship holds between {@link HostJvmStartupOptions} and the launcher.
 */
public class BlazeServerStartupOptions extends OptionsBase {
  /**
   * Converter for the <code>option_sources</code> option. Takes a string in the form of
   * "option_name1:source1:option_name2:source2:.." and converts it into an option name to
   * source map.
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
  @Option(name = "install_base",
      defaultValue = "", // NOTE: purely decorative!  See class docstring.
      category = "hidden",
      converter = OptionsUtils.PathFragmentConverter.class,
      help = "This launcher option is intended for use only by tests.")
  public PathFragment installBase;

  /*
   * The installation MD5 - a content hash of the blaze binary (includes the Blaze deploy JAR and
   * any other embedded binaries - anything that ends up in the install_base).
   */
  @Option(name = "install_md5",
      defaultValue = "", // NOTE: purely decorative!  See class docstring.
      category = "hidden",
      help = "This launcher option is intended for use only by tests.")
  public String installMD5;

  /* Note: The help string in this option applies to the client code; not
   * the server code. The server code will only accept a non-empty path; it's
   * the responsibility of the client to compute a proper default if
   * necessary.
   */
  @Option(name = "output_base",
      defaultValue = "null", // NOTE: purely decorative!  See class docstring.
      category = "server startup",
      converter = OptionsUtils.PathFragmentConverter.class,
      valueHelp = "<path>",
      help = "If set, specifies the output location to which all build output will be written. "
          + "Otherwise, the location will be "
          + "${OUTPUT_ROOT}/_blaze_${USER}/${MD5_OF_WORKSPACE_ROOT}. Note: If you specify a "
          + "different option from one to the next Blaze invocation for this value, you'll likely "
          + "start up a new, additional Blaze server. Blaze starts exactly one server per "
          + "specified output base. Typically there is one output base per workspace - however, "
          + "with this option you may have multiple output bases per workspace and thereby run "
          + "multiple builds for the same client on the same machine concurrently. See "
          + "'blaze help shutdown' on how to shutdown a Blaze server.")
  public PathFragment outputBase;

  /* Note: This option is only used by the C++ client, never by the Java server.
   * It is included here to make sure that the option is documented in the help
   * output, which is auto-generated by Java code.
   */
  @Option(name = "output_user_root",
      defaultValue = "null", // NOTE: purely decorative!  See class docstring.
      category = "server startup",
      converter = OptionsUtils.PathFragmentConverter.class,
      valueHelp = "<path>",
      help = "The user-specific directory beneath which all build outputs are written; "
          + "by default, this is a function of $USER, but by specifying a constant, build outputs "
          + "can be shared between collaborating users.")
  public PathFragment outputUserRoot;

  @Option(name = "workspace_directory",
      defaultValue = "",
      category = "hidden",
      converter = OptionsUtils.PathFragmentConverter.class,
      help = "The root of the workspace, that is, the directory that Blaze uses as the root of the "
          + "build. This flag is only to be set by the blaze client.")
  public PathFragment workspaceDirectory;

  @Option(name = "max_idle_secs",
      defaultValue = "" + (3 * 3600), // NOTE: purely decorative!  See class docstring.
      category = "server startup",
      valueHelp = "<integer>",
      help = "The number of seconds the build server will wait idling before shutting down. Zero "
          + "means that the server will never shutdown.")
  public int maxIdleSeconds;

  @Option(name = "batch",
      defaultValue = "false", // NOTE: purely decorative!  See class docstring.
      category = "server startup",
      help = "If set, Blaze will be run in batch mode, instead of the standard client/server mode. "
          + "Running in batch mode prevents Blaze from caching data in memory, which makes it a "
          + "lot slower. We recommend against using this flag.")
  public boolean batch;

  @Option(name = "deep_execroot",
      defaultValue = "true", // NOTE: purely decorative!  See class docstring.
      category = "server startup",
      help = "If set, the execution root will be under $OUTPUT_BASE/execroot instead of "
          + "$OUTPUT_BASE.")
  public boolean deepExecRoot;

  @Option(
    name = "experimental_oom_more_eagerly",
    defaultValue = "false", // NOTE: purely decorative!  See class docstring.
    category = "server startup",
    help = "If set, attempt to detect Java heap OOM conditions and exit before thrashing. Only "
        + "honored when --batch is also passed. In some cases, builds that previously succeeded "
        + "may OOM if they were close to OOMing before.")
  public boolean oomMoreEagerly;

  @Option(
    name = "experimental_oom_more_eagerly_threshold",
    defaultValue = "100", // NOTE: purely decorative!  See class docstring.
    category = "server startup",
    help = "If this flag is set, Blaze will OOM if, after two full GC's, more than this "
        + "percentage of the (old gen) heap is still occupied.")
  public int oomMoreEagerlyThreshold;

  @Option(name = "block_for_lock",
      defaultValue = "true", // NOTE: purely decorative!  See class docstring.
      category = "server startup",
      help = "When --noblock_for_lock is passed, Blaze does not wait for a running command to "
          + "complete, but instead exits immediately.")
  public boolean blockForLock;

  @Option(name = "io_nice_level",
      defaultValue = "-1",  // NOTE: purely decorative!
      category = "server startup",
      valueHelp = "{-1,0,1,2,3,4,5,6,7}",
      help = "Only on Linux; set a level from 0-7 for best-effort IO scheduling using the "
          + "sys_ioprio_set system call. 0 is highest priority, 7 is lowest. The anticipatory "
          + "scheduler may only honor up to priority 4. If set to a negative value, then Blaze "
          + "does not perform a system call.")
  public int ioNiceLevel;

  @Option(name = "batch_cpu_scheduling",
      defaultValue = "false",  // NOTE: purely decorative!
      category = "server startup",
      help = "Only on Linux; use 'batch' CPU scheduling for Blaze. This policy is useful for "
          + "workloads that are non-interactive, but do not want to lower their nice value. "
          + "See 'man 2 sched_setscheduler'. If false, then Blaze does not perform a system call.")
  public boolean batchCpuScheduling;

  @Option(name = "blazerc",
      defaultValue = "null", // NOTE: purely decorative!
      category = "misc",
      valueHelp = "<path>",
      help = "The location of the .%{product}rc file containing default values of "
          + "Blaze command options. By default, Blaze first checks the current directory, then "
          + "the user's home directory, and then looks for a file named .$(basename $0)rc "
          + "(i.e. .%{product}rc). Use /dev/null to disable the search for a %{product}rc file, "
          + "e.g. in release builds.")
  public String blazerc;

  @Option(name = "master_blazerc",
      defaultValue = "true",  // NOTE: purely decorative!
      category = "misc",
      help = "If this option is false, the master %{product}rc next to the binary is not read.")
  public boolean masterBlazerc;

  @Option(name = "fatal_event_bus_exceptions",
      defaultValue = "false",  // NOTE: purely decorative!
      category = "undocumented",
      help = "Whether or not to exit if an exception is thrown by an internal EventBus handler.")
  public boolean fatalEventBusExceptions;

  @Option(name = "option_sources",
      converter = OptionSourcesConverter.class,
      defaultValue = "",
      category = "hidden",
      help = "")
  public Map<String, String> optionSources;

  // TODO(bazel-team): In order to make it easier to have local watchers in open source Bazel,
  // turn this into a non-startup option.
  @Option(
    name = "watchfs",
    defaultValue = "false",
    category = "server startup",
    help =
        "If true, %{product} tries to use the operating system's file watch service for local "
            + "changes instead of scanning every file for a change."
  )
  public boolean watchFS;

  @Option(name = "invocation_policy",
      defaultValue = "",
      category = "undocumented",
      help = "A base64-encoded-binary-serialized or text-formated "
          + "invocation_policy.InvocationPolicy proto. Unlike other options, it is an error to "
          + "specify --invocation_policy multiple times.")
  public String invocationPolicy;

  @Option(name = "command_port",
      defaultValue = "0",
      category = "undocumented",
      help = "Port to start up the gRPC command server on. If 0, let the kernel choose.")
  public int commandPort;

  @Option(name = "product_name",
      defaultValue = "bazel", // NOTE: purely decorative!
      category = "hidden",
      help = "The name of the build system. It is used as part of the name of the generated "
          + "directories (e.g. productName-bin for binaries) as well as for printing error "
          + "messages and logging")
  public String productName;

  @Option(name = "exoblaze",
      defaultValue = "false",
      category = "server startup",
      help = "If true, Blaze runs as Exoblaze")
  public boolean exoblaze;

  @Option(name = "write_command_log",
      defaultValue = "true",
      category = "undocumented",
      help = "Whether or not to write the command.log file")
  public boolean writeCommandLog;
}
