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

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;

/**
 * Options that will be evaluated by the blaze client startup code only.
 *
 * The only reason we have this interface is that we'd like to print a nice
 * help page for the client startup options. These options do not affect the
 * server's behavior in any way.
 */
public class HostJvmStartupOptions extends OptionsBase {

  @Option(
      name = "server_javabase",
      defaultValue = "", // NOTE: purely decorative! See BlazeServerStartupOptions.
      valueHelp = "<jvm path>",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to the JVM used to execute Bazel itself.")
  public String serverJavabase;

  @Option(
      name = "host_jvm_args",
      defaultValue = "null", // NOTE: purely decorative!  See BlazeServerStartupOptions.
      allowMultiple = true,
      valueHelp = "<jvm_arg>",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Flags to pass to the JVM executing Blaze.")
  public List<String> hostJvmArgs;

  @Option(
      name = "host_jvm_profile",
      defaultValue = "", // NOTE: purely decorative!  See BlazeServerStartupOptions.
      valueHelp = "<profiler_name>",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Convenience option to add some profiler/debugger-specific JVM startup flags. "
              + "Bazel has a list of known values that it maps to hard-coded JVM startup flags, "
              + "possibly searching some hardcoded paths for certain files.")
  public String hostJvmProfile;

  @Option(
    name = "host_jvm_debug",
    defaultValue = "null", // NOTE: purely decorative!  See BlazeServerStartupOptions.
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Convenience option to add some additional JVM startup flags, which cause "
            + "the JVM to wait during startup until you connect from a JDWP-compliant debugger "
            + "(like Eclipse) to port 5005.",
    expansion = {
      "--host_jvm_args=-Xdebug",
      "--host_jvm_args=-Xrunjdwp:transport=dt_socket,server=y,address=5005",
    }
  )
  public Void hostJvmDebug;
}
