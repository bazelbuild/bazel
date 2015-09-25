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
package com.google.devtools.build.lib.runtime.commands;

import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.ShutdownBlazeServerException;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

/**
 * The 'blaze shutdown' command.
 */
@Command(name = "shutdown",
         options = { ShutdownCommand.Options.class },
         allowResidue = false,
         mustRunInWorkspace = false,
         shortDescription = "Stops the %{product} server.",
         help = "This command shuts down the memory resident %{product} server process.\n"
              + "%{options}")
public final class ShutdownCommand implements BlazeCommand {

  public static class Options extends OptionsBase {

    @Option(name="iff_heap_size_greater_than",
            defaultValue = "0",
            category = "misc",
            help="Iff non-zero, then shutdown will only shut down the " +
                 "server if the total memory (in MB) consumed by the JVM " +
                 "exceeds this value.")
    public int heapSizeLimit;
  }

  @Override
  public void editOptions(CommandEnvironment env, OptionsParser optionsParser) {}

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options)
      throws ShutdownBlazeServerException {
    int limit = options.getOptions(Options.class).heapSizeLimit;

    // Iff limit is non-zero, shut down the server if total memory exceeds the
    // limit. totalMemory is the actual heap size that the VM currently uses
    // *from the OS perspective*. That is, it's not the size occupied by all
    // objects (which is totalMemory() - freeMemory()), and not the -Xmx
    // (which is maxMemory()). It's really how much memory this process
    // currently consumes, in addition to the JVM code and C heap.

    if (limit == 0 ||
        Runtime.getRuntime().totalMemory() > limit * 1000L * 1000) {
      throw new ShutdownBlazeServerException(0);
    }
    return ExitCode.SUCCESS;
  }

}
