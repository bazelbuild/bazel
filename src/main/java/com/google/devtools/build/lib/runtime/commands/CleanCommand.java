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

import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.OutputDirectoryLinksUtils;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.ShutdownBlazeServerException;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Implements 'blaze clean'.
 */
@Command(name = "clean",
         builds = true,  // Does not, but people expect build options to be there
         options = { CleanCommand.Options.class },
         help = "resource:clean.txt",
         shortDescription = "Removes output files and optionally stops the server.",
         // TODO(bazel-team): Remove this - we inherit a huge number of unused options.
         inherits = { BuildCommand.class })
public final class CleanCommand implements BlazeCommand {

  /**
   * An interface for special options for the clean command.
   */
  public static class Options extends OptionsBase {
    @Option(name = "clean_style",
            defaultValue = "",
            category = "clean",
            help = "Can be either 'expunge' or 'expunge_async'.")
    public String cleanStyle;

    @Option(name = "expunge",
            defaultValue = "false",
            category = "clean",
            expansion = "--clean_style=expunge",
            help = "If specified, clean will remove the entire working tree for this %{product} "
                 + "instance, which includes all %{product}-created temporary and build output "
                 + "files, and it will stop the %{product} server if it is running.")
    public boolean expunge;

    @Option(name = "expunge_async",
        defaultValue = "false",
        category = "clean",
        expansion = "--clean_style=expunge_async",
        help = "If specified, clean will asynchronously remove the entire working tree for "
             + "this %{product} instance, which includes all %{product}-created temporary and "
             + "build output files, and it will stop the %{product} server if it is running. When "
             + "this command completes, it will be safe to execute new commands in the same "
             + "client, even though the deletion may continue in the background.")
    public boolean expunge_async;
  }

  private static Logger LOG = Logger.getLogger(CleanCommand.class.getName());

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options)
      throws ShutdownBlazeServerException {
    Options cleanOptions = options.getOptions(Options.class);
    cleanOptions.expunge_async = cleanOptions.cleanStyle.equals("expunge_async");
    cleanOptions.expunge = cleanOptions.cleanStyle.equals("expunge");

    env.getEventBus().post(new NoBuildEvent());

    if (!cleanOptions.expunge && !cleanOptions.expunge_async
        && !cleanOptions.cleanStyle.isEmpty()) {
      env.getReporter().handle(Event.error(
          null, "Invalid clean_style value '" + cleanOptions.cleanStyle + "'"));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    String cleanBanner = cleanOptions.expunge_async ?
        "Starting clean." :
        "Starting clean (this may take a while). " +
            "Consider using --expunge_async if the clean takes more than several minutes.";

    env.getReporter().handle(Event.info(null/*location*/, cleanBanner));
    try {
      String symlinkPrefix =
          options.getOptions(BuildRequest.BuildRequestOptions.class).getSymlinkPrefix();
      actuallyClean(env, env.getOutputBase(), cleanOptions, symlinkPrefix);
      return ExitCode.SUCCESS;
    } catch (IOException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
    } catch (CommandException | ExecException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return ExitCode.RUN_FAILURE;
    } catch (InterruptedException e) {
      env.getReporter().handle(Event.error("clean interrupted"));
      return ExitCode.INTERRUPTED;
    }
  }

  private void actuallyClean(CommandEnvironment env,
      Path outputBase, Options cleanOptions, String symlinkPrefix) throws IOException,
      ShutdownBlazeServerException, CommandException, ExecException, InterruptedException {
    BlazeRuntime runtime = env.getRuntime();
    if (env.getOutputService() != null) {
      env.getOutputService().clean();
    }
    if (cleanOptions.expunge) {
      LOG.info("Expunging...");
      // Delete the big subdirectories with the important content first--this
      // will take the most time. Then quickly delete the little locks, logs
      // and links right before we exit. Once the lock file is gone there will
      // be a small possibility of a server race if a client is waiting, but
      // all significant files will be gone by then.
      FileSystemUtils.deleteTreesBelow(outputBase);
      FileSystemUtils.deleteTree(outputBase);
    } else if (cleanOptions.expunge_async) {
      LOG.info("Expunging asynchronously...");
      String tempBaseName = outputBase.getBaseName() + "_tmp_" + ProcessUtils.getpid();

      // Keeping tempOutputBase in the same directory ensures it remains in the
      // same file system, and therefore the mv will be atomic and fast.
      Path tempOutputBase = outputBase.getParentDirectory().getChild(tempBaseName);
      outputBase.renameTo(tempOutputBase);
      env.getReporter().handle(Event.info(
          null, "Output base moved to " + tempOutputBase + " for deletion"));

      // Daemonize the shell and use the double-fork idiom to ensure that the shell
      // exits even while the "rm -rf" command continues.
      String command = String.format("exec >&- 2>&- <&- && (/usr/bin/setsid /bin/rm -rf %s &)&",
          ShellEscaper.escapeString(tempOutputBase.getPathString()));

      LOG.info("Executing shell commmand " + ShellEscaper.escapeString(command));

      // Doesn't throw iff command exited and was successful.
      new CommandBuilder().addArg(command).useShell(true)
        .setWorkingDir(tempOutputBase.getParentDirectory())
        .build().execute();
    } else {
      LOG.info("Output cleaning...");
      runtime.clearCaches();
      // In order to be sure that we delete everything, delete the workspace directory both for
      // --deep_execroot and for --nodeep_execroot.
      for (String directory : new String[] {
          env.getWorkspaceName(), "execroot/" + env.getWorkspaceName() }) {
        Path child = outputBase.getRelative(directory);
        if (child.exists()) {
          LOG.finest("Cleaning " + child);
          FileSystemUtils.deleteTreesBelow(child);
        }
      }
    }
    // remove convenience links
    OutputDirectoryLinksUtils.removeOutputDirectoryLinks(
        env.getWorkspaceName(), env.getWorkspace(), env.getReporter(), symlinkPrefix);
    // shutdown on expunge cleans
    if (cleanOptions.expunge || cleanOptions.expunge_async) {
      throw new ShutdownBlazeServerException(0);
    }
  }

  @Override
  public void editOptions(CommandEnvironment env, OptionsParser optionsParser) {}
}
