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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.OutputDirectoryLinksUtils;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.logging.LogManager;
import java.util.logging.Logger;

/** Implements 'blaze clean'. */
@Command(
  name = "clean",
  builds = true, // Does not, but people expect build options to be there
  writeCommandLog = false, // Do not create a command.log, otherwise we couldn't delete it.
  options = {CleanCommand.Options.class},
  help = "resource:clean.txt",
  shortDescription = "Removes output files and optionally stops the server.",
  // TODO(bazel-team): Remove this - we inherit a huge number of unused options.
  inherits = {BuildCommand.class}
)
public final class CleanCommand implements BlazeCommand {
  /** An interface for special options for the clean command. */
  public static class Options extends OptionsBase {
    @Option(
      name = "expunge",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      help =
          "If true, clean removes the entire working tree for this %{product} instance, "
              + "which includes all %{product}-created temporary and build output files, "
              + "and stops the %{product} server if it is running."
    )
    public boolean expunge;

    @Option(
      name = "expunge_async",
      defaultValue = "null",
      expansion = {"--expunge", "--async"},
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      help =
          "If specified, clean asynchronously removes the entire working tree for "
              + "this %{product} instance, which includes all %{product}-created temporary and "
              + "build output files, and stops the %{product} server if it is running. When "
              + "this command completes, it will be safe to execute new commands in the same "
              + "client, even though the deletion may continue in the background."
    )
    public Void expungeAsync;

    @Option(
      name = "async",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      help =
          "If true, output cleaning is asynchronous. When this command completes, it will be safe "
              + "to execute new commands in the same client, even though the deletion may continue "
              + "in the background."
    )
    public boolean async;
  }

  /** Posted on the public event stream to announce that a clean is happening. */
  public static class CleanStartingEvent {
    private final OptionsParsingResult optionsParsingResult;

    public CleanStartingEvent(OptionsParsingResult optionsParsingResult) {
      this.optionsParsingResult = optionsParsingResult;
    }

    public OptionsParsingResult getOptionsProvider() {
      return optionsParsingResult;
    }
  }

  private final OS os;

  public CleanCommand() {
    this(OS.getCurrent());
  }

  @VisibleForTesting
  public CleanCommand(OS os) {
    this.os = os;
  }

  private static final Logger logger = Logger.getLogger(CleanCommand.class.getName());

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    Options cleanOptions = options.getOptions(Options.class);
    boolean async = cleanOptions.async;
    env.getEventBus().post(new NoBuildEvent());

    // TODO(dmarting): Deactivate expunge_async on non-Linux platform until we completely fix it
    // for non-Linux platforms (https://github.com/bazelbuild/bazel/issues/1906).
    // MacOS and FreeBSD support setsid(2) but don't have /usr/bin/setsid, so if we wanted to
    // support --expunge_async on these platforms, we'd have to write a wrapper that calls setsid(2)
    // and exec(2).
    boolean asyncSupport = os == OS.LINUX;
    if (async && !asyncSupport) {
      String fallbackName = cleanOptions.expunge ? "--expunge" : "synchronous clean";
      env.getReporter()
          .handle(
              Event.info(
                  null /*location*/,
                  "--async cannot be used on non-Linux platforms, falling back to "
                      + fallbackName));
      async = false;
    }

    String cleanBanner =
        (async || !asyncSupport)
            ? "Starting clean."
            : "Starting clean (this may take a while). "
                + "Consider using --async if the clean takes more than several minutes.";

    env.getEventBus().post(new CleanStartingEvent(options));
    env.getReporter().handle(Event.info(null /*location*/, cleanBanner));

    try {
      String symlinkPrefix =
          options
              .getOptions(BuildRequestOptions.class)
              .getSymlinkPrefix(env.getRuntime().getProductName());
      return actuallyClean(env, env.getOutputBase(), cleanOptions.expunge, async, symlinkPrefix);
    } catch (IOException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return BlazeCommandResult.exitCode(ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    } catch (CommandException | ExecException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return BlazeCommandResult.exitCode(ExitCode.RUN_FAILURE);
    } catch (InterruptedException e) {
      env.getReporter().handle(Event.error("clean interrupted"));
      return BlazeCommandResult.exitCode(ExitCode.INTERRUPTED);
    }
  }

  private static void asyncClean(CommandEnvironment env, Path path, String pathItemName)
      throws IOException, CommandException {
    String tempBaseName = path.getBaseName() + "_tmp_" + ProcessUtils.getpid();

    // Keeping tempOutputBase in the same directory ensures it remains in the
    // same file system, and therefore the mv will be atomic and fast.
    Path tempPath = path.getParentDirectory().getChild(tempBaseName);
    path.renameTo(tempPath);
    env.getReporter()
        .handle(Event.info(null, pathItemName + " moved to " + tempPath + " for deletion"));

    // Daemonize the shell and use the double-fork idiom to ensure that the shell
    // exits even while the "rm -rf" command continues.
    String command =
        String.format(
            "exec >&- 2>&- <&- && (/usr/bin/setsid /bin/rm -rf %s &)&",
            ShellEscaper.escapeString(tempPath.getPathString()));

    logger.info("Executing shell command " + ShellEscaper.escapeString(command));

    // Doesn't throw iff command exited and was successful.
    new CommandBuilder()
        .addArg(command)
        .useShell(true)
        .setWorkingDir(tempPath.getParentDirectory())
        .build()
        .execute();
  }

  private BlazeCommandResult actuallyClean(
      CommandEnvironment env, Path outputBase, boolean expunge, boolean async, String symlinkPrefix)
      throws IOException, CommandException, ExecException,
          InterruptedException {
    BlazeRuntime runtime = env.getRuntime();
    String workspaceDirectory = env.getWorkspace().getBaseName();
    if (env.getOutputService() != null) {
      env.getOutputService().clean();
    }
    env.getBlazeWorkspace().clearCaches();
    if (expunge && !async) {
      logger.info("Expunging...");
      runtime.prepareForAbruptShutdown();
      // Close java.log.
      LogManager.getLogManager().reset();
      // Close the default stdout/stderr.
      if (FileDescriptor.out.valid()) {
        new FileOutputStream(FileDescriptor.out).close();
      }
      if (FileDescriptor.err.valid()) {
        new FileOutputStream(FileDescriptor.err).close();
      }
      // Close the redirected stdout/stderr.
      System.out.close();
      System.err.close();
      // Delete the big subdirectories with the important content first--this
      // will take the most time. Then quickly delete the little locks, logs
      // and links right before we exit. Once the lock file is gone there will
      // be a small possibility of a server race if a client is waiting, but
      // all significant files will be gone by then.
      outputBase.deleteTreesBelow();
      outputBase.deleteTree();
    } else if (expunge && async) {
      logger.info("Expunging asynchronously...");
      runtime.prepareForAbruptShutdown();
      asyncClean(env, outputBase, "Output base");
    } else {
      logger.info("Output cleaning...");
      env.getBlazeWorkspace().resetEvaluator();
      Path execroot = outputBase.getRelative("execroot");
      if (execroot.exists()) {
        logger.finest("Cleaning " + execroot + (async ? " asynchronously..." : ""));
        if (async) {
          asyncClean(env, execroot, "Output tree");
        } else {
          execroot.deleteTreesBelow();
        }
      }
    }
    // remove convenience links
    OutputDirectoryLinksUtils.removeOutputDirectoryLinks(
        runtime.getRuleClassProvider().getSymlinkDefinitions(),
        workspaceDirectory,
        env.getWorkspace(),
        env.getReporter(),
        symlinkPrefix,
        env.getRuntime().getProductName());

    // shutdown on expunge cleans
    if (expunge) {
      return BlazeCommandResult.shutdown(ExitCode.SUCCESS);
    }
    System.gc();
    return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
  }

  @Override
  public void editOptions(OptionsParser optionsParser) {}
}
