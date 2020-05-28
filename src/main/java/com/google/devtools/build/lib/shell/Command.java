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

package com.google.devtools.build.lib.shell;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.flogger.LazyArgs;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.shell.Consumers.OutErrConsumers;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * An executable command, including its arguments and runtime environment (environment variables,
 * working directory). It lets a caller execute a command, get its results, and optionally forward
 * interrupts to the subprocess. This class creates threads to ensure timely reading of subprocess
 * outputs.
 *
 * <p>This class is immutable and thread-safe.
 *
 * <p>The use of "shell" in the package name of this class is a misnomer.  In terms of the way its
 * arguments are interpreted, this class is closer to {@code execve(2)} than to {@code system(3)}.
 * No shell is executed.
 *
 * <h4>Examples</h4>
 *
 * <p>The most basic use-case for this class is as follows:
 * <pre>
 *   String[] args = { "/bin/du", "-s", directory };
 *   BlazeCommandResult result = new Command(args).execute();
 *   String output = new String(result.getStdout());
 * </pre>
 * which writes the output of the {@code du(1)} command into {@code output}. More complex cases
 * might inspect the stderr stream, kill the subprocess asynchronously, feed input to its standard
 * input, handle the exceptions thrown if the command fails, or print the termination status (exit
 * code or signal name).
 *
 * <h4>Other Features</h4>
 *
 * <p>A caller can optionally specify bytes to be written to the process's "stdin". The returned
 * {@link CommandResult} object gives the caller access to the exit status, as well as output from
 * "stdout" and "stderr". To use this class with processes that generate very large amounts of
 * input/output, consider {@link #execute(OutputStream, OutputStream)},
 * {@link #executeAsync(OutputStream, OutputStream)}, or
 * {@link #executeAsync(InputStream, OutputStream, OutputStream, boolean)}.
 *
 * <p>This class ensures that stdout and stderr streams are read promptly, avoiding potential
 * deadlock if the output is large. See
 * <a href="http://www.javaworld.com/javaworld/jw-12-2000/jw-1229-traps.html"> when
 * <code>Runtime.exec()</code> won't</a>.
 *
 * <h4>Caution: Invoking Shell Commands</h4>
 *
 * <p>Perhaps the most common command invoked programmatically is the UNIX shell, {@code /bin/sh}.
 * Because the shell is a general-purpose programming language, care must be taken to ensure that
 * variable parts of the shell command (e.g. strings entered by the user) do not contain shell
 * metacharacters, as this poses a correctness and/or security risk.
 *
 * <p>To execute a shell command directly, use the following pattern:
 * <pre>
 *   String[] args = { "/bin/sh", "-c", shellCommand };
 *   BlazeCommandResult result = new Command(args).execute();
 * </pre>
 * {@code shellCommand} is a complete Bourne shell program, possibly containing all kinds of
 * unescaped metacharacters.  For example, here's a shell command that enumerates the working
 * directories of all processes named "foo":
 * <pre>ps auxx | grep foo | awk '{print $1}' |
 *      while read pid; do readlink /proc/$pid/cwd; done</pre>
 * It is the responsibility of the caller to ensure that this string means what they intend.
 *
 * <p>Consider the risk posed by allowing the "foo" part of the previous command to be some
 * arbitrary (untrusted) string called {@code processName}:
 * <pre>
 *  // WARNING: unsafe!
 *  String shellCommand = "ps auxx | grep " + processName + " | awk '{print $1}' | "
 *  + "while read pid; do readlink /proc/$pid/cwd; done";</pre>
 * </pre>
 * Passing this string to {@link Command} is unsafe because if the string {@processName} contains
 * shell metacharacters, the meaning of the command can be arbitrarily changed; consider:
 * <pre>String processName = ". ; rm -fr $HOME & ";</pre>
 *
 * <p>To defend against this possibility, it is essential to properly quote the variable portions of
 * the shell command so that shell metacharacters are escaped.  Use {@link ShellUtils#shellEscape}
 * for this purpose:
 * <pre>
 *  // Safe.
 *  String shellCommand = "ps auxx | grep " + ShellUtils.shellEscape(processName)
 *      + " | awk '{print $1}' | while read pid; do readlink /proc/$pid/cwd; done";
 * </pre>
 *
 * <p>Tip: if you are only invoking a single known command, and no shell features (e.g. $PATH
 * lookup, output redirection, pipelines, etc) are needed, call it directly without using a shell,
 * as in the {@code du(1)} example above.
 */
public final class Command {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Pass this value to {@link #execute} to indicate that no input should be written to stdin. */
  public static final InputStream NO_INPUT = new NullInputStream();

  public static final boolean KILL_SUBPROCESS_ON_INTERRUPT = true;
  public static final boolean CONTINUE_SUBPROCESS_ON_INTERRUPT = false;

  private final SubprocessBuilder subprocessBuilder;

  /**
   * Creates a new {@link Command} for the given command line. The environment is inherited from the
   * current process, as is the working directory. No timeout is enforced. The command line is
   * executed exactly as given, without a shell. Subsequent calls to {@link #execute()} will use the
   * JVM's working directory and environment.
   *
   * @param commandLineElements elements of raw command line to execute
   * @throws IllegalArgumentException if commandLine is null or empty
   */
  public Command(String[] commandLineElements) {
    this(commandLineElements, null, null, Duration.ZERO);
  }

  /**
   * Just like {@link #Command(String[], Map, File, Duration)}, but without a timeout.
   */
  public Command(
      String[] commandLineElements,
      @Nullable Map<String, String> environmentVariables,
      @Nullable File workingDirectory) {
    this(commandLineElements, environmentVariables, workingDirectory, Duration.ZERO);
  }

  /**
   * Creates a new {@link Command} for the given command line elements. The command line is executed
   * without a shell.
   *
   * <p>The given environment variables and working directory are used in subsequent calls to {@link
   * #execute()}.
   *
   * <p>This command treats the 0-th element of {@code commandLineElement} (the name of an
   * executable to run) specially.
   *
   * <ul>
   *   <li>If it is an absolute path, it is used as it
   *   <li>If it is a single file name, the PATH lookup is performed
   *   <li>If it is a relative path that is not a single file name, the command will attempt to
   *       execute the binary at that path relative to {@code workingDirectory}.
   * </ul>
   *
   * @param commandLineElements elements of raw command line to execute
   * @param environmentVariables environment variables to replace JVM's environment variables; may
   *     be null
   * @param workingDirectory working directory for execution; if null, the VM's current working
   *     directory is used
   * @param timeout timeout; a value less than or equal to 0 is treated as no timeout
   * @throws IllegalArgumentException if commandLine is null or empty
   */
  // TODO(ulfjack): Throw a special exception if there was a timeout.
  public Command(
      String[] commandLineElements,
      @Nullable Map<String, String> environmentVariables,
      @Nullable File workingDirectory,
      Duration timeout) {
    Preconditions.checkNotNull(commandLineElements);
    Preconditions.checkArgument(
        commandLineElements.length != 0, "cannot run an empty command line");

    File executable = new File(commandLineElements[0]);
    if (!executable.isAbsolute() && executable.getParent() != null) {
      commandLineElements = commandLineElements.clone();
      commandLineElements[0] = new File(workingDirectory, commandLineElements[0]).getAbsolutePath();
    }

    this.subprocessBuilder = new SubprocessBuilder();
    subprocessBuilder.setArgv(ImmutableList.copyOf(commandLineElements));
    subprocessBuilder.setEnv(environmentVariables);
    subprocessBuilder.setWorkingDirectory(workingDirectory);
    subprocessBuilder.setTimeoutMillis(timeout.toMillis());
  }

  /** Returns the raw command line elements to be executed */
  public String[] getCommandLineElements() {
    final List<String> elements = subprocessBuilder.getArgv();
    return elements.toArray(new String[elements.size()]);
  }

  /** Returns an (unmodifiable) {@link Map} view of command's environment variables or null. */
  @Nullable public Map<String, String> getEnvironmentVariables() {
    return subprocessBuilder.getEnv();
  }

  /** Returns the working directory to be used for execution, or null. */
  @Nullable public File getWorkingDirectory() {
    return subprocessBuilder.getWorkingDirectory();
  }

  /**
   * Execute this command with no input to stdin, and with the output captured in memory. If the
   * current process is interrupted, then the subprocess is also interrupted. This call blocks until
   * the subprocess completes or an error occurs.
   *
   * <p>This method is a convenience wrapper for <code>executeAsync().get()</code>.
   *
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any reason
   * @throws AbnormalTerminationException if an {@link IOException} is encountered while reading
   *    from the process, or the process was terminated due to a signal
   * @throws BadExitStatusException if the process exits with a non-zero status
   */
  public CommandResult execute() throws CommandException {
    return executeAsync().get();
  }

  /**
   * Execute this command with no input to stdin, and with the output streamed to the given output
   * streams, which must be thread-safe. If the current process is interrupted, then the subprocess
   * is also interrupted. This call blocks until the subprocess completes or an error occurs.
   *
   * <p>Note that the given output streams are never closed by this class.
   *
   * <p>This method is a convenience wrapper for <code>executeAsync(stdOut, stdErr).get()</code>.
   *
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any reason
   * @throws AbnormalTerminationException if an {@link IOException} is encountered while reading
   *    from the process, or the process was terminated due to a signal
   * @throws BadExitStatusException if the process exits with a non-zero status
   */
  public CommandResult execute(OutputStream stdOut, OutputStream stdErr) throws CommandException {
    return doExecute(
        NO_INPUT, Consumers.createStreamingConsumers(stdOut, stdErr), KILL_SUBPROCESS_ON_INTERRUPT)
            .get();
  }

  /**
   * Execute this command with no input to stdin, and with the output captured in memory. If the
   * current process is interrupted, then the subprocess is also interrupted. This call blocks until
   * the subprocess is started or throws an error if that fails, but does not wait for the
   * subprocess to exit.
   *
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any reason
   * @throws AbnormalTerminationException if an {@link IOException} is encountered while reading
   *    from the process, or the process was terminated due to a signal
   * @throws BadExitStatusException if the process exits with a non-zero status
   */
  public FutureCommandResult executeAsync() throws CommandException {
    return doExecute(
        NO_INPUT, Consumers.createAccumulatingConsumers(), KILL_SUBPROCESS_ON_INTERRUPT);
  }

  /**
   * Execute this command with no input to stdin, and with the output streamed to the given output
   * streams, which must be thread-safe. If the current process is interrupted, then the subprocess
   * is also interrupted. This call blocks until the subprocess is started or throws an error if
   * that fails, but does not wait for the subprocess to exit.
   *
   * <p>Note that the given output streams are never closed by this class.
   *
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any reason
   * @throws AbnormalTerminationException if an {@link IOException} is encountered while reading
   *    from the process, or the process was terminated due to a signal
   * @throws BadExitStatusException if the process exits with a non-zero status
   */
  public FutureCommandResult executeAsync(OutputStream stdOut, OutputStream stdErr)
      throws CommandException {
    return doExecute(
        NO_INPUT, Consumers.createStreamingConsumers(stdOut, stdErr), KILL_SUBPROCESS_ON_INTERRUPT);
  }

  /**
   * Execute this command with no input to stdin, and with the output captured in memory. This call
   * blocks until the subprocess is started or throws an error if that fails, but does not wait for
   * the subprocess to exit.
   *
   * @param killSubprocessOnInterrupt whether the subprocess should be killed if the current process
   *     is interrupted
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any reason
   * @throws AbnormalTerminationException if an {@link IOException} is encountered while reading
   *    from the process, or the process was terminated due to a signal
   * @throws BadExitStatusException if the process exits with a non-zero status
   */
  public FutureCommandResult executeAsync(
      InputStream stdinInput, boolean killSubprocessOnInterrupt) throws CommandException {
    return doExecute(
        stdinInput, Consumers.createAccumulatingConsumers(), killSubprocessOnInterrupt);
  }

  /**
   * Execute this command with no input to stdin, and with the output streamed to the given output
   * streams, which must be thread-safe. This call blocks until the subprocess is started or throws
   * an error if that fails, but does not wait for the subprocess to exit.
   *
   * <p>Note that the given output streams are never closed by this class.
   *
   * @param killSubprocessOnInterrupt whether the subprocess should be killed if the current process
   *     is interrupted
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any reason
   * @throws AbnormalTerminationException if an {@link IOException} is encountered while reading
   *    from the process, or the process was terminated due to a signal
   * @throws BadExitStatusException if the process exits with a non-zero status
   */
  public FutureCommandResult executeAsync(
      InputStream stdinInput,
      OutputStream stdOut,
      OutputStream stdErr,
      boolean killSubprocessOnInterrupt)
          throws CommandException {
    return doExecute(
        stdinInput, Consumers.createStreamingConsumers(stdOut, stdErr), killSubprocessOnInterrupt);
  }

  /**
   * A string representation of this command object which includes the arguments, the environment,
   * and the working directory. Avoid relying on the specifics of this format. Note that the size of
   * the result string will reflect the size of the command.
   */
  public String toDebugString() {
    StringBuilder message = new StringBuilder(128);
    message.append("Executing (without brackets):");
    for (String arg : subprocessBuilder.getArgv()) {
      message.append(" [");
      message.append(arg);
      message.append(']');
    }
    message.append("; environment: ");
    message.append(subprocessBuilder.getEnv());
    message.append("; working dir: ");
    File workingDirectory = subprocessBuilder.getWorkingDirectory();
    message.append(workingDirectory == null ?
                   "(current)" :
                   workingDirectory.toString());
    return message.toString();
  }

  private FutureCommandResult doExecute(
      InputStream stdinInput, OutErrConsumers outErrConsumers, boolean killSubprocessOnInterrupt) 
          throws ExecFailedException {
    Preconditions.checkNotNull(stdinInput, "stdinInput");
    logCommand();

    Subprocess process = startProcess();

    outErrConsumers.logConsumptionStrategy();
    outErrConsumers.registerInputs(
        process.getInputStream(), process.getErrorStream(), /* closeStreams= */ false);

    // TODO(ulfjack): This call blocks until all input is written. If stdinInput is large (or
    // unbounded), then the async calls can block for a long time, and the timeout is not properly
    // enforced.
    processInput(stdinInput, process);

    return new FutureCommandResultImpl(this, process, outErrConsumers, killSubprocessOnInterrupt);
  }

  private Subprocess startProcess() throws ExecFailedException {
    try {
      return subprocessBuilder.start();
    } catch (IOException ioe) {
      throw new ExecFailedException(this, ioe);
    }
  }

  private static class NullInputStream extends InputStream {
    @Override
    public int read() {
      return -1;
    }

    @Override
    public int available() {
      return 0;
    }
  }

  private static void processInput(InputStream stdinInput, Subprocess process) {
    logger.atFiner().log(stdinInput.toString());
    try (OutputStream out = process.getOutputStream()) {
      ByteStreams.copy(stdinInput, out);
    } catch (IOException ioe) {
      // Note: this is not an error!  Perhaps the command just isn't hungry for our input and exited
      // with success. Process.waitFor (later) will tell us.
      //
      // (Unlike out/err streams, which are read asynchronously, the input stream is written
      // synchronously, in its entirety, before processInput returns.  If the input is infinite, and
      // is passed through e.g. "cat" subprocess and back into the ByteArrayOutputStream, that will
      // eventually run out of memory, causing the output stream to be closed, "cat" to terminate
      // with SIGPIPE, and processInput to receive an IOException.
    }
  }

  private void logCommand() {
    logger.atFine().log("%s", LazyArgs.lazy(this::toDebugString));
  }
}
