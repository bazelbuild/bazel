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


import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * <p>Represents an executable command, including its arguments and
 * runtime environment (environment variables, working directory). This class
 * lets a caller execute a command, get its results, and optionally try to kill
 * the task during execution.</p>
 *
 * <p>The use of "shell" in the full name of this class is a misnomer.  In
 * terms of the way its arguments are interpreted, this class is closer to
 * {@code execve(2)} than to {@code system(3)}.  No Bourne shell is executed.
 *
 * <p>The most basic use-case for this class is as follows:
 * <pre>
 *   String[] args = { "/bin/du", "-s", directory };
 *   CommandResult result = new Command(args).execute();
 *   String output = new String(result.getStdout());
 * </pre>
 * which writes the output of the {@code du(1)} command into {@code output}.
 * More complex cases might inspect the stderr stream, kill the subprocess
 * asynchronously, feed input to its standard input, handle the exceptions
 * thrown if the command fails, or print the termination status (exit code or
 * signal name).
 *
 * <h4>Invoking the Bourne shell</h4>
 *
 * <p>Perhaps the most common command invoked programmatically is the UNIX
 * shell, {@code /bin/sh}.  Because the shell is a general-purpose programming
 * language, care must be taken to ensure that variable parts of the shell
 * command (e.g. strings entered by the user) do not contain shell
 * metacharacters, as this poses a correctness and/or security risk.
 *
 * <p>To execute a shell command directly, use the following pattern:
 * <pre>
 *   String[] args = { "/bin/sh", "-c", shellCommand };
 *   CommandResult result = new Command(args).execute();
 * </pre>
 * {@code shellCommand} is a complete Bourne shell program, possibly containing
 * all kinds of unescaped metacharacters.  For example, here's a shell command
 * that enumerates the working directories of all processes named "foo":
 * <pre>ps auxx | grep foo | awk '{print $1}' |
 *      while read pid; do readlink /proc/$pid/cwd; done</pre>
 * It is the responsibility of the caller to ensure that this string means what
 * they intend.
 *
 * <p>Consider the risk posed by allowing the "foo" part of the previous
 * command to be some arbitrary (untrusted) string called {@code processName}:
 * <pre>
 *  // WARNING: unsafe!
 *  String shellCommand = "ps auxx | grep " + processName + " | awk '{print $1}' | "
 *  + "while read pid; do readlink /proc/$pid/cwd; done";</pre>
 * </pre>
 * Passing this string to {@link Command} is unsafe because if the string
 * {@processName} contains shell metacharacters, the meaning of the command can
 * be arbitrarily changed;  consider:
 * <pre>String processName = ". ; rm -fr $HOME & ";</pre>
 *
 * <p>To defend against this possibility, it is essential to properly quote the
 * variable portions of the shell command so that shell metacharacters are
 * escaped.  Use {@link ShellUtils#shellEscape} for this purpose:
 * <pre>
 *  // Safe.
 *  String shellCommand = "ps auxx | grep " + ShellUtils.shellEscape(processName)
 *      + " | awk '{print $1}' | while read pid; do readlink /proc/$pid/cwd; done";
 * </pre>
 *
 * <p>Tip: if you are only invoking a single known command, and no shell
 * features (e.g. $PATH lookup, output redirection, pipelines, etc) are needed,
 * call it directly without using a shell, as in the {@code du(1)} example
 * above.
 *
 * <h4>Other features</h4>
 *
 * <p>A caller can optionally specify bytes to be written to the process's
 * "stdin". The returned {@link CommandResult} object gives the caller access to
 * the exit status, as well as output from "stdout" and "stderr". To use
 * this class with processes that generate very large amounts of input/output,
 * consider
 * {@link #execute(InputStream, KillableObserver, OutputStream, OutputStream)}
 * and
 * {@link #execute(byte[], KillableObserver, OutputStream, OutputStream)}.
 * </p>
 *
 * <p>This class ensures that stdout and stderr streams are read promptly,
 * avoiding potential deadlock if the output is large. See <a
 * href="http://www.javaworld.com/javaworld/jw-12-2000/jw-1229-traps.html"> When
 * <code>Runtime.exec()</code> won't</a>.</p>
 *
 * <p>This class is immutable and therefore thread-safe.</p>
 */
public final class Command {

  private static final Logger log =
    Logger.getLogger("com.google.devtools.build.lib.shell.Command");

  /**
   * Pass this value to {@link #execute(byte[])} to indicate that no input
   * should be written to stdin.
   */
  public static final byte[] NO_INPUT = new byte[0];

  private static final String[] EMPTY_STRING_ARRAY = new String[0];

  /**
   * Pass this to {@link #execute(byte[], KillableObserver, boolean)} to
   * indicate that you do not wish to observe / kill the underlying
   * process.
   */
  public static final KillableObserver NO_OBSERVER = new KillableObserver() {
    @Override
    public void startObserving(final Killable killable) {
      // do nothing
    }
    @Override
    public void stopObserving(final Killable killable) {
      // do nothing
    }
  };

  private final ProcessBuilder processBuilder;

  // Start of public API -----------------------------------------------------

  /**
   * Creates a new {@link Command} that will execute a command line that
   * is described by a {@link ProcessBuilder}. Command line elements,
   * environment, and working directory are taken from this object. The
   * command line is executed exactly as given, without a shell.
   *
   * @param processBuilder {@link ProcessBuilder} describing command line
   *  to execute
   */
  public Command(final ProcessBuilder processBuilder) {
    this(processBuilder.command().toArray(EMPTY_STRING_ARRAY),
         processBuilder.environment(),
         processBuilder.directory());
  }

  /**
   * Creates a new {@link Command} for the given command line elements. The
   * command line is executed exactly as given, without a shell.
   * Subsequent calls to {@link #execute()} will use the JVM's working
   * directory and environment.
   *
   * @param commandLineElements elements of raw command line to execute
   * @throws IllegalArgumentException if commandLine is null or empty
   */
  /* TODO(bazel-team): Use varargs here
   */
  public Command(final String[] commandLineElements) {
    this(commandLineElements, null, null);
  }

  /**
   * Creates a new {@link Command} for the given command line elements. The
   * command line is executed without a shell.
   *
   * The given environment variables and working directory are used in subsequent
   * calls to {@link #execute()}.
   *
   * This command treats the  0-th element of {@code commandLineElement}
   * (the name of an executable to run) specially.
   * <ul>
   *  <li>If it is an absolute path, it is used as it</li>
   *  <li>If it is a single file name, the PATH lookup is performed</li>
   *  <li>If it is a relative path that is not a single file name, the command will attempt to
   *       execute the the binary at that path relative to {@code workingDirectory}.</li>
   * </ul>
   *
   * @param commandLineElements elements of raw command line to execute
   * @param environmentVariables environment variables to replace JVM's
   *  environment variables; may be null
   * @param workingDirectory working directory for execution; if null, current
   * working directory is used
   * @throws IllegalArgumentException if commandLine is null or empty
   */
  public Command(
      String[] commandLineElements,
      final Map<String, String> environmentVariables,
      final File workingDirectory) {
    if (commandLineElements == null || commandLineElements.length == 0) {
      throw new IllegalArgumentException("command line is null or empty");
    }

    File executable = new File(commandLineElements[0]);
    if (!executable.isAbsolute() && executable.getParent() != null) {
      commandLineElements = commandLineElements.clone();
      commandLineElements[0] = new File(workingDirectory, commandLineElements[0]).getAbsolutePath();
    }

    this.processBuilder =
      new ProcessBuilder(commandLineElements);
    if (environmentVariables != null) {
      // TODO(bazel-team) remove next line eventually; it is here to mimic old
      // Runtime.exec() behavior
      this.processBuilder.environment().clear();
      this.processBuilder.environment().putAll(environmentVariables);
    }
    this.processBuilder.directory(workingDirectory);
  }

  /**
   * @return raw command line elements to be executed
   */
  public String[] getCommandLineElements() {
    final List<String> elements = processBuilder.command();
    return elements.toArray(new String[elements.size()]);
  }

  /**
   * @return (unmodifiable) {@link Map} view of command's environment variables
   */
  public Map<String, String> getEnvironmentVariables() {
    return Collections.unmodifiableMap(processBuilder.environment());
  }

  /**
   * @return working directory used for execution, or null if the current
   *         working directory is used
   */
  public File getWorkingDirectory() {
    return processBuilder.directory();
  }

  /**
   * Execute this command with no input to stdin. This call will block until the
   * process completes or an error occurs.
   *
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws AbnormalTerminationException if an {@link IOException} is
   *  encountered while reading from the process, or the process was terminated
   *  due to a signal.
   * @throws BadExitStatusException if the process exits with a
   *  non-zero status
   */
  public CommandResult execute() throws CommandException {
    return execute(NO_INPUT);
  }

  /**
   * Execute this command with given input to stdin. This call will block until
   * the process completes or an error occurs.
   *
   * @param stdinInput bytes to be written to process's stdin
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws AbnormalTerminationException if an {@link IOException} is
   *  encountered while reading from the process, or the process was terminated
   *  due to a signal.
   * @throws BadExitStatusException if the process exits with a
   *  non-zero status
   * @throws NullPointerException if stdin is null
   */
  public CommandResult execute(final byte[] stdinInput)
    throws CommandException {
    nullCheck(stdinInput, "stdinInput");
    return doExecute(new ByteArrayInputSource(stdinInput),
                     NO_OBSERVER,
                     Consumers.createAccumulatingConsumers(),
                     /*killSubprocess=*/false, /*closeOutput=*/false).get();
  }

  /**
   * <p>Execute this command with given input to stdin. This call will block
   * until the process completes or an error occurs. Caller may specify
   * whether the method should ignore stdout/stderr output. If the
   * given number of milliseconds elapses before the command has
   * completed, this method will attempt to kill the command.</p>
   *
   * @param stdinInput bytes to be written to process's stdin, or
   * {@link #NO_INPUT} if no bytes should be written
   * @param timeout number of milliseconds to wait for command completion
   *  before attempting to kill the command
   * @param ignoreOutput if true, method will ignore stdout/stderr output
   *  and return value will not contain this data
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws AbnormalTerminationException if an {@link IOException} is
   *  encountered while reading from the process, or the process was terminated
   *  due to a signal.
   * @throws BadExitStatusException if the process exits with a
   *  non-zero status
   * @throws NullPointerException if stdin is null
   */
  public CommandResult execute(final byte[] stdinInput,
                               final long timeout,
                               final boolean ignoreOutput)
    throws CommandException {
    return execute(stdinInput,
                   new TimeoutKillableObserver(timeout),
                   ignoreOutput);
  }

  /**
   * <p>Execute this command with given input to stdin. This call will block
   * until the process completes or an error occurs. Caller may specify
   * whether the method should ignore stdout/stderr output. The given {@link
   * KillableObserver} may also terminate the process early while running.</p>
   *
   * @param stdinInput bytes to be written to process's stdin, or
   *  {@link #NO_INPUT} if no bytes should be written
   * @param observer {@link KillableObserver} that should observe the running
   *  process, or {@link #NO_OBSERVER} if caller does not wish to kill
   *  the process
   * @param ignoreOutput if true, method will ignore stdout/stderr output
   *  and return value will not contain this data
   * @return {@link CommandResult} representing result of the execution
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws AbnormalTerminationException if the process is interrupted (or
   *  killed) before completion, if an {@link IOException} is encountered while
   *  reading from the process, or the process was terminated due to a signal.
   * @throws BadExitStatusException if the process exits with a
   *  non-zero status
   * @throws NullPointerException if stdin is null
   */
  public CommandResult execute(final byte[] stdinInput,
                               final KillableObserver observer,
                               final boolean ignoreOutput)
    throws CommandException {
    // supporting "null" here for backwards compatibility
    final KillableObserver theObserver =
      observer == null ? NO_OBSERVER : observer;
    return doExecute(new ByteArrayInputSource(stdinInput),
                     theObserver,
                     ignoreOutput ? Consumers.createDiscardingConsumers()
                                  : Consumers.createAccumulatingConsumers(),
                     /*killSubprocess=*/false, /*closeOutput=*/false).get();
  }

  /**
   * <p>Execute this command with given input to stdin. This call blocks
   * until the process completes or an error occurs. The caller provides
   * {@link OutputStream} instances into which the process writes its
   * stdout/stderr output; these streams are <em>not</em> closed when the
   * process terminates. The given {@link KillableObserver} may also
   * terminate the process early while running.</p>
   *
   * <p>Note that stdout and stderr are written concurrently. If these are
   * aliased to each other, it is the caller's duty to ensure thread safety.
   * </p>
   *
   * @param stdinInput bytes to be written to process's stdin, or
   * {@link #NO_INPUT} if no bytes should be written
   * @param observer {@link KillableObserver} that should observe the running
   *  process, or {@link #NO_OBSERVER} if caller does not wish to kill the
   *  process
   * @param stdOut the process will write its standard output into this stream.
   *  E.g., you could pass {@link System#out} as <code>stdOut</code>.
   * @param stdErr the process will write its standard error into this stream.
   *  E.g., you could pass {@link System#err} as <code>stdErr</code>.
   * @return {@link CommandResult} representing result of the execution. Note
   *  that {@link CommandResult#getStdout()} and
   *  {@link CommandResult#getStderr()} will yield {@link IllegalStateException}
   *  in this case, as the output is written to <code>stdOut/stdErr</code>
   *  instead.
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws AbnormalTerminationException if the process is interrupted (or
   *  killed) before completion, if an {@link IOException} is encountered while
   *  reading from the process, or the process was terminated due to a signal.
   * @throws BadExitStatusException if the process exits with a
   *  non-zero status
   * @throws NullPointerException if any argument is null.
   */
  public CommandResult execute(final byte[] stdinInput,
                               final KillableObserver observer,
                               final OutputStream stdOut,
                               final OutputStream stdErr)
    throws CommandException {
    return execute(stdinInput, observer, stdOut, stdErr, false);
  }

  /**
   * Like {@link #execute(byte[], KillableObserver, OutputStream, OutputStream)}
   * but enables setting of the killSubprocessOnInterrupt attribute.
   *
   * @param killSubprocessOnInterrupt if set to true, the execution of
   * this command is <i>interruptible</i>: in other words, if this thread is
   * interrupted during a call to execute, the subprocess will be terminated
   * and the call will return in a timely manner.  If false, the subprocess
   * will run to completion; this is the default value use by all other
   * constructors.  The thread's interrupted status is preserved in all cases,
   * however.
   */
  public CommandResult execute(final byte[] stdinInput,
                               final KillableObserver observer,
                               final OutputStream stdOut,
                               final OutputStream stdErr,
                               final boolean killSubprocessOnInterrupt)
    throws CommandException {
    nullCheck(stdinInput, "stdinInput");
    nullCheck(observer, "observer");
    nullCheck(stdOut, "stdOut");
    nullCheck(stdErr, "stdErr");
    return doExecute(new ByteArrayInputSource(stdinInput),
                     observer,
                     Consumers.createStreamingConsumers(stdOut, stdErr),
                     killSubprocessOnInterrupt, false).get();
  }

  /**
   * <p>Execute this command with given input to stdin; this stream is closed
   * when the process terminates, and exceptions raised when closing this
   * stream are ignored. This call blocks
   * until the process completes or an error occurs. The caller provides
   * {@link OutputStream} instances into which the process writes its
   * stdout/stderr output; these streams are <em>not</em> closed when the
   * process terminates. The given {@link KillableObserver} may also
   * terminate the process early while running.</p>
   *
   * @param stdinInput The input to this process's stdin
   * @param observer {@link KillableObserver} that should observe the running
   *  process, or {@link #NO_OBSERVER} if caller does not wish to kill the
   *  process
   * @param stdOut the process will write its standard output into this stream.
   *  E.g., you could pass {@link System#out} as <code>stdOut</code>.
   * @param stdErr the process will write its standard error into this stream.
   *  E.g., you could pass {@link System#err} as <code>stdErr</code>.
   * @return {@link CommandResult} representing result of the execution. Note
   *  that {@link CommandResult#getStdout()} and
   *  {@link CommandResult#getStderr()} will yield {@link IllegalStateException}
   *  in this case, as the output is written to <code>stdOut/stdErr</code>
   *  instead.
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws AbnormalTerminationException if the process is interrupted (or
   *  killed) before completion, if an {@link IOException} is encountered while
   *  reading from the process, or the process was terminated due to a signal.
   * @throws BadExitStatusException if the process exits with a
   *  non-zero status
   * @throws NullPointerException if any argument is null.
   */
  public CommandResult execute(final InputStream stdinInput,
                               final KillableObserver observer,
                               final OutputStream stdOut,
                               final OutputStream stdErr)
    throws CommandException {
    nullCheck(stdinInput, "stdinInput");
    nullCheck(observer, "observer");
    nullCheck(stdOut, "stdOut");
    nullCheck(stdErr, "stdErr");
    return doExecute(new InputStreamInputSource(stdinInput),
                     observer,
                     Consumers.createStreamingConsumers(stdOut, stdErr),
                     /*killSubprocess=*/false, /*closeOutput=*/false).get();
  }

  /**
   * <p>Execute this command with given input to stdin; this stream is closed
   * when the process terminates, and exceptions raised when closing this
   * stream are ignored. This call blocks
   * until the process completes or an error occurs. The caller provides
   * {@link OutputStream} instances into which the process writes its
   * stdout/stderr output; these streams are closed when the process terminates
   * if closeOut is set. The given {@link KillableObserver} may also
   * terminate the process early while running.</p>
   *
   * @param stdinInput The input to this process's stdin
   * @param observer {@link KillableObserver} that should observe the running
   *  process, or {@link #NO_OBSERVER} if caller does not wish to kill the
   *  process
   * @param stdOut the process will write its standard output into this stream.
   *  E.g., you could pass {@link System#out} as <code>stdOut</code>.
   * @param stdErr the process will write its standard error into this stream.
   *  E.g., you could pass {@link System#err} as <code>stdErr</code>.
   * @param closeOut whether to close the output streams when the subprocess
   *  terminates.
   * @return {@link CommandResult} representing result of the execution. Note
   *  that {@link CommandResult#getStdout()} and
   *  {@link CommandResult#getStderr()} will yield {@link IllegalStateException}
   *  in this case, as the output is written to <code>stdOut/stdErr</code>
   *  instead.
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws AbnormalTerminationException if the process is interrupted (or
   *  killed) before completion, if an {@link IOException} is encountered while
   *  reading from the process, or the process was terminated due to a signal.
   * @throws BadExitStatusException if the process exits with a
   *  non-zero status
   * @throws NullPointerException if any argument is null.
   */
  public CommandResult execute(final InputStream stdinInput,
      final KillableObserver observer,
      final OutputStream stdOut,
      final OutputStream stdErr,
      boolean closeOut)
      throws CommandException {
    nullCheck(stdinInput, "stdinInput");
    nullCheck(observer, "observer");
    nullCheck(stdOut, "stdOut");
    nullCheck(stdErr, "stdErr");
    return doExecute(new InputStreamInputSource(stdinInput),
        observer,
        Consumers.createStreamingConsumers(stdOut, stdErr),
        false, closeOut).get();
  }

  /**
   * <p>Executes this command with the given stdinInput, but does not
   * wait for it to complete. The caller may choose to observe the status
   * of the launched process by calling methods on the returned object.
   *
   * @param stdinInput bytes to be written to process's stdin, or
   * {@link #NO_INPUT} if no bytes should be written
   * @return An object that can be used to check if the process terminated and
   *  obtain the process results.
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws NullPointerException if stdin is null
   */
  public FutureCommandResult executeAsynchronously(final byte[] stdinInput)
      throws CommandException {
    return executeAsynchronously(stdinInput, NO_OBSERVER);
  }

  /**
   * <p>Executes this command with the given input to stdin, but does
   * not wait for it to complete. The caller may choose to observe the
   * status of the launched process by calling methods on the returned
   * object.  This method performs the minimum cleanup after the
   * process terminates: It closes the input stream, and it ignores
   * exceptions that result from closing it. The given {@link
   * KillableObserver} may also terminate the process early while
   * running.</p>
   *
   * <p>Note that in this case the {@link KillableObserver} will be assigned
   * to start observing the process via
   * {@link KillableObserver#startObserving(Killable)} but will only be
   * unassigned via {@link KillableObserver#stopObserving(Killable)}, if
   * {@link FutureCommandResult#get()} is called. If the
   * {@link KillableObserver} implementation used with this method will
   * not work correctly without calls to
   * {@link KillableObserver#stopObserving(Killable)} then a new instance
   * should be used for each call to this method.</p>
   *
   * @param stdinInput bytes to be written to process's stdin, or
   * {@link #NO_INPUT} if no bytes should be written
   * @param observer {@link KillableObserver} that should observe the running
   *  process, or {@link #NO_OBSERVER} if caller does not wish to kill
   *  the process
   * @return An object that can be used to check if the process terminated and
   *  obtain the process results.
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws NullPointerException if stdin is null
   */
  public FutureCommandResult executeAsynchronously(final byte[] stdinInput,
                                    final KillableObserver observer)
    throws CommandException {
    // supporting "null" here for backwards compatibility
    final KillableObserver theObserver =
      observer == null ? NO_OBSERVER : observer;
    nullCheck(stdinInput, "stdinInput");
    return doExecute(new ByteArrayInputSource(stdinInput),
        theObserver,
        Consumers.createDiscardingConsumers(),
        /*killSubprocess=*/false, /*closeOutput=*/false);
  }

  /**
   * <p>Executes this command with the given input to stdin, but does
   * not wait for it to complete. The caller may choose to observe the
   * status of the launched process by calling methods on the returned
   * object.  This method performs the minimum cleanup after the
   * process terminates: It closes the input stream, and it ignores
   * exceptions that result from closing it. The caller provides
   * {@link OutputStream} instances into which the process writes its
   * stdout/stderr output; these streams are <em>not</em> closed when
   * the process terminates. The given {@link KillableObserver} may
   * also terminate the process early while running.</p>
   *
   * <p>Note that stdout and stderr are written concurrently. If these are
   * aliased to each other, or if the caller continues to write to these
   * streams, it is the caller's duty to ensure thread safety.
   * </p>
   *
   * <p>Note that in this case the {@link KillableObserver} will be assigned
   * to start observing the process via
   * {@link KillableObserver#startObserving(Killable)} but will only be
   * unassigned via {@link KillableObserver#stopObserving(Killable)}, if
   * {@link FutureCommandResult#get()} is called. If the
   * {@link KillableObserver} implementation used with this method will
   * not work correctly without calls to
   * {@link KillableObserver#stopObserving(Killable)} then a new instance
   * should be used for each call to this method.</p>
   *
   * @param stdinInput The input to this process's stdin
   * @param observer {@link KillableObserver} that should observe the running
   *  process, or {@link #NO_OBSERVER} if caller does not wish to kill
   *  the process
   * @param stdOut the process will write its standard output into this stream.
   *  E.g., you could pass {@link System#out} as <code>stdOut</code>.
   * @param stdErr the process will write its standard error into this stream.
   *  E.g., you could pass {@link System#err} as <code>stdErr</code>.
   * @param closeOutput whether to close stdout / stderr when the process closes its output streams.
   * @return An object that can be used to check if the process terminated and
   *  obtain the process results.
   * @throws ExecFailedException if {@link Runtime#exec(String[])} fails for any
   *  reason
   * @throws NullPointerException if stdin is null
   */
  public FutureCommandResult executeAsynchronously(final InputStream stdinInput,
                                    final KillableObserver observer,
                                    final OutputStream stdOut,
                                    final OutputStream stdErr,
                                    final boolean closeOutput)
      throws CommandException {
    // supporting "null" here for backwards compatibility
    final KillableObserver theObserver =
        observer == null ? NO_OBSERVER : observer;
    nullCheck(stdinInput, "stdinInput");
    return doExecute(new InputStreamInputSource(stdinInput),
        theObserver,
        Consumers.createStreamingConsumers(stdOut, stdErr),
        /*killSubprocess=*/false, closeOutput);
  }
  public FutureCommandResult executeAsynchronously(final InputStream stdinInput,
      final KillableObserver observer,
      final OutputStream stdOut,
      final OutputStream stdErr)
      throws CommandException {
    return executeAsynchronously(stdinInput, observer, stdOut, stdErr, /*closeOutput=*/false);
  }

  // End of public API -------------------------------------------------------

  private void nullCheck(Object argument, String argumentName) {
    if (argument == null) {
      String message = argumentName + " argument must not be null.";
      throw new NullPointerException(message);
    }
  }

  private FutureCommandResult doExecute(final InputSource stdinInput,
      final KillableObserver observer,
      final Consumers.OutErrConsumers outErrConsumers,
      final boolean killSubprocessOnInterrupt,
      final boolean closeOutputStreams)
    throws CommandException {

    logCommand();

    final Process process = startProcess();

    outErrConsumers.logConsumptionStrategy();

    outErrConsumers.registerInputs(process.getInputStream(),
                                   process.getErrorStream(),
                                   closeOutputStreams);

    processInput(stdinInput, process);

    // TODO(bazel-team): if the input stream is unbounded, observers will not get start
    // notification in a timely manner!
    final Killable processKillable = observeProcess(process, observer);

    return new FutureCommandResult() {
      @Override
      public CommandResult get() throws AbnormalTerminationException {
        return waitForProcessToComplete(process,
            observer,
            processKillable,
            outErrConsumers,
            killSubprocessOnInterrupt);
      }

      @Override
      public boolean isDone() {
        try {
          // exitValue seems to be the only non-blocking call for
          // checking process liveness.
          process.exitValue();
          return true;
        } catch (IllegalThreadStateException e) {
          return false;
        }
      }
    };
  }

  private Process startProcess()
    throws ExecFailedException {
    try {
      return processBuilder.start();
    } catch (IOException ioe) {
      throw new ExecFailedException(this, ioe);
    }
  }

  private static interface InputSource {
    void copyTo(OutputStream out) throws IOException;
    boolean isEmpty();
    String toLogString(String sourceName);
  }

  private static class ByteArrayInputSource implements InputSource {
    private byte[] bytes;
    ByteArrayInputSource(byte[] bytes){
      this.bytes = bytes;
    }
    @Override
    public void copyTo(OutputStream out) throws IOException {
      out.write(bytes);
      out.flush();
    }
    @Override
    public boolean isEmpty() {
      return bytes.length == 0;
    }
    @Override
    public String toLogString(String sourceName) {
      if (isEmpty()) {
        return "No input to " + sourceName;
      } else {
        return "Input to " + sourceName + ": " +
            LogUtil.toTruncatedString(bytes);
      }
    }
  }

  private static class InputStreamInputSource implements InputSource {
    private InputStream inputStream;
    InputStreamInputSource(InputStream inputStream){
      this.inputStream = inputStream;
    }
    @Override
    public void copyTo(OutputStream out) throws IOException {
      byte[] buf = new byte[4096];
      int r;
      while ((r = inputStream.read(buf)) != -1) {
        out.write(buf, 0, r);
        out.flush();
      }
    }
    @Override
    public boolean isEmpty() {
      return false;
    }
    @Override
    public String toLogString(String sourceName) {
      return "Input to " + sourceName + " is a stream.";
    }
  }

  private static void processInput(final InputSource stdinInput,
                                   final Process process) {
    if (log.isLoggable(Level.FINER)) {
      log.finer(stdinInput.toLogString("stdin"));
    }
    try {
      if (stdinInput.isEmpty()) {
        return;
      }
      stdinInput.copyTo(process.getOutputStream());
    } catch (IOException ioe) {
      // Note: this is not an error!  Perhaps the command just isn't hungry for
      // our input and exited with success.  Process.waitFor (later) will tell
      // us.
      //
      // (Unlike out/err streams, which are read asynchronously, the input stream is written
      // synchronously, in its entirety, before processInput returns.  If the input is
      // infinite, and is passed through e.g. "cat" subprocess and back into the
      // ByteArrayOutputStream, that will eventually run out of memory, causing the output stream
      // to be closed, "cat" to terminate with SIGPIPE, and processInput to receive an IOException.
    } finally {
      // if this statement is ever deleted, the process's outputStream
      // must be closed elsewhere -- it is not closed automatically
      Command.silentClose(process.getOutputStream());
    }
  }

  private static Killable observeProcess(final Process process,
                                         final KillableObserver observer) {
    final Killable processKillable = new ProcessKillable(process);
    observer.startObserving(processKillable);
    return processKillable;
  }

  private CommandResult waitForProcessToComplete(
    final Process process,
    final KillableObserver observer,
    final Killable processKillable,
    final Consumers.OutErrConsumers outErr,
    final boolean killSubprocessOnInterrupt)
    throws AbnormalTerminationException {

    log.finer("Waiting for process...");

    TerminationStatus status =
        waitForProcess(process, killSubprocessOnInterrupt);

    observer.stopObserving(processKillable);

    log.finer(status.toString());

    try {
      outErr.waitForCompletion();
    } catch (IOException ioe) {
      CommandResult noOutputResult =
        new CommandResult(CommandResult.EMPTY_OUTPUT,
                          CommandResult.EMPTY_OUTPUT,
                          status);
      if (status.success()) {
        // If command was otherwise successful, throw an exception about this
        throw new AbnormalTerminationException(this, noOutputResult, ioe);
      } else {
        // Otherwise, throw the more important exception -- command
        // was not successful
        String message = status
          + "; also encountered an error while attempting to retrieve output";
        throw status.exited()
          ? new BadExitStatusException(this, noOutputResult, message, ioe)
          : new AbnormalTerminationException(this,
              noOutputResult, message, ioe);
      }
    }

    CommandResult result = new CommandResult(outErr.getAccumulatedOut(),
                                             outErr.getAccumulatedErr(),
                                             status);
    result.logThis();
    if (status.success()) {
      return result;
    } else if (status.exited()) {
      throw new BadExitStatusException(this, result, status.toString());
    } else {
      throw new AbnormalTerminationException(this, result, status.toString());
    }
  }

  private static TerminationStatus waitForProcess(Process process,
                                       boolean killSubprocessOnInterrupt) {
    boolean wasInterrupted = false;
    try {
      while (true) {
        try {
          return new TerminationStatus(process.waitFor());
        } catch (InterruptedException ie) {
          wasInterrupted = true;
          if (killSubprocessOnInterrupt) {
            process.destroy();
          }
        }
      }
    } finally {
      // Read this for detailed explanation: http://www.ibm.com/developerworks/library/j-jtp05236/
      if (wasInterrupted) {
        Thread.currentThread().interrupt(); // preserve interrupted status
      }
    }
  }

  private void logCommand() {
    if (!log.isLoggable(Level.FINE)) {
      return;
    }
    log.fine(toDebugString());
  }

  /**
   * A string representation of this command object which includes
   * the arguments, the environment, and the working directory. Avoid
   * relying on the specifics of this format. Note that the size
   * of the result string will reflect the size of the command.
   */
  public String toDebugString() {
    StringBuilder message = new StringBuilder(128);
    message.append("Executing (without brackets):");
    for (final String arg : processBuilder.command()) {
      message.append(" [");
      message.append(arg);
      message.append(']');
    }
    message.append("; environment: ");
    message.append(processBuilder.environment());
    final File workingDirectory = processBuilder.directory();
    message.append("; working dir: ");
    message.append(workingDirectory == null ?
                   "(current)" :
                   workingDirectory.toString());
    return message.toString();
  }

  /**
   * Close the <code>out</code> stream and log a warning if anything happens.
   */
  private static void silentClose(final OutputStream out) {
    try {
      out.close();
    } catch (IOException ioe) {
      String message = "Unexpected exception while closing output stream";
      log.log(Level.WARNING, message, ioe);
    }
  }
}
