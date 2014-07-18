// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.devtools.build.lib.vfs.FileSystemUtils.createDirectoryAndParents;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.Artifact.MiddlemanExpander;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * An abstract action executor class that takes care on action execution stuff, such as:
 * action prepare, action execution, validation that all output artifacts were created,
 * error reporting etc.
 */
public abstract class AbstractActionExecutor {
  protected final Reporter reporter;
  private final Profiler profiler = Profiler.instance();
  protected Executor executorEngine;
  private ActionLogBufferPathGenerator actionLogBufferPathGenerator;

  public AbstractActionExecutor(Reporter reporter) {
    this.reporter = reporter;
  }

  public void setExecutorEngine(Executor executorEngine) {
    this.executorEngine = executorEngine;
  }

  public void setActionLogBufferPathGenerator(
      ActionLogBufferPathGenerator actionLogBufferPathGenerator) {
    this.actionLogBufferPathGenerator = actionLogBufferPathGenerator;
  }

  /**
   * Prepare, schedule, execute, and then complete the action.
   * When this function is called, we know that this action needs to be executed.
   * This function will prepare for the action's execution (i.e. delete the outputs);
   * schedule its execution; execute the action;
   * and then do some post-execution processing to complete the action:
   * set the outputs readonly and executable, and insert the action results in the
   * action cache.
   *
   * @param action  The action to execute
   * @param token  The token returned by dependencyChecker.needToExecute()
   * @param actionInputFileCache source of file metadata.
   * @param metadataHandler source of file data for the action cache and output-checking.
   * @param middlemanExpander The object that can expand middleman inputs of the action.
   * @param actionStartTime time when we started the first phase of the action execution.
   * @throws ActionExecutionException if the execution of the specified action
   *   failed for any reason.
   * @throws InterruptedException if the thread was interrupted.
   */
  public void prepareScheduleExecuteAndCompleteAction(Action action, Token token,
      ActionInputFileCache actionInputFileCache, MetadataHandler metadataHandler,
      MiddlemanExpander middlemanExpander, long actionStartTime)
  throws ActionExecutionException, InterruptedException {
    // Delete the metadataHandler's cache of the action's outputs, since they are being deleted.
    metadataHandler.discardMetadata(action.getOutputs());
    // Delete the outputs before executing the action, just to ensure that
    // the action really does produce the outputs.
    try {
      action.prepare();
    } catch (IOException e) {
      reportError("failed to delete output files before executing action", e, action);
    }

    FileOutErr fileOutErr = actionLogBufferPathGenerator.generate();
    postEvent(new ActionStartedEvent(action, actionStartTime));
    try {
      scheduleAndExecuteAction(action,
          new ActionExecutionContext(executorEngine, actionInputFileCache, metadataHandler,
              fileOutErr, middlemanExpander));
      completeAction(action, token, metadataHandler, fileOutErr);
    } finally {
      postEvent(new ActionCompletionEvent(action, action.describeStrategy(executorEngine)));
    }
  }

  public void postActionNotExecutedEvents(Action action, Iterable<Label> rootCauses) {
    for (Label rootCause : rootCauses) {
      postEvent(new ActionNotExecutedEvent(action, rootCause));
    }
  }

  public void createOutputDirectories(Action action) throws ActionExecutionException {
    try {
      Set<Path> done = new HashSet<>(); // avoid redundant calls for the same directory.
      for (Artifact outputFile : action.getOutputs()) {
        Path outputDir = outputFile.getPath().getParentDirectory();
        if (done.add(outputDir)) {
          try {
            createDirectoryAndParents(outputDir);
            continue;
          } catch (IOException e) {
            /* Fall through to plan B. */
          }

          // Possibly some direct ancestors are not directories.  In that case, we unlink all the
          // ancestors until we reach a directory, then try again. This handles the case where a
          // file becomes a directory, either from one build to another, or within a single build.
          try {
            for (Path p = outputDir; !p.isDirectory(); p = p.getParentDirectory()) {
              // p may be a file or dangling symlink.
              p.delete(); // throws IOException
            }
            createDirectoryAndParents(outputDir);
          } catch (IOException e) {
            throw new ActionExecutionException(
                "failed to create output directory '" + outputDir + "'", e, action, false);
          }
        }
      }
    } catch (ActionExecutionException ex) {
      printError(ex.getMessage(), action, null);
      throw ex;
    }
  }

  /**
   * Convenience function for reporting that the action failed due to a
   * particular reason. Reports the message as an error and throws an
   * ActionExecutionException. Does not display any output for the action.
   * See also {@link #reportError(String, Throwable, Action)}.
   *
   * @param message A small text that explains why the action failed
   * @param action The action that failed
   */
  public void reportError(String message, Action action) throws ActionExecutionException {
    reportError(message, null, action);
  }

  /**
   * Convenience function for reporting that the action failed due to a
   * the exception cause. Reports the exceptions' message as error cause and
   * throws an ActionExecutionException that wraps the cause. Does not display
   * any output for the action.
   * See also {@link #reportError(String, Throwable, Action)}.
   *
   * @param cause The exception that caused the action to fail
   * @param action The action that failed
   */
  public void reportError(Throwable cause, Action action) throws ActionExecutionException {
    reportError(cause.getMessage(), cause, action);
  }

  /**
   * Convenience function for reporting that the action failed due to a
   * the exception cause, if there is an additional explanatory message that
   * clarifies the message of the exception. Combines the user-provided message
   * and the exceptions' message and reports the combination as error.
   * Then, throws an ActionExecutionException with the reported error as
   * message and the provided exception as the cause.
   *
   * @param message A small text that explains why the action failed
   * @param cause The exception that caused the action to fail
   * @param action The action that failed
   */
  public void reportError(String message, Throwable cause, Action action)
      throws ActionExecutionException {
    ActionExecutionException ex;
    if (cause == null) {
      ex = new ActionExecutionException(message, action, false);
    } else {
      ex = new ActionExecutionException(message, cause, action, false);
    }
    printError(ex.getMessage(), action, null);
    throw ex;
  }

  /**
   * Execute the specified action, in a profiler task.
   * The caller is responsible for having already checked that we need to
   * execute it and for acquiring/releasing any scheduling locks needed.
   *
   * This is thread-safe so long as you don't try to execute the same action
   * twice at the same time (or overlapping times).
   * May execute in a worker thread.
   *
   * @throws ActionExecutionException if the execution of the specified action
   *   failed for any reason.
   * @throws InterruptedException if the thread was interrupted.
   */
  protected void executeActionTask(Action action, ActionExecutionContext actionExecutionContext)
  throws ActionExecutionException, InterruptedException {

    profiler.startTask(ProfilerTask.ACTION_EXECUTE, action);
    // ActionExecutionExceptions that occur as the thread is interrupted are
    // assumed to be a result of that, so we throw InterruptedException
    // instead.
    FileOutErr outErrBuffer = actionExecutionContext.getFileOutErr();
    try {
      action.execute(actionExecutionContext);

      // Action terminated fine, now report the output.
      // The .shouldShowOutput() method is not necessarily a quick check: in its
      // current implementation it uses regular expression matching.
      if (outErrBuffer.hasRecordedOutput() && action.shouldShowOutput(reporter)) {
        dumpRecordedOutErr(action, outErrBuffer);
      }
      // Defer reporting action success until outputs are checked
    } catch (ActionExecutionException e) {
      reportActionExecution(action, e, outErrBuffer);
      boolean reported = reportErrorIfNotAbortingMode(e, outErrBuffer);

      ActionExecutionException toThrow = e;
      if (reported){
        // If we already printed the error for the exception we mark it as already reported
        // so that we do not print it again in upper levels.
        // Note that we need to report it here since we want immediate feedback of the errors
        // and in some cases the upper-level printing mechanism only prints one of the errors.
        toThrow = new AlreadyReportedActionExecutionException(e);
      }

      // Now, rethrow the exception.
      // This can have two effects:
      // If we're still building, the exception will get retrieved by the
      // completor and rethrown.
      // If we're aborting, the exception will never be retrieved from the
      // completor, since the completor is waiting for all outstanding jobs
      // to finish. After they have finished, it will only rethrow the
      // exception that initially caused it to abort will and not check the
      // exit status of any actions that had finished in the meantime.
      throw toThrow;
    } finally {
      profiler.completeTask(ProfilerTask.ACTION_EXECUTE);
    }
  }

  /**
   * For each of the action's outputs that is a regular file (not a symbolic
   * link or directory), make it read-only and executable.
   *
   * <p>Making the outputs read-only helps preventing accidental editing of
   * them (e.g. in case of generated source code), while making them executable
   * helps running generated files (such as generated shell scripts) on the
   * command line.
   *
   * <p>May execute in a worker thread.
   *
   * <p>Note: setting these bits maintains transparency regarding the locality of the build;
   * because the remote execution engine sets them, they should be set for local builds too.
   *
   * @throws IOException if an I/O error occurred.
   */
  public final void setOutputsReadOnlyAndExecutable(Action action, MetadataHandler metadataHandler)
      throws IOException {
    if (action.getActionType().isMiddleman()) {
      // We "execute" target completion middlemen, but should not attempt to chmod() their
      // virtual outputs.
      return;
    }

    for (Artifact output : action.getOutputs()) {
      Path path = output.getPath();
      if (metadataHandler.isInjected(output)) {
        // We trust the files created by the execution-engine to be non symlinks with expected
        // chmod() settings already applied. The follow stanza implies a total of 6 system calls,
        // since the UnixFileSystem implementation of setWritable() and setExecutable() both
        // do a stat() internally.
        continue;
      }
      if (path.isFile(Symlinks.NOFOLLOW)) { // i.e. regular files only.
        path.setWritable(false);
        path.setExecutable(true);
      }
    }
  }

  public void reportActionExecution(Action action,
      ActionExecutionException exception, FileOutErr outErr) {
    String stdout = null;
    String stderr = null;

    if (outErr.hasRecordedStdout()) {
      stdout = outErr.getOutputFile().toString();
    }
    if (outErr.hasRecordedStderr()) {
      stderr = outErr.getErrorFile().toString();
    }
    postEvent(new ActionExecutedEvent(action, exception, stdout, stderr));
  }

  protected void reportMissingOutputFile(Action action, Artifact output, Reporter reporter,
      boolean isSymlink) {
    boolean genrule = action.getMnemonic().equals("Genrule");
    String prefix = (genrule ? "declared output '" : "output '") + output.prettyPrint() + "' ";
    if (isSymlink) {
      reporter.error(action.getOwner().getLocation(), prefix + "is a dangling symbolic link");
    } else {
      String suffix = genrule ? " by genrule. This is probably "
          + "because the genrule actually didn't create this output, or because the output was a "
          + "directory and the genrule was run remotely (note that only the contents of "
          + "declared file outputs are copied from genrules run remotely." : "";
      reporter.error(action.getOwner().getLocation(), prefix + "was not created" + suffix);
    }
  }

  /**
   * Validates that all action outputs were created.
   *
   * @return false if some outputs are missing, true - otherwise.
   */
  protected boolean checkOutputs(Action action, MetadataHandler metadataHandler) {
    boolean success = true;
    for (Artifact output : action.getOutputs()) {
      if (!metadataHandler.artifactExists(output)) {
        reportMissingOutputFile(action, output, reporter, output.getPath().isSymbolicLink());
        success = false;
      }
    }
    return success;
  }

  protected void completeAction(Action action, Token token, MetadataHandler metadataHandler,
      FileOutErr fileOutErr) throws ActionExecutionException {
    try {
      Preconditions.checkState(action.inputsKnown(),
          "Action %s successfully executed, but inputs still not known", action);

      profiler.startTask(ProfilerTask.ACTION_COMPLETE, action);
      try {
        if (!checkOutputs(action, metadataHandler)) {
          reportError("not all outputs were created", action);
        }
        // Prevent accidental stomping on files.
        // This will also throw a FileNotFoundException
        // if any of the output files doesn't exist.
        try {
          setOutputsReadOnlyAndExecutable(action, metadataHandler);
        } catch (IOException e) {
          reportError("failed to set outputs read-only", e, action);
        }
        updateCache(action, token, metadataHandler);
      } finally {
        profiler.completeTask(ProfilerTask.ACTION_COMPLETE);
      }
      reportActionExecution(action, null, fileOutErr);
    } catch (ActionExecutionException actionException) {
      // Success in execution but failure in completion.
      reportActionExecution(action, actionException, fileOutErr);
      throw actionException;
    } catch (IllegalStateException exception) {
      // More serious internal error, but failure still reported.
      reportActionExecution(action,
          new ActionExecutionException(exception, action, true), fileOutErr);
      throw exception;
    }
  }

  /**
   * For the action 'action' that failed due to 'ex' with the output
   * 'actionOutput', notify the user about the error. To notify the user, the
   * method first displays the output of the action and then reports an error
   * via the reporter. The method ensures that the two messages appear next to
   * each other by locking the outErr object where the output is displayed.
   *
   * @param message The reason why the action failed
   * @param action The action that failed, must not be null.
   * @param actionOutput The output of the failed Action.
   *     May be null, if there is no output to display
   */
  protected abstract void printError(String message, Action action, FileOutErr actionOutput);

  /**
   * Dump the output from the action.
   *
   * @param action The action whose output is being dumped
   * @param outErrBuffer The OutErr that recorded the actions output
   */
  protected void dumpRecordedOutErr(Action action, FileOutErr outErrBuffer) {
    StringBuilder message = new StringBuilder("");
    message.append("From ");
    message.append(action.describe());
    message.append(":");

    // Synchronize this on the reporter, so that the output from multiple
    // actions will not be interleaved.
    synchronized (reporter) {
      // Only print the output if we're not winding down.
      if (isBuilderAborting()) {
        return;
      }
      reporter.info(null, message.toString());

      OutErr outErr = this.reporter.getOutErr();
      outErrBuffer.dumpOutAsLatin1(outErr.getOutputStream());
      outErrBuffer.dumpErrAsLatin1(outErr.getErrorStream());
    }
  }

  /**
   * Returns true if the Builder is winding down (i.e. cancelling outstanding
   * actions and preparing to abort.)
   * The builder is winding down iff:
   * <ul>
   * <li>we had an execution error
   * <li>we are not running with --keep_going
   * </ul>
   */
  protected abstract boolean isBuilderAborting();

  protected abstract void updateCache(Action action, Token token, MetadataHandler metadataHandler)
      throws ActionExecutionException;

  protected abstract void postEvent(Object event);

  /**
   * Returns true if the exception was reported. False otherwise.
   */
  protected abstract boolean reportErrorIfNotAbortingMode(ActionExecutionException ex,
      FileOutErr outErrBuffer);

  /**
   * Execute the specified action, acquiring any scheduling locks needed. The
   * caller is responsible for having already checked that we need to execute
   * it. This function is responsible for profiling.
   *
   * This is thread-safe so long as you don't try to execute the same action
   * twice at the same time (or overlapping times). May execute in a worker
   * thread.
   *
   * @throws ActionExecutionException if the execution of the specified action
   *         failed for any reason.
   * @throws InterruptedException if the thread was interrupted.
   */
  protected abstract void scheduleAndExecuteAction(Action action,
      ActionExecutionContext actionExecutionContext)
  throws ActionExecutionException, InterruptedException;
}
