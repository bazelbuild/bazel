// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.bazel.debug.WorkspaceRuleEvent;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.rules.repository.NeedsSkyframeRestartException;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor.ExecutionResult;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.InvalidPathException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.Location;

/** A common base class for Starlark "ctx" objects related to external dependencies. */
public abstract class StarlarkBaseExternalContext implements StarlarkValue {
  /** Max. number of command line args added as a profiler description. */
  private static final int MAX_PROFILE_ARGS_LEN = 80;

  protected final Path workingDirectory;
  protected final Environment env;
  protected final ImmutableMap<String, String> envVariables;
  private final StarlarkOS osObject;
  protected final DownloadManager downloadManager;
  protected final double timeoutScaling;
  @Nullable private final ProcessWrapper processWrapper;
  protected final StarlarkSemantics starlarkSemantics;
  private final HashMap<Label, String> accumulatedFileDigests = new HashMap<>();
  private final RepositoryRemoteExecutor remoteExecutor;

  protected StarlarkBaseExternalContext(
      Path workingDirectory,
      Environment env,
      Map<String, String> envVariables,
      DownloadManager downloadManager,
      double timeoutScaling,
      @Nullable ProcessWrapper processWrapper,
      StarlarkSemantics starlarkSemantics,
      @Nullable RepositoryRemoteExecutor remoteExecutor) {
    this.workingDirectory = workingDirectory;
    this.env = env;
    this.envVariables = ImmutableMap.copyOf(envVariables);
    this.osObject = new StarlarkOS(this.envVariables);
    this.downloadManager = downloadManager;
    this.timeoutScaling = timeoutScaling;
    this.processWrapper = processWrapper;
    this.starlarkSemantics = starlarkSemantics;
    this.remoteExecutor = remoteExecutor;
  }

  /** A string that can be used to identify this context object. Used for logging purposes. */
  protected abstract String getIdentifyingStringForLogging();

  /** Returns the file digests used by this context object so far. */
  public ImmutableMap<Label, String> getAccumulatedFileDigests() {
    return ImmutableMap.copyOf(accumulatedFileDigests);
  }

  @StarlarkMethod(
      name = "path",
      doc =
          "Returns a path from a string, label or path. If the path is relative, it will resolve "
              + "relative to the repository directory. If the path is a label, it will resolve to "
              + "the path of the corresponding file. Note that remote repositories are executed "
              + "during the analysis phase and thus cannot depends on a target result (the "
              + "label should point to a non-generated file). If path is a path, it will return "
              + "that path as is.",
      parameters = {
        @Param(
            name = "path",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            doc = "string, label or path from which to create a path from")
      })
  public StarlarkPath path(Object path) throws EvalException, InterruptedException {
    return getPath("path()", path);
  }

  protected StarlarkPath getPath(String method, Object path)
      throws EvalException, InterruptedException {
    if (path instanceof String) {
      PathFragment pathFragment = PathFragment.create(path.toString());
      return new StarlarkPath(
          pathFragment.isAbsolute()
              ? workingDirectory.getFileSystem().getPath(pathFragment)
              : workingDirectory.getRelative(pathFragment));
    } else if (path instanceof Label) {
      return getPathFromLabel((Label) path);
    } else if (path instanceof StarlarkPath) {
      return (StarlarkPath) path;
    } else {
      throw Starlark.errorf("%s can only take a string or a label.", method);
    }
  }

  @StarlarkMethod(
      name = "read",
      doc = "Reads the content of a file on the filesystem.",
      useStarlarkThread = true,
      parameters = {
          @Param(
              name = "path",
              allowedTypes = {
                  @ParamType(type = String.class),
                  @ParamType(type = Label.class),
                  @ParamType(type = StarlarkPath.class)
              },
              doc = "path of the file to read from."),
      })
  public String readFile(Object path, StarlarkThread thread)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    StarlarkPath p = getPath("read()", path);
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newReadEvent(
            p.toString(), getIdentifyingStringForLogging(), thread.getCallerLocation());
    env.getListener().post(w);
    try {
      return FileSystemUtils.readContent(p.getPath(), StandardCharsets.ISO_8859_1);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  // Create parent directories for the given path
  protected static void makeDirectories(Path path) throws IOException {
    Path parent = path.getParentDirectory();
    if (parent != null) {
      parent.createDirectoryAndParents();
    }
  }

  @StarlarkMethod(
      name = "os",
      structField = true,
      doc = "A struct to access information from the system.")
  public StarlarkOS getOS() {
    // Historically this event reported the location of the ctx.os expression, but that's no longer
    // available in the interpreter API. Now we just use a dummy location, and the user must
    // manually inspect the code where this context object is used if they wish to find the
    // offending ctx.os expression.
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newOsEvent(getIdentifyingStringForLogging(), Location.BUILTIN);
    env.getListener().post(w);
    return osObject;
  }

  protected static void createDirectory(Path directory) throws RepositoryFunctionException {
    try {
      if (!directory.exists()) {
        makeDirectories(directory);
        directory.createDirectory();
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    } catch (InvalidPathException e) {
      throw new RepositoryFunctionException(
          Starlark.errorf("Could not create %s: %s", directory, e.getMessage()),
          Transience.PERSISTENT);
    }
  }

  /** Whether this context supports remote execution. */
  protected abstract boolean isRemotable();

  private boolean canExecuteRemote() {
    boolean featureEnabled =
        starlarkSemantics.getBool(BuildLanguageOptions.EXPERIMENTAL_REPO_REMOTE_EXEC);
    boolean remoteExecEnabled = remoteExecutor != null;
    return featureEnabled && isRemotable() && remoteExecEnabled;
  }

  protected abstract ImmutableMap<String, String> getRemoteExecProperties() throws EvalException;

  private Map.Entry<PathFragment, Path> getRemotePathFromLabel(Label label)
      throws EvalException, InterruptedException {
    Path localPath = getPathFromLabel(label).getPath();
    PathFragment remotePath =
        label.getPackageIdentifier().getSourceRoot().getRelative(label.getName());
    return Maps.immutableEntry(remotePath, localPath);
  }

  private StarlarkExecutionResult executeRemote(
      Sequence<?> argumentsUnchecked, // <String> or <Label> expected
      int timeout,
      Map<String, String> environment,
      boolean quiet,
      String workingDirectory)
      throws EvalException, InterruptedException {
    Preconditions.checkState(canExecuteRemote());

    ImmutableSortedMap.Builder<PathFragment, Path> inputsBuilder =
        ImmutableSortedMap.naturalOrder();
    ImmutableList.Builder<String> argumentsBuilder = ImmutableList.builder();
    for (Object argumentUnchecked : argumentsUnchecked) {
      if (argumentUnchecked instanceof Label) {
        Label label = (Label) argumentUnchecked;
        Map.Entry<PathFragment, Path> remotePath = getRemotePathFromLabel(label);
        argumentsBuilder.add(remotePath.getKey().toString());
        inputsBuilder.put(remotePath);
      } else {
        argumentsBuilder.add(argumentUnchecked.toString());
      }
    }

    ImmutableList<String> arguments = argumentsBuilder.build();

    try (SilentCloseable c =
        Profiler.instance()
            .profile(
                ProfilerTask.STARLARK_REPOSITORY_FN, () -> profileArgsDesc("remote", arguments))) {
      ExecutionResult result =
          remoteExecutor.execute(
              arguments,
              inputsBuilder.buildOrThrow(),
              getRemoteExecProperties(),
              ImmutableMap.copyOf(environment),
              workingDirectory,
              Duration.ofSeconds(timeout));

      String stdout = new String(result.stdout(), StandardCharsets.US_ASCII);
      String stderr = new String(result.stderr(), StandardCharsets.US_ASCII);

      if (!quiet) {
        OutErr outErr = OutErr.SYSTEM_OUT_ERR;
        outErr.printOut(stdout);
        outErr.printErr(stderr);
      }

      return new StarlarkExecutionResult(result.exitCode(), stdout, stderr);
    } catch (IOException e) {
      throw Starlark.errorf("remote_execute failed: %s", e.getMessage());
    }
  }

  private void validateExecuteArguments(Sequence<?> arguments) throws EvalException {
    boolean isRemotable = isRemotable();
    for (int i = 0; i < arguments.size(); i++) {
      Object arg = arguments.get(i);
      if (isRemotable) {
        if (!(arg instanceof String || arg instanceof Label)) {
          throw Starlark.errorf("Argument %d of execute is neither a label nor a string.", i);
        }
      } else {
        if (!(arg instanceof String || arg instanceof Label || arg instanceof StarlarkPath)) {
          throw Starlark.errorf("Argument %d of execute is neither a path, label, nor string.", i);
        }
      }
    }
  }

  /** Returns the command line arguments as a string for display in the profiler. */
  private static String profileArgsDesc(String method, List<String> args) {
    StringBuilder b = new StringBuilder();
    b.append(method).append(":");

    final String sep = " ";
    for (String arg : args) {
      int appendLen = sep.length() + arg.length();
      int remainingLen = MAX_PROFILE_ARGS_LEN - b.length();

      if (appendLen <= remainingLen) {
        b.append(sep);
        b.append(arg);
      } else {
        String shortenedArg = (sep + arg).substring(0, remainingLen);
        b.append(shortenedArg);
        b.append("...");
        break;
      }
    }

    return b.toString();
  }

  @StarlarkMethod(
      name = "execute",
      doc =
          "Executes the command given by the list of arguments. The execution time of the command"
              + " is limited by <code>timeout</code> (in seconds, default 600 seconds). This method"
              + " returns an <code>exec_result</code> structure containing the output of the"
              + " command. The <code>environment</code> map can be used to override some"
              + " environment variables to be passed to the process.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "arguments",
            doc =
                "List of arguments, the first element should be the path to the program to "
                    + "execute."),
        @Param(
            name = "timeout",
            named = true,
            defaultValue = "600",
            doc = "maximum duration of the command in seconds (default is 600 seconds)."),
        @Param(
            name = "environment",
            defaultValue = "{}",
            named = true,
            doc = "force some environment variables to be set to be passed to the process."),
        @Param(
            name = "quiet",
            defaultValue = "True",
            named = true,
            doc = "If stdout and stderr should be printed to the terminal."),
        @Param(
            name = "working_directory",
            defaultValue = "\"\"",
            named = true,
            doc =
                "Working directory for command execution.\n"
                    + "Can be relative to the repository root or absolute."),
      })
  public StarlarkExecutionResult execute(
      Sequence<?> arguments, // <String> or <StarlarkPath> or <Label> expected
      StarlarkInt timeoutI,
      Dict<?, ?> uncheckedEnvironment, // <String, String> expected
      boolean quiet,
      String overrideWorkingDirectory,
      StarlarkThread thread)
      throws EvalException, RepositoryFunctionException, InterruptedException {
    validateExecuteArguments(arguments);
    int timeout = Starlark.toInt(timeoutI, "timeout");

    Map<String, String> forceEnvVariables =
        Dict.cast(uncheckedEnvironment, String.class, String.class, "environment");

    if (canExecuteRemote()) {
      return executeRemote(arguments, timeout, forceEnvVariables, quiet, overrideWorkingDirectory);
    }

    // Execute on the local/host machine

    List<String> args = new ArrayList<>(arguments.size());
    for (Object arg : arguments) {
      if (arg instanceof Label) {
        args.add(getPathFromLabel((Label) arg).toString());
      } else {
        // String or StarlarkPath expected
        args.add(arg.toString());
      }
    }

    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newExecuteEvent(
            args,
            timeout,
            envVariables,
            forceEnvVariables,
            workingDirectory.getPathString(),
            quiet,
            getIdentifyingStringForLogging(),
            thread.getCallerLocation());
    env.getListener().post(w);
    createDirectory(workingDirectory);

    long timeoutMillis = Math.round(timeout * 1000L * timeoutScaling);
    if (processWrapper != null) {
      args =
          processWrapper
              .commandLineBuilder(args)
              .setTimeout(Duration.ofMillis(timeoutMillis))
              .build();
    }

    Path workingDirectoryPath;
    if (overrideWorkingDirectory != null && !overrideWorkingDirectory.isEmpty()) {
      workingDirectoryPath = getPath("execute()", overrideWorkingDirectory).getPath();
    } else {
      workingDirectoryPath = workingDirectory;
    }
    createDirectory(workingDirectoryPath);

    final List<String> fargs = args;
    try (SilentCloseable c =
        Profiler.instance()
            .profile(ProfilerTask.STARLARK_REPOSITORY_FN, () -> profileArgsDesc("local", fargs))) {
      return StarlarkExecutionResult.builder(osObject.getEnvironmentVariables())
          .addArguments(args)
          .setDirectory(workingDirectoryPath.getPathFile())
          .addEnvironmentVariables(forceEnvVariables)
          .setTimeout(timeoutMillis)
          .setQuiet(quiet)
          .execute();
    }
  }

  @StarlarkMethod(
      name = "which",
      doc =
          "Returns the path of the corresponding program or None "
              + "if there is no such program in the path.",
      allowReturnNones = true,
      useStarlarkThread = true,
      parameters = {
        @Param(name = "program", named = false, doc = "Program to find in the path."),
      })
  @Nullable
  public StarlarkPath which(String program, StarlarkThread thread) throws EvalException {
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newWhichEvent(
            program, getIdentifyingStringForLogging(), thread.getCallerLocation());
    env.getListener().post(w);
    if (program.contains("/") || program.contains("\\")) {
      throw Starlark.errorf(
          "Program argument of which() may not contain a / or a \\ ('%s' given)", program);
    }
    if (program.length() == 0) {
      throw Starlark.errorf("Program argument of which() may not be empty");
    }
    try {
      StarlarkPath commandPath = findCommandOnPath(program);
      if (commandPath != null) {
        return commandPath;
      }

      if (!program.endsWith(OsUtils.executableExtension())) {
        program += OsUtils.executableExtension();
        return findCommandOnPath(program);
      }
    } catch (IOException e) {
      // IOException when checking executable file means we cannot read the file data so
      // we cannot execute it, swallow the exception.
    }
    return null;
  }

  @Nullable
  private StarlarkPath findCommandOnPath(String program) throws IOException {
    String pathEnvVariable = envVariables.get("PATH");
    if (pathEnvVariable == null) {
      return null;
    }
    for (String p : pathEnvVariable.split(File.pathSeparator)) {
      PathFragment fragment = PathFragment.create(p);
      if (fragment.isAbsolute()) {
        // We ignore relative path as they don't mean much here (relative to where? the workspace
        // root?).
        Path path = workingDirectory.getFileSystem().getPath(fragment).getChild(program.trim());
        if (path.exists() && path.isFile(Symlinks.FOLLOW) && path.isExecutable()) {
          return new StarlarkPath(path);
        }
      }
    }
    return null;
  }

  // Resolve the label given by value into a file path.
  protected StarlarkPath getPathFromLabel(Label label) throws EvalException, InterruptedException {
    RootedPath rootedPath = RepositoryFunction.getRootedPathFromLabel(label, env);
    SkyKey fileSkyKey = FileValue.key(rootedPath);
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);
    } catch (IOException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }

    if (fileValue == null) {
      throw new NeedsSkyframeRestartException();
    }
    if (!fileValue.isFile() || fileValue.isSpecialFile()) {
      throw Starlark.errorf("Not a regular file: %s", rootedPath.asPath().getPathString());
    }

    try {
      accumulatedFileDigests.put(label, RepositoryFunction.fileValueToMarkerValue(fileValue));
    } catch (IOException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }
    return new StarlarkPath(rootedPath.asPath());
  }
}
