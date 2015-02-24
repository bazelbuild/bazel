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
package com.google.devtools.build.lib.standalone;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.unix.FilesystemUtils;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.DependencySet;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/**
 * Strategy that uses sandboxing to execute a process.
 */
@ExecutionStrategy(name = {"sandboxed"}, 
                   contextType = SpawnActionContext.class)
public class LinuxSandboxedStrategy implements SpawnActionContext {
  private final boolean verboseFailures;
  private final BlazeDirectories directories;
  
  public LinuxSandboxedStrategy(BlazeDirectories blazeDirectories, boolean verboseFailures) {
    this.directories = blazeDirectories;
    this.verboseFailures = verboseFailures;
  }

  /**
   * Executes the given {@code spawn}.
   */
  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException {
    Executor executor = actionExecutionContext.getExecutor();
    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(Label.print(spawn.getOwner().getLabel()),
          spawn.asShellCommand(executor.getExecRoot()));
    }
    boolean processHeaders = spawn.getResourceOwner() instanceof CppCompileAction;
    
    Path execPath = this.directories.getExecRoot();
    List<String> spawnArguments = new ArrayList<>();

    for (String arg : spawn.getArguments()) {
      if (arg.startsWith(execPath.getPathString())) {
        // make all paths relative for the sandbox
        spawnArguments.add(arg.substring(execPath.getPathString().length()));
      } else {
        spawnArguments.add(arg);
      }
    }

    List<? extends ActionInput> expandedInputs =
        ActionInputHelper.expandMiddlemen(spawn.getInputFiles(),
            actionExecutionContext.getMiddlemanExpander());
    
    String cwd = executor.getExecRoot().getPathString();

    FileOutErr outErr = actionExecutionContext.getFileOutErr();
    try {
      PathFragment includePrefix = null; // null when there's no include mangling to do
      List<PathFragment> includeDirectories = ImmutableList.of();
      if (processHeaders) {
        CppCompileAction cppAction = (CppCompileAction) spawn.getResourceOwner();
        // headers are mounted in the sandbox in a separate include dir, so their names are mangled
        // when running the compilation and will have to be unmangled after it's done in the *.pic.d
        includeDirectories = extractIncludeDirs(execPath, cppAction, spawnArguments);
        includePrefix = getSandboxIncludeDir(cppAction);
      }      
      
      NamespaceSandboxRunner runner = new NamespaceSandboxRunner(directories, spawn, includePrefix,
          includeDirectories, spawn.getRunfilesManifests(), verboseFailures);
      runner.setupSandbox(expandedInputs, spawn.getOutputFiles());
      runner.run(spawnArguments, spawn.getEnvironment(), new File(cwd), outErr);
      runner.copyOutputs(spawn.getOutputFiles(), outErr);
      if (processHeaders) {
        CppCompileAction cppAction = (CppCompileAction) spawn.getResourceOwner();
        unmangleHeaderFiles(cppAction);
      }
      runner.cleanup();
    } catch (CommandException e) {
      String message = CommandFailureUtils.describeCommandFailure(verboseFailures,
          spawn.getArguments(), spawn.getEnvironment(), cwd);
      throw new UserExecException(String.format("%s: %s", message, e));
    } catch (IOException e) {
      throw new UserExecException(e.getMessage());
    }
  }

  private void unmangleHeaderFiles(CppCompileAction cppCompileAction) throws IOException {
    Path execPath = this.directories.getExecRoot();
    CppCompileAction.DotdFile dotdfile = cppCompileAction.getDotdFile();
    DependencySet depset = new DependencySet(execPath).read(dotdfile.getPath());
    DependencySet unmangled = new DependencySet(execPath);
    PathFragment sandboxIncludeDir = getSandboxIncludeDir(cppCompileAction);
    PathFragment prefix = sandboxIncludeDir.getRelative(execPath.asFragment().relativeTo("/"));
    for (PathFragment dep : depset.getDependencies()) {
      if (dep.startsWith(prefix)) {
        dep = dep.relativeTo(prefix);
      }
      unmangled.addDependency(dep);
    }
    unmangled.write(execPath.getRelative(depset.getOutputFileName()), ".d");
  }

  private PathFragment getSandboxIncludeDir(CppCompileAction cppCompileAction) {
    return new PathFragment(
        "include-" + Actions.escapedPath(cppCompileAction.getPrimaryOutput().toString()));
  }

  private ImmutableList<PathFragment> extractIncludeDirs(Path execPath,
      CppCompileAction cppCompileAction, List<String> spawnArguments) throws IOException {
    List<PathFragment> includes = new ArrayList<>();
    includes.addAll(cppCompileAction.getQuoteIncludeDirs());
    includes.addAll(cppCompileAction.getIncludeDirs());
    includes.addAll(cppCompileAction.getSystemIncludeDirs());
    
    // gcc implicitly includes headers in the same dir as .cc file
    PathFragment sourceDirectory =
        cppCompileAction.getSourceFile().getPath().getParentDirectory().asFragment();
    includes.add(sourceDirectory);
    spawnArguments.add("-iquote");
    spawnArguments.add(sourceDirectory.toString());
    
    TreeSet<PathFragment> processedIncludes = new TreeSet<>();
    for (int i = 0; i < includes.size(); i++) {
      PathFragment absolutePath;
      if (!includes.get(i).isAbsolute()) {
        absolutePath = execPath.getRelative(includes.get(i)).asFragment();
      } else {
        absolutePath = includes.get(i);
      }
      // CppCompileAction may provide execPath as one of the include directories. This is a big 
      // overestimation of what is actually needed and doesn't make for very hermetic sandbox
      // (since everything from the workspace will be somehow accessed in the sandbox). To have
      // some more hermeticity in this situation we mount all the include dirs in:
      // sandbox-directory/include-prefix/actual-include-dir
      // (where include-prefix is obtained from this.getSandboxIncludeDir(cppCompileAction))
      // and make so gcc looks there for includes. This should prevent the user from accessing
      // files that technically should not be in the sandbox.
      // TODO(bazel-team): change CppCompileAction so that include dirs contain only subsets of the
      // execPath
      if (absolutePath.equals(execPath.asFragment())) {
        // we can't mount execPath because it will lead to a circular mount; instead mount its
        // subdirs inside (other than the ones containing sandbox)
        String[] subdirs = FilesystemUtils.readdir(absolutePath.toString());
        for (String dirName : subdirs) {
          if (dirName.equals("_bin") || dirName.equals("bazel-out")) {
            continue;
          }
          PathFragment child = absolutePath.getChild(dirName);
          processedIncludes.add(child);
        }
      } else {
        processedIncludes.add(absolutePath);
      }
    }
    
    // pseudo random name for include directory inside sandbox, so it won't be accessed by accident
    String prefix = getSandboxIncludeDir(cppCompileAction).toString();
    
    // change names in the invocation
    for (int i = 0; i < spawnArguments.size(); i++) {
      if (spawnArguments.get(i).startsWith("-I")) {
        String argument = spawnArguments.get(i).substring(2);
        spawnArguments.set(i, setIncludeDirSandboxPath(execPath, argument, "-I" + prefix));
      }
      if (spawnArguments.get(i).equals("-iquote") || spawnArguments.get(i).equals("-isystem")) {
        spawnArguments.set(i + 1, setIncludeDirSandboxPath(execPath, 
            spawnArguments.get(i + 1), prefix));  
      }
    }
    return ImmutableList.copyOf(processedIncludes);
  }

  private String setIncludeDirSandboxPath(Path execPath, String argument, String prefix) {
    StringBuilder builder = new StringBuilder(prefix);
    if (argument.charAt(0) != '/') {
      // relative path
      builder.append(execPath);
      builder.append('/');
    }
    builder.append(argument);
    
    return builder.toString();
  }

  @Override
  public String strategyLocality(String mnemonic, boolean remotable) {
    return "linux-sandboxing";
  }

  @Override
  public boolean isRemotable(String mnemonic, boolean remotable) {
    return false;
  }
}
