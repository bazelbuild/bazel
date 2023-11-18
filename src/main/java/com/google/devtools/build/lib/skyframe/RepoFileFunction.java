// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BazelStarlarkEnvironment;
import com.google.devtools.build.lib.packages.DotBazelFileSyntaxChecker;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.packages.PackageArgs;
import com.google.devtools.build.lib.packages.RepoThreadContext;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.SyntaxError;

/** The function to evaluate the REPO.bazel file at the root of a repo. */
public class RepoFileFunction implements SkyFunction {
  private final BazelStarlarkEnvironment starlarkEnv;
  private final Path workspaceRoot;

  public RepoFileFunction(BazelStarlarkEnvironment starlarkEnv, Path workspaceRoot) {
    this.starlarkEnv = starlarkEnv;
    this.workspaceRoot = workspaceRoot;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RepositoryName repoName = (RepositoryName) skyKey.argument();
    // First we need to find the REPO.bazel file. How we do this depends on whether this is for the
    // main repo or an external repo.
    Path repoRoot;
    if (repoName.isMain()) {
      repoRoot = workspaceRoot;
    } else {
      RepositoryDirectoryValue repoDirValue =
          (RepositoryDirectoryValue) env.getValue(RepositoryDirectoryValue.key(repoName));
      if (repoDirValue == null) {
        return null;
      }
      repoRoot = repoDirValue.getPath();
    }
    RootedPath repoFilePath =
        RootedPath.toRootedPath(Root.fromPath(repoRoot), LabelConstants.REPO_FILE_NAME);
    FileValue repoFileValue = (FileValue) env.getValue(FileValue.key(repoFilePath));
    if (repoFileValue == null) {
      return null;
    }
    if (!repoFileValue.exists()) {
      // It's okay to not have a REPO.bazel file.
      return RepoFileValue.of(PackageArgs.EMPTY);
    }

    // Now we can actually evaluate the file.
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    RepositoryMappingValue repoMapping =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(repoName));
    RepositoryMappingValue mainRepoMapping =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (env.valuesMissing()) {
      return null;
    }
    StarlarkFile repoFile = readAndParseRepoFile(repoFilePath.asPath(), env);
    PackageArgs packageArgs =
        evalRepoFile(
            repoFile,
            repoName,
            getDisplayNameForRepo(repoName, mainRepoMapping.getRepositoryMapping()),
            repoMapping.getRepositoryMapping(),
            starlarkSemantics,
            env.getListener());

    return RepoFileValue.of(packageArgs);
  }

  private static StarlarkFile readAndParseRepoFile(Path path, Environment env)
      throws RepoFileFunctionException {
    byte[] contents;
    try {
      contents = FileSystemUtils.readWithKnownFileSize(path, path.getFileSize());
    } catch (IOException e) {
      throw new RepoFileFunctionException(
          new IOException("error reading REPO.bazel file at " + path, e), Transience.TRANSIENT);
    }
    StarlarkFile starlarkFile =
        StarlarkFile.parse(ParserInput.fromUTF8(contents, path.getPathString()));
    if (!starlarkFile.ok()) {
      Event.replayEventsOn(env.getListener(), starlarkFile.errors());
      throw new RepoFileFunctionException(
          new BadRepoFileException("error parsing REPO.bazel file at " + path));
    }
    return starlarkFile;
  }

  private static String getDisplayNameForRepo(
      RepositoryName repoName, RepositoryMapping mainRepoMapping) {
    String displayName = repoName.getDisplayForm(mainRepoMapping);
    if (displayName.isEmpty()) {
      return "the main repo";
    }
    return displayName;
  }

  private PackageArgs evalRepoFile(
      StarlarkFile starlarkFile,
      RepositoryName repoName,
      String repoDisplayName,
      RepositoryMapping repoMapping,
      StarlarkSemantics starlarkSemantics,
      ExtendedEventHandler handler)
      throws RepoFileFunctionException, InterruptedException {
    try (Mutability mu = Mutability.create("repo file", repoName)) {
      new DotBazelFileSyntaxChecker("REPO.bazel files", /* canLoadBzl= */ false)
          .check(starlarkFile);
      Module predeclared =
          Module.withPredeclared(
              starlarkSemantics, starlarkEnv.getStarlarkGlobals().getRepoToplevels());
      Program program = Program.compileFile(starlarkFile, predeclared);
      StarlarkThread thread = new StarlarkThread(mu, starlarkSemantics);
      thread.setPrintHandler(Event.makeDebugPrintHandler(handler));
      RepoThreadContext context =
          new RepoThreadContext(
              new LabelConverter(
                  PackageIdentifier.create(repoName, PathFragment.EMPTY_FRAGMENT), repoMapping));
      context.storeInThread(thread);
      Starlark.execFileProgram(program, predeclared, thread);
      return context.getPackageArgs();
    } catch (SyntaxError.Exception e) {
      Event.replayEventsOn(handler, e.errors());
      throw new RepoFileFunctionException(
          new BadRepoFileException("error parsing REPO.bazel file for " + repoDisplayName, e));
    } catch (EvalException e) {
      handler.handle(Event.error(e.getMessageWithStack()));
      throw new RepoFileFunctionException(
          new BadRepoFileException("error evaluating REPO.bazel file for " + repoDisplayName, e));
    }
  }

  /** Thrown when something is wrong with the contents of the REPO.bazel file of a certain repo. */
  public static class BadRepoFileException extends Exception {
    public BadRepoFileException(String message) {
      super(message);
    }

    public BadRepoFileException(String message, Exception cause) {
      super(message, cause);
    }
  }

  static class RepoFileFunctionException extends SkyFunctionException {
    private RepoFileFunctionException(IOException e, Transience transience) {
      super(e, transience);
    }

    private RepoFileFunctionException(BadRepoFileException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
