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

package com.google.devtools.build.lib.rules.repository;

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * Encapsulates the 2-step behavior of creating workspace and build files for the new_*_repository
 * rules.
 */
public class NewRepositoryFileHandler {

  private NewRepositoryWorkspaceFileHandler workspaceFileHandler;
  private NewRepositoryBuildFileHandler buildFileHandler;

  public NewRepositoryFileHandler(Path workspacePath) {
    this.workspaceFileHandler = new NewRepositoryWorkspaceFileHandler(workspacePath);
    this.buildFileHandler = new NewRepositoryBuildFileHandler(workspacePath);
  }

  public boolean prepareFile(Rule rule, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    if (!this.workspaceFileHandler.prepareFile(rule, env)) {
      return false;
    }
    if (!this.buildFileHandler.prepareFile(rule, env)) {
      return false;
    }

    return true;
  }

  public void finishFile(
      Rule rule, Path outputDirectory, Map<RepoRecordedInput, String> recordedInputValues)
      throws RepositoryFunctionException {
    this.workspaceFileHandler.finishFile(rule, outputDirectory, recordedInputValues);
    this.buildFileHandler.finishFile(rule, outputDirectory, recordedInputValues);
  }

  /**
   * Encapsulates the 2-step behavior of creating files for the new_*_repository rules, based on a
   * pair of attributes defined in {@link #getFileAttrName()} and {@link #getFileContentAttrName()}.
   */
  private abstract static class BaseFileHandler {

    private final String filename;
    private RootedPath rootedPath;
    private FileValue fileValue;
    private String fileContent;

    private BaseFileHandler(String filename) {
      this.filename = filename;
    }

    protected abstract String getFileAttrName();

    protected abstract String getFileContentAttrName();

    protected abstract String getDefaultContent(Rule rule) throws RepositoryFunctionException;

    /**
     * Prepares for writing a file by validating the FOO_file and FOO_file_content attributes of the
     * rule.
     *
     * @return true if the file was successfully created, false if the environment is missing values
     *     (the calling fetch() function should return null in this case).
     * @throws RepositoryFunctionException if the rule does defines both the FOO_file and
     *     FOO_file_content attributes, or if the workspace file could not be retrieved, written, or
     *     symlinked.
     */
    public boolean prepareFile(Rule rule, Environment env)
        throws RepositoryFunctionException, InterruptedException {

      WorkspaceAttributeMapper mapper = WorkspaceAttributeMapper.of(rule);
      boolean hasFile = mapper.isAttributeValueExplicitlySpecified(getFileAttrName());
      boolean hasFileContent = mapper.isAttributeValueExplicitlySpecified(getFileContentAttrName());

      if (hasFile && hasFileContent) {
        throw new RepositoryFunctionException(
            Starlark.errorf(
                "Rule cannot have both a '%s' and '%s' attribute",
                getFileAttrName(), getFileContentAttrName()),
            Transience.PERSISTENT);
      } else if (hasFile) {

        Pair<RootedPath, FileValue> rootedPathAndFileValue = getFileValue(rule, env);
        rootedPath = rootedPathAndFileValue.getFirst();
        fileValue = rootedPathAndFileValue.getSecond();
        if (env.valuesMissing()) {
          return false;
        }

      } else if (hasFileContent) {

        try {
          fileContent = mapper.get(getFileContentAttrName(), Type.STRING);
        } catch (EvalException e) {
          throw new RepositoryFunctionException(e, Transience.PERSISTENT);
        }

      } else {
        fileContent = getDefaultContent(rule);
      }

      return true;
    }

    /**
     * Writes the file, based on the state set by prepareFile().
     *
     * @param outputDirectory the directory to write the file.
     * @throws RepositoryFunctionException if the file could not be written or symlinked
     * @throws IllegalStateException if {@link #prepareFile} was not called before this, or if
     *     {@link #prepareFile} failed and this was called.
     */
    public void finishFile(
        Rule rule, Path outputDirectory, Map<RepoRecordedInput, String> recordedInputValues)
        throws RepositoryFunctionException {
      if (fileValue != null) {
        // Link x/FILENAME to <build_root>/x.FILENAME.
        symlinkFile(rootedPath, fileValue, filename, outputDirectory);
        try {
          Label label = getFileAttributeAsLabel(rule);
          recordedInputValues.put(
              new RepoRecordedInput.File(
                  RepoRecordedInput.RepoCacheFriendlyPath.createInsideWorkspace(
                      label.getRepository(), label.toPathFragment())),
              RepoRecordedInput.File.fileValueToMarkerValue(rootedPath, fileValue));
        } catch (IOException e) {
          throw new RepositoryFunctionException(e, Transience.TRANSIENT);
        }
      } else if (fileContent != null) {
        RepositoryFunction.writeFile(outputDirectory, filename, fileContent);
      } else {
        throw new IllegalStateException("prepareFile() must be called before finishFile()");
      }
    }

    private Label getFileAttributeAsLabel(Rule rule) throws RepositoryFunctionException {
      try {
        return WorkspaceAttributeMapper.of(rule).get(getFileAttrName(), BuildType.NODEP_LABEL);
      } catch (EvalException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
    }

    @Nullable
    private Pair<RootedPath, FileValue> getFileValue(Rule rule, Environment env)
        throws RepositoryFunctionException, InterruptedException {
      Label label = getFileAttributeAsLabel(rule);
      SkyKey pkgSkyKey = PackageLookupValue.key(label.getPackageIdentifier());
      PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(pkgSkyKey);
      if (pkgLookupValue == null) {
        return null;
      }
      if (!pkgLookupValue.packageExists()) {
        String message = pkgLookupValue.getErrorMsg();
        if (pkgLookupValue == PackageLookupValue.NO_BUILD_FILE_VALUE) {
          message =
              PackageLookupFunction.explainNoBuildFileValue(label.getPackageIdentifier(), env);
        }
        throw new RepositoryFunctionException(
            Starlark.errorf("Unable to load package for %s: %s", label, message),
            Transience.PERSISTENT);
      }

      // And now for the file
      Root packageRoot = pkgLookupValue.getRoot();
      RootedPath rootedFile = RootedPath.toRootedPath(packageRoot, label.toPathFragment());
      SkyKey fileKey = FileValue.key(rootedFile);
      FileValue fileValue;
      try {
        // Note that this dependency is, strictly speaking, not necessary: the symlink could simply
        // point to this FileValue and the symlink chasing could be done while loading the package
        // but this results in a nicer error message and it's correct as long as RepositoryFunctions
        // don't write to things in the file system this FileValue depends on. In theory, the latter
        // is possible if the file referenced by workspace_file is a symlink to somewhere under the
        // external/ directory, but if you do that, you are really asking for trouble.
        fileValue = (FileValue) env.getValueOrThrow(fileKey, IOException.class);
        if (fileValue == null) {
          return null;
        }
      } catch (IOException e) {
        throw new RepositoryFunctionException(
            new IOException("Cannot lookup " + label + ": " + e.getMessage()),
            Transience.TRANSIENT);
      }

      if (!fileValue.isFile() || fileValue.isSpecialFile()) {
        throw new RepositoryFunctionException(
            Starlark.errorf(
                "%s is not a regular file; if you're using a relative or absolute path for "
                    + "`build_file` in your `new_local_repository` rule, please switch to using a "
                    + "label instead",
                rootedFile.asPath()),
            Transience.PERSISTENT);
      }

      return Pair.of(rootedFile, fileValue);
    }

    /**
     * Symlinks a file from the local filesystem into the external repository's root.
     *
     * @param rootedPath {@link RootedPath} of the file to be linked in
     * @param fileValue {@link FileValue} representing the file to be linked in
     * @param outputDirectory the directory of the remote repository
     * @throws RepositoryFunctionException if the file specified does not exist or cannot be linked.
     */
    private static void symlinkFile(
        RootedPath rootedPath, FileValue fileValue, String filename, Path outputDirectory)
        throws RepositoryFunctionException {
      Path filePath = outputDirectory.getRelative(filename);
      RepositoryFunction.createSymbolicLink(
          filePath, fileValue.realRootedPath(rootedPath).asPath());
    }
  }

  /**
   * Encapsulates the 2-step behavior of creating workspace files for the new_*_repository rules.
   */
  public static class NewRepositoryWorkspaceFileHandler extends BaseFileHandler {

    public NewRepositoryWorkspaceFileHandler(Path workspacePath) {
      super("WORKSPACE");
    }

    @Override
    protected String getFileAttrName() {
      return "workspace_file";
    }

    @Override
    protected String getFileContentAttrName() {
      return "workspace_file_content";
    }

    @Override
    protected String getDefaultContent(Rule rule) {
      return String.format(
          "# DO NOT EDIT: automatically generated WORKSPACE file for %s\n"
              + "workspace(name = \"%s\")\n",
          rule.getTargetKind(), rule.getName());
    }
  }

  /** Encapsulates the 2-step behavior of creating build files for the new_*_repository rules. */
  public static class NewRepositoryBuildFileHandler extends BaseFileHandler {

    public NewRepositoryBuildFileHandler(Path workspacePath) {
      super("BUILD.bazel");
    }

    @Override
    protected String getFileAttrName() {
      return "build_file";
    }

    @Override
    protected String getFileContentAttrName() {
      return "build_file_content";
    }

    @Override
    protected String getDefaultContent(Rule rule) throws RepositoryFunctionException {
      throw new RepositoryFunctionException(
          Starlark.errorf("Rule requires a 'build_file' or 'build_file_content' attribute"),
          Transience.PERSISTENT);
    }
  }
}
