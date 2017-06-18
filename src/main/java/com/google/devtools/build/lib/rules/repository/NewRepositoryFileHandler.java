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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;

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

  public void finishFile(Path outputDirectory) throws RepositoryFunctionException {
    this.workspaceFileHandler.finishFile(outputDirectory);
    this.buildFileHandler.finishFile(outputDirectory);
  }

  /**
   * Encapsulates the 2-step behavior of creating files for the new_*_repository rules, based on a
   * pair of attributes defined in {@link #getFileAttrName()} and {@link #getFileContentAttrName()}.
   */
  private abstract static class BaseFileHandler {

    private final Path workspacePath;
    private final String filename;
    private FileValue fileValue;
    private String fileContent;

    private BaseFileHandler(Path workspacePath, String filename) {
      this.workspacePath = workspacePath;
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

        String error =
            String.format(
                "Rule %s cannot have both a '%s' and '%s' attribute",
                rule, getFileAttrName(), getFileContentAttrName());
        throw new RepositoryFunctionException(
            new EvalException(rule.getLocation(), error), Transience.PERSISTENT);

      } else if (hasFile) {

        fileValue = getFileValue(rule, env);
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
    public void finishFile(Path outputDirectory) throws RepositoryFunctionException {
      if (fileValue != null) {
        // Link x/FILENAME to <build_root>/x.FILENAME.
        symlinkFile(fileValue, filename, outputDirectory);
      } else if (fileContent != null) {
        RepositoryFunction.writeFile(outputDirectory, filename, fileContent);
      } else {
        throw new IllegalStateException("prepareFile() must be called before finishFile()");
      }
    }

    private FileValue getFileValue(Rule rule, Environment env)
        throws RepositoryFunctionException, InterruptedException {
      WorkspaceAttributeMapper mapper = WorkspaceAttributeMapper.of(rule);
      String fileAttribute;
      try {
        fileAttribute = mapper.get(getFileAttrName(), Type.STRING);
      } catch (EvalException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
      RootedPath rootedFile;

      if (LabelValidator.isAbsolute(fileAttribute)) {
        try {
          // Parse a label
          Label label = Label.parseAbsolute(fileAttribute);
          SkyKey pkgSkyKey = PackageLookupValue.key(label.getPackageIdentifier());
          PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(pkgSkyKey);
          if (pkgLookupValue == null) {
            return null;
          }
          if (!pkgLookupValue.packageExists()) {
            throw new RepositoryFunctionException(
                new EvalException(
                    rule.getLocation(),
                    "Unable to load package for " + fileAttribute + ": not found."),
                Transience.PERSISTENT);
          }

          // And now for the file
          Path packageRoot = pkgLookupValue.getRoot();
          rootedFile = RootedPath.toRootedPath(packageRoot, label.toPathFragment());
        } catch (LabelSyntaxException ex) {
          throw new RepositoryFunctionException(
              new EvalException(
                  rule.getLocation(),
                  String.format(
                      "In %s the '%s' attribute does not specify a valid label: %s",
                      rule, getFileAttrName(), ex.getMessage())),
              Transience.PERSISTENT);
        }
      } else {
        // TODO(dmarting): deprecate using a path for the workspace_file attribute.
        PathFragment file = PathFragment.create(fileAttribute);
        Path fileTarget = workspacePath.getRelative(file);
        if (!fileTarget.exists()) {
          throw new RepositoryFunctionException(
              new EvalException(
                  rule.getLocation(),
                  String.format(
                      "In %s the '%s' attribute does not specify an existing file "
                          + "(%s does not exist)",
                      rule, getFileAttrName(), fileTarget)),
              Transience.PERSISTENT);
        }

        if (file.isAbsolute()) {
          rootedFile =
              RootedPath.toRootedPath(
                  fileTarget.getParentDirectory(), PathFragment.create(fileTarget.getBaseName()));
        } else {
          rootedFile = RootedPath.toRootedPath(workspacePath, file);
        }
      }
      SkyKey fileKey = FileValue.key(rootedFile);
      FileValue fileValue;
      try {
        // Note that this dependency is, strictly speaking, not necessary: the symlink could simply
        // point to this FileValue and the symlink chasing could be done while loading the package
        // but this results in a nicer error message and it's correct as long as RepositoryFunctions
        // don't write to things in the file system this FileValue depends on. In theory, the latter
        // is possible if the file referenced by workspace_file is a symlink to somewhere under the
        // external/ directory, but if you do that, you are really asking for trouble.
        fileValue =
            (FileValue)
                env.getValueOrThrow(
                    fileKey,
                    IOException.class,
                    FileSymlinkException.class,
                    InconsistentFilesystemException.class);
        if (fileValue == null) {
          return null;
        }
      } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
        throw new RepositoryFunctionException(
            new IOException("Cannot lookup " + fileAttribute + ": " + e.getMessage()),
            Transience.TRANSIENT);
      }

      return fileValue;
    }

    /**
     * Symlinks a file from the local filesystem into the external repository's root.
     *
     * @param fileValue {@link FileValue} representing the file to be linked in
     * @param outputDirectory the directory of the remote repository
     * @throws RepositoryFunctionException if the file specified does not exist or cannot be linked.
     */
    private static void symlinkFile(FileValue fileValue, String filename, Path outputDirectory)
        throws RepositoryFunctionException {
      Path filePath = outputDirectory.getRelative(filename);
      RepositoryFunction.createSymbolicLink(filePath, fileValue.realRootedPath().asPath());
    }
  }

  /**
   * Encapsulates the 2-step behavior of creating workspace files for the new_*_repository rules.
   */
  public static class NewRepositoryWorkspaceFileHandler extends BaseFileHandler {

    public NewRepositoryWorkspaceFileHandler(Path workspacePath) {
      super(workspacePath, "WORKSPACE");
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
      super(workspacePath, "BUILD.bazel");
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
      String error =
          String.format("Rule %s requires a 'build_file' or 'build_file_content' attribute", rule);
      throw new RepositoryFunctionException(
          new EvalException(rule.getLocation(), error), Transience.PERSISTENT);
    }
  }
}
