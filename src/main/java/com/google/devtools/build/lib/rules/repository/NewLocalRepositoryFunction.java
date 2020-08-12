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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ResolvedEvent;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;

/**
 * Create a repository from a directory on the local filesystem.
 */
public class NewLocalRepositoryFunction extends RepositoryFunction {

  @Override
  public boolean isLocal(Rule rule) {
    return true;
  }

  @Override
  public RepositoryDirectoryValue.Builder fetch(
      Rule rule,
      Path outputDirectory,
      BlazeDirectories directories,
      Environment env,
      Map<String, String> markerData,
      SkyKey key)
      throws SkyFunctionException, InterruptedException {

    NewRepositoryFileHandler fileHandler = new NewRepositoryFileHandler(directories.getWorkspace());
    if (!fileHandler.prepareFile(rule, env)) {
      return null;
    }

    String userDefinedPath = getPathAttr(rule);
    PathFragment pathFragment = getTargetPath(userDefinedPath, directories.getWorkspace());

    FileSystem fs = directories.getOutputBase().getFileSystem();
    Path path = fs.getPath(pathFragment);

    RootedPath dirPath = RootedPath.toRootedPath(Root.absoluteRoot(fs), path);

    try {
      FileValue dirFileValue =
          (FileValue) env.getValueOrThrow(FileValue.key(dirPath), IOException.class);
      if (dirFileValue == null) {
        return null;
      }

      if (!dirFileValue.exists()) {
        throw new RepositoryFunctionException(
            new IOException(
                String.format(
                    "The repository's path is \"%s\" (absolute: \"%s\") "
                        + "but this directory does not exist.",
                    userDefinedPath, dirPath.asPath().getPathString())),
            Transience.PERSISTENT);
      }
      if (!dirFileValue.isDirectory()) {
        // Someone tried to create a local repository from a file.
        throw new RepositoryFunctionException(
            new IOException(
                String.format(
                    "The repository's path is \"%s\" (absolute: \"%s\") "
                        + "but this is not a directory.",
                    userDefinedPath, dirPath.asPath().getPathString())),
            Transience.PERSISTENT);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }

    // fetch() creates symlinks to each child under 'path'.
    //
    // On Linux/macOS (i.e. where file symlinks are supported), DiffAwareness handles checking all
    // of these files and directories for changes.
    //
    // On Windows (when file symlinks are disabled), the supposed symlinks are actually copies and
    // DiffAwareness doesn't pick up changes of the "symlink" targets. To rectify that, we request
    // FileValues for each child of 'path'.
    // See https://github.com/bazelbuild/bazel/issues/7063
    //
    // Furthermore, if a new file/directory is added directly under 'path', Bazel doesn't know that
    // this has to be symlinked in. So we also create a dependency on the contents of the 'path'
    // directory.
    SkyKey dirKey = DirectoryListingValue.key(dirPath);
    DirectoryListingValue directoryValue;
    try {
      directoryValue = (DirectoryListingValue) env.getValueOrThrow(
          dirKey, InconsistentFilesystemException.class);
    } catch (InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(new IOException(e), Transience.PERSISTENT);
    }
    if (directoryValue == null) {
      return null;
    }

    Map<SkyKey, SkyValue> fileValues =
        env.getValues(
            Iterables.transform(
                directoryValue.getDirents(),
                e ->
                    (SkyKey)
                        FileValue.key(
                            RootedPath.toRootedPath(
                                dirPath.getRoot(),
                                dirPath.getRootRelativePath().getRelative(e.getName())))));
    if (env.valuesMissing()) {
      return null;
    }

    // Link x/y/z to /some/path/to/y/z.
    if (!symlinkLocalRepositoryContents(outputDirectory, path, userDefinedPath)) {
      return null;
    }

    fileHandler.finishFile(rule, outputDirectory, markerData);
    env.getListener().post(resolve(rule, directories));

    return RepositoryDirectoryValue.builder()
        .setPath(outputDirectory)
        .setSourceDir(directoryValue)
        .setFileValues(fileValues);
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return NewLocalRepositoryRule.class;
  }

  private static ResolvedEvent resolve(Rule rule, BlazeDirectories directories) {
    String name = rule.getName();
    Object pathObj = rule.getAttr("path");
    ImmutableMap.Builder<String, Object> origAttr =
        ImmutableMap.<String, Object>builder().put("name", name).put("path", pathObj);

    StringBuilder repr =
        new StringBuilder()
            .append("new_local_repository(name = ")
            .append(Printer.getPrinter().repr(name))
            .append(", path = ")
            .append(Printer.getPrinter().repr(pathObj));

    Object buildFileObj = rule.getAttr("build_file");
    if ((buildFileObj instanceof String) && ((String) buildFileObj).length() > 0) {
      // Build fiels might refer to an embedded file (as they to for "local_jdk"),
      // so we have to describe the argument in a portable way.
      origAttr.put("build_file", buildFileObj);
      String buildFileArg;
      PathFragment pathFragment = PathFragment.create((String) buildFileObj);
      PathFragment embeddedDir = directories.getEmbeddedBinariesRoot().asFragment();
      if (pathFragment.isAbsolute() && pathFragment.startsWith(embeddedDir)) {
        buildFileArg =
            "__embedded_dir__ + \"/\" + "
                + Printer.getPrinter().repr(pathFragment.relativeTo(embeddedDir).toString());
      } else {
        buildFileArg = Printer.getPrinter().repr(buildFileObj).toString();
      }
      repr.append(", build_file = ").append(buildFileArg);
    } else {
      Object buildFileContentObj = rule.getAttr("build_file_content");
      if (buildFileContentObj != null) {
        origAttr.put("build_file_content", buildFileContentObj);
        repr.append(", build_file_content = ")
            .append(Printer.getPrinter().repr(buildFileContentObj));
      }
    }

    String nativeCommand = repr.append(")").toString();
    ImmutableMap<String, Object> orig = origAttr.build();

    return new ResolvedEvent() {
      @Override
      public String getName() {
        return name;
      }

      @Override
      public Object getResolvedInformation() {
        return ImmutableMap.<String, Object>builder()
            .put(ResolvedHashesFunction.ORIGINAL_RULE_CLASS, "new_local_repository")
            .put(ResolvedHashesFunction.ORIGINAL_ATTRIBUTES, orig)
            .put(ResolvedHashesFunction.NATIVE, nativeCommand)
            .build();
      }
    };
  }
}
