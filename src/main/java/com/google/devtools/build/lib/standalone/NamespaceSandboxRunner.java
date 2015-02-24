// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.standalone;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.unix.FilesystemUtils;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map.Entry;

/**
 * Helper class for running the namespace sandbox. This runner prepares environment inside the
 * sandbox (copies inputs, creates file structure), handles sandbox output, performs cleanup and
 * changes invocation if necessary.
 */
public class NamespaceSandboxRunner {
  private final boolean debug;
  private final PathFragment sandboxDirectory;
  private final Path sandboxPath;
  private final List<String> mounts;
  private final Path embeddedBinaries;
  private final Path tools;
  private final ImmutableList<PathFragment> includeDirectories;
  private final PathFragment includePrefix;
  private final ImmutableMap<PathFragment, Artifact> manifests;
  private final Path execRoot;

  public NamespaceSandboxRunner(BlazeDirectories directories, Spawn spawn,
      PathFragment includePrefix, List<PathFragment> includeDirectories,
      ImmutableMap<PathFragment, Artifact> manifests, boolean debug) {
    String md5sum = Fingerprint.md5Digest(spawn.getResourceOwner().getPrimaryOutput().toString());
    this.sandboxDirectory = new PathFragment("sandbox-root-" + md5sum);
    this.sandboxPath =
        directories.getExecRoot().getRelative("sandboxes").getRelative(sandboxDirectory);
    this.debug = debug;
    this.mounts = new ArrayList<>();
    this.tools = directories.getExecRoot().getChild("tools");
    this.embeddedBinaries = directories.getEmbeddedBinariesRoot();
    this.includePrefix = includePrefix;
    this.includeDirectories = ImmutableList.copyOf(includeDirectories);
    this.manifests = manifests;
    this.execRoot = directories.getExecRoot();
  }

  private void createFileSystem(Collection<? extends ActionInput> outputs) throws IOException {
    // create the sandboxes' parent directory if needed
    // TODO(bazel-team): create this with rest of the workspace dirs
    if (!sandboxPath.getParentDirectory().isDirectory()) {
      FilesystemUtils.mkdir(sandboxPath.getParentDirectory().getPathString(), 0755);
    }

    FilesystemUtils.mkdir(sandboxPath.getPathString(), 0755);
    String[] dirs = { "bin", "etc" };
    for (String dir : dirs) {
      FilesystemUtils.mkdir(sandboxPath.getChild(dir).getPathString(), 0755);
      mounts.add("/" + dir);
    }

    // usr
    String[] dirsUsr = { "bin", "include" };
    FilesystemUtils.mkdir(sandboxPath.getChild("usr").getPathString(), 0755);
    Path usr = sandboxPath.getChild("usr");
    for (String dir : dirsUsr) {
      FilesystemUtils.mkdir(usr.getChild(dir).getPathString(), 0755);
      mounts.add("/usr/" + dir);
    }
    FileSystemUtils.createDirectoryAndParents(usr.getChild("local").getChild("include"));
    mounts.add("/usr/local/include");

    // shared libs
    String[] rootDirs = FilesystemUtils.readdir("/");
    for (String entry : rootDirs) {
      if (entry.startsWith("lib")) {
        FilesystemUtils.mkdir(sandboxPath.getChild(entry).getPathString(), 0755);
        mounts.add("/" + entry);
      }
    }

    String[] usrDirs = FilesystemUtils.readdir("/usr/");
    for (String entry : usrDirs) {
      if (entry.startsWith("lib")) {
        String lib = usr.getChild(entry).getPathString();
        FilesystemUtils.mkdir(lib, 0755);
        mounts.add("/usr/" + entry);
      }
    }

    if (this.includePrefix != null) {
      FilesystemUtils.mkdir(sandboxPath.getRelative(includePrefix).getPathString(), 0755);

      for (PathFragment fullPath : includeDirectories) {
        // includeDirectories should be absolute paths like /usr/include/foo.h. we want to combine
        // them into something like sandbox/include-prefix/usr/include/foo.h - for that we remove
        // the leading '/' from the path string and concatenate with sandbox/include/prefix
        FileSystemUtils.createDirectoryAndParents(sandboxPath.getRelative(includePrefix)
            .getRelative(fullPath.getPathString().substring(1)));
      }
    }
    
    // output directories
    for (ActionInput output : outputs) {
      PathFragment parentDirectory =
          new PathFragment(output.getExecPathString()).getParentDirectory();
      FileSystemUtils.createDirectoryAndParents(sandboxPath.getRelative(parentDirectory));
    }
  }

  public void setupSandbox(List<? extends ActionInput> inputs,
      Collection<? extends ActionInput> outputs) throws IOException {
    createFileSystem(outputs);
    setupBlazeUtils();
    includeManifests();
    copyInputs(inputs);
  }

  private void copyInputs(List<? extends ActionInput> inputs) throws IOException {    
    for (ActionInput input : inputs) {
      if (input.getExecPathString().contains("internal/_middlemen/")) {
        continue;
      }
      // entire tools will be mounted in the sandbox, so don't copy parts of it
      if (input.getExecPathString().startsWith("tools/")) {
        continue;
      }
      Path target = sandboxPath.getRelative(input.getExecPathString());
      Path source = execRoot.getRelative(input.getExecPathString());
      FileSystemUtils.createDirectoryAndParents(target.getParentDirectory());
      File targetFile = new File(target.getPathString());
      // TODO(bazel-team): mount inputs inside sandbox instead of copying
      Files.copy(new File(source.getPathString()), targetFile);
      FilesystemUtils.chmod(targetFile, 0755);
    }
  }

  private void includeManifests() throws IOException {
    for (Entry<PathFragment, Artifact> manifest : this.manifests.entrySet()) {
      String path = manifest.getValue().getPath().getPathString();
      for (String line : Files.readLines(new File(path), Charset.defaultCharset())) {
        String[] fields = line.split(" ");
        String targetPath = sandboxPath.getPathString() + PathFragment.SEPARATOR_CHAR + fields[0];
        String sourcePath = fields[1];
        File source = new File(sourcePath);
        File target = new File(targetPath);
        Files.createParentDirs(target);
        Files.copy(source, target);
      }
    }
  }

  private void setupBlazeUtils() throws IOException {
    Path bin = this.sandboxPath.getChild("_bin");
    if (!bin.isDirectory()) {
      FilesystemUtils.mkdir(bin.getPathString(), 0755);
    }
    Files.copy(new File(this.embeddedBinaries.getChild("build-runfiles").getPathString()),
               new File(bin.getChild("build-runfiles").getPathString()));
    FilesystemUtils.chmod(bin.getChild("build-runfiles").getPathString(), 0755);
    // TODO(bazel-team) filter tools out of input files instead
    // some of the tools could be in inputs; we will mount entire tools anyway so it's just 
    // easier to remove them and remount inside sandbox
    FilesystemUtils.rmTree(sandboxPath.getChild("tools").getPathString());
  }

 
  /**
   * Runs given 
   * 
   * @param spawnArguments - arguments of spawn to run inside the sandbox
   * @param env - environment to run sandbox in
   * @param cwd - current working directory
   * @param outErr - error output to capture sandbox's and command's stderr
   * @throws CommandException
   */
  public void run(List<String> spawnArguments, ImmutableMap<String, String> env, File cwd,
      FileOutErr outErr) throws CommandException {
    List<String> args = new ArrayList<>();
    args.add(execRoot.getRelative("_bin/namespace-sandbox").getPathString());

    // Only for c++ compilation
    if (includePrefix != null) {
      for (PathFragment include : includeDirectories) {
        args.add("-n");
        args.add(include.getPathString());
      }

      args.add("-N");
      args.add(includePrefix.getPathString());
    }

    if (debug) {
      args.add("-D");
    }
    args.add("-t");
    args.add(tools.getPathString());
    
    args.add("-S");
    args.add(sandboxPath.getPathString());
    for (String mount : mounts) {
      args.add("-m");
      args.add(mount);
    }

    args.add("-C");
    args.addAll(spawnArguments);
    Command cmd = new Command(args.toArray(new String[] {}), env, cwd);

    cmd.execute(
    /* stdin */new byte[] {}, 
    Command.NO_OBSERVER, 
    outErr.getOutputStream(),
    outErr.getErrorStream(),
    /* killSubprocessOnInterrupt */true);
  }


  public void cleanup() throws IOException {
    FilesystemUtils.rmTree(sandboxPath.getPathString());
  }

  
  public void copyOutputs(Collection<? extends ActionInput> outputs, FileOutErr outErr)
      throws IOException {
    for (ActionInput output : outputs) {
      Path source = this.sandboxPath.getRelative(output.getExecPathString());
      Path target = this.execRoot.getRelative(output.getExecPathString());
      FileSystemUtils.createDirectoryAndParents(target.getParentDirectory());
      // TODO(bazel-team): eliminate cases when there are excessive outputs in spawns
      // (java compilation expects "srclist" file in its outputs which is sometimes not produced)
      if (source.isFile()) {
        Files.move(new File(source.getPathString()), new File(target.getPathString()));
      } else {
        outErr.getErrorStream().write(("Output wasn't created by action: " + output + "\n")
            .getBytes(StandardCharsets.UTF_8));
      }
    }
  }
}
