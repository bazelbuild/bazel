// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox.backend;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.OutputDirectory;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.OutputSymlink;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/**
 * Brings the outputs of an action that ran in a sandbox backend into the exec root.
 *
 * <p>Nothing is written to the exec root before {@link #commit}: all outputs are first assembled
 * below a private directory, so that an action whose output is not needed in the end, for example
 * the loser of a dynamic execution race, doesn't leave anything behind. The contents of an output
 * file are read with {@code ByteStream.Read}, and the backend usually answers with the file URI of
 * a staged copy instead of the contents, see {@link FileUriBlobs}. Such a file URI is linked, or
 * if that's not possible copied, but never moved, since several outputs with the same digest share
 * one file.
 */
final class SandboxBackendOutputs {
  // A file URI is a path, which is short.
  private static final long MAX_FILE_URI_SIZE = 64 * 1024;

  private final SandboxBackendSession session;
  private final DigestUtil digestUtil;
  private final Path scratch;
  private final AtomicInteger counter = new AtomicInteger();
  // The outputs, as the paths below the scratch directory and the exec root they are moved to.
  private final Map<Path, Path> outputs = new HashMap<>();
  private final List<ListenableFuture<Void>> downloads = new ArrayList<>();

  /**
   * @param scratch the directory to assemble the outputs in. Has to be on the file system of the
   *     exec root so that outputs can be moved cheaply, and is created and deleted by the caller.
   */
  SandboxBackendOutputs(SandboxBackendSession session, Path scratch) {
    this.session = session;
    this.digestUtil = session.digestUtil();
    this.scratch = scratch;
  }

  /**
   * Starts to assemble the files and directories described by {@code result}, and the contents of
   * the file {@code stdoutOutput} if the standard output of the action was redirected into one.
   *
   * <p>Nothing that the backend returns is trusted to stay within what the action declared: every
   * path in {@code result} has to be one of {@code declaredOutputPaths}, and the names in a tree
   * have to be plain names. Outputs that weren't declared are an error rather than ignored, since
   * a backend that reports them is confused about the action it ran, and what else it got wrong
   * can't be known. The target of a symlink is not checked, as symlinks that point anywhere are
   * legitimate outputs.
   *
   * @param declaredOutputPaths the {@code output_paths} that were sent in the command
   * @param stdout where the standard output of the action goes: a file output if non-null,
   *     otherwise {@code outErr}
   */
  void download(
      RemoteActionExecutionContext context,
      ActionResult result,
      RemotePathResolver resolver,
      Collection<String> declaredOutputPaths,
      @Nullable Path stdoutOutput)
      throws IOException {
    Set<String> declared = new HashSet<>();
    for (String path : declaredOutputPaths) {
      declared.add(PathFragment.create(unicodeToInternal(path)).getPathString());
    }
    for (OutputFile file : result.getOutputFilesList()) {
      Path real = resolver.outputPathToLocalPath(declaredOutput(declared, file.getPath()));
      if (outputs.containsKey(real)) {
        continue;
      }
      Path assembled = newScratchPath();
      outputs.put(real, assembled);
      downloads.add(fetchFile(context, file.getDigest(), assembled, file.getIsExecutable()));
    }
    for (OutputDirectory directory : result.getOutputDirectoriesList()) {
      Path real = resolver.outputPathToLocalPath(declaredOutput(declared, directory.getPath()));
      if (outputs.containsKey(real)) {
        continue;
      }
      Path assembled = newScratchPath();
      outputs.put(real, assembled);
      downloads.add(fetchTree(context, directory.getTreeDigest(), assembled));
    }
    List<OutputSymlink> symlinks = new ArrayList<>(result.getOutputSymlinksList());
    symlinks.addAll(result.getOutputFileSymlinksList());
    symlinks.addAll(result.getOutputDirectorySymlinksList());
    for (OutputSymlink symlink : symlinks) {
      Path real = resolver.outputPathToLocalPath(declaredOutput(declared, symlink.getPath()));
      if (outputs.containsKey(real)) {
        continue;
      }
      Path assembled = newScratchPath();
      outputs.put(real, assembled);
      assembled.createSymbolicLink(PathFragment.create(unicodeToInternal(symlink.getTarget())));
    }
    if (stdoutOutput != null) {
      Path assembled = newScratchPath();
      outputs.put(stdoutOutput, assembled);
      if (!result.getStdoutRaw().isEmpty()) {
        try (OutputStream out = assembled.getOutputStream()) {
          result.getStdoutRaw().writeTo(out);
        }
      } else if (result.hasStdoutDigest()) {
        downloads.add(fetchFile(context, result.getStdoutDigest(), assembled, false));
      } else {
        FileSystemUtils.createEmptyFile(assembled);
      }
    }
  }

  /**
   * Returns {@code path}, as a Bazel internal string, if it is one of the {@code declared} paths.
   * Those are relative to the exec root, so a path that is one of them can't leave it.
   */
  private static String declaredOutput(Set<String> declared, String path) throws IOException {
    String internal = unicodeToInternal(path);
    if (path.isEmpty() || path.indexOf('\0') >= 0 || path.startsWith("/")) {
      throw new IOException(
          "sandbox backend returned the invalid output path '" + printable(path) + "'");
    }
    PathFragment normalized = PathFragment.create(internal);
    if (normalized.containsUplevelReferences() || !declared.contains(normalized.getPathString())) {
      throw new IOException(
          "sandbox backend returned the output '" + printable(path) + "', which the action didn't declare");
    }
    return internal;
  }

  /** Returns {@code name}, a name in a tree, if it can't refer to anything but a child. */
  private static String childName(String name) throws IOException {
    if (name.isEmpty()
        || name.equals(".")
        || name.equals("..")
        || name.indexOf('/') >= 0
        || name.indexOf('\0') >= 0) {
      throw new IOException(
          "sandbox backend returned a tree with the invalid name '" + printable(name) + "'");
    }
    return unicodeToInternal(name);
  }

  private static String printable(String s) {
    return s.replace("\0", "\\0");
  }

  /**
   * Writes the standard output and error that the backend returned in {@code result}, unless
   * standard output was redirected into an output file, to {@code out} and {@code err}.
   */
  void downloadOutErr(
      RemoteActionExecutionContext context,
      ActionResult result,
      @Nullable OutputStream out,
      OutputStream err)
      throws IOException {
    if (out != null) {
      writeStream(context, result.getStdoutRaw(), result.getStdoutDigest(), out);
    }
    writeStream(context, result.getStderrRaw(), result.getStderrDigest(), err);
  }

  private void writeStream(
      RemoteActionExecutionContext context, ByteString raw, Digest digest, OutputStream out)
      throws IOException {
    if (!raw.isEmpty()) {
      raw.writeTo(out);
    } else if (digest.getSizeBytes() > 0) {
      Path assembled = newScratchPath();
      waitFor(fetchFile(context, digest, assembled, false));
      try (InputStream in = assembled.getInputStream()) {
        in.transferTo(out);
      }
    }
    out.flush();
  }

  /** Waits for all downloads to finish. */
  void await() throws IOException, InterruptedException {
    try {
      Futures.allAsList(downloads).get();
    } catch (ExecutionException e) {
      throw asIOException(e);
    } catch (InterruptedException e) {
      for (ListenableFuture<Void> download : downloads) {
        download.cancel(/* mayInterruptIfRunning= */ true);
      }
      throw e;
    }
  }

  /** Moves the outputs to where they belong. Replaces what is there. */
  void commit() throws IOException {
    for (Map.Entry<Path, Path> output : outputs.entrySet()) {
      Path real = output.getKey();
      Path assembled = output.getValue();
      real.getParentDirectory().createDirectoryAndParents();
      real.deleteTree();
      try {
        assembled.renameTo(real);
      } catch (IOException e) {
        // Possibly across file systems.
        if (assembled.isDirectory()) {
          real.createDirectory();
          FileSystemUtils.copyTreesBelow(assembled, real);
        } else {
          FileSystemUtils.copyFile(assembled, real);
        }
      }
    }
  }

  private Path newScratchPath() {
    return scratch.getRelative("o" + counter.getAndIncrement());
  }

  private ListenableFuture<Void> fetchTree(
      RemoteActionExecutionContext context, Digest treeDigest, Path assembled) {
    Path treeBlob = newScratchPath();
    return Futures.transformAsync(
        fetchFile(context, treeDigest, treeBlob, false),
        unused -> {
          Tree tree;
          try (InputStream in = treeBlob.getInputStream()) {
            tree = Tree.parseFrom(in);
          } catch (InvalidProtocolBufferException e) {
            throw new IOException("sandbox backend returned an invalid tree " + treeDigest, e);
          }
          Map<String, Directory> children = new HashMap<>();
          for (Directory child : tree.getChildrenList()) {
            children.put(digestUtil.compute(child).getHash(), child);
          }
          List<ListenableFuture<Void>> files = new ArrayList<>();
          assembled.createDirectoryAndParents();
          populate(context, tree.getRoot(), children, assembled, files);
          return Futures.whenAllSucceed(files).call(() -> null, directExecutor());
        },
        directExecutor());
  }

  private void populate(
      RemoteActionExecutionContext context,
      Directory directory,
      Map<String, Directory> children,
      Path path,
      List<ListenableFuture<Void>> files)
      throws IOException {
    for (FileNode file : directory.getFilesList()) {
      files.add(
          fetchFile(
              context,
              file.getDigest(),
              path.getRelative(childName(file.getName())),
              file.getIsExecutable()));
    }
    for (SymlinkNode symlink : directory.getSymlinksList()) {
      path.getRelative(childName(symlink.getName()))
          .createSymbolicLink(PathFragment.create(unicodeToInternal(symlink.getTarget())));
    }
    for (DirectoryNode node : directory.getDirectoriesList()) {
      Directory child = children.get(node.getDigest().getHash());
      if (child == null) {
        throw new IOException(
            "sandbox backend returned a tree without the directory " + node.getName());
      }
      Path childPath = path.getRelative(childName(node.getName()));
      childPath.createDirectory();
      populate(context, child, children, childPath, files);
    }
  }

  /** Makes {@code assembled} a file with the contents described by {@code digest}. */
  private ListenableFuture<Void> fetchFile(
      RemoteActionExecutionContext context, Digest digest, Path assembled, boolean executable) {
    try {
      assembled.getParentDirectory().createDirectoryAndParents();
      if (digest.getSizeBytes() == 0) {
        FileSystemUtils.createEmptyFile(assembled);
        assembled.setExecutable(executable);
        return Futures.immediateVoidFuture();
      }
      Path received = newScratchPath();
      OutputStream out = received.getOutputStream();
      ListenableFuture<Void> download =
          session.cacheClient().downloadStreamedBytes(context, digest, out);
      download.addListener(
          () -> {
            try {
              out.close();
            } catch (IOException e) {
              // Reported by the transformation below, if it matters.
            }
          },
          directExecutor());
      return Futures.transformAsync(
          Futures.catchingAsync(
              download,
              CacheNotFoundException.class,
              e ->
                  Futures.immediateFailedFuture(
                      new IOException("sandbox backend has lost the output " + digest.getHash(), e)),
              directExecutor()),
          unused -> {
            place(digest, received, assembled, executable);
            return Futures.immediateVoidFuture();
          },
          directExecutor());
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }
  }

  /** Turns what was received for {@code digest} into the file {@code assembled}. */
  private void place(Digest digest, Path received, Path assembled, boolean executable)
      throws IOException {
    long size = received.getFileSize();
    if (startsWithFileScheme(received)) {
      // This is the same decision as FileUriBlobs#isFileUri, without reading large files.
      boolean isBlob = size == digest.getSizeBytes() && hasDigest(received, digest);
      if (!isBlob) {
        if (size > MAX_FILE_URI_SIZE) {
          throw new IOException(
              String.format(
                  "sandbox backend returned %d bytes that don't match %s/%d",
                  size, digest.getHash(), digest.getSizeBytes()));
        }
        String stagedPath;
        try {
          stagedPath = FileUriBlobs.decode(ByteString.copyFrom(FileSystemUtils.readContent(received)));
        } catch (IllegalArgumentException e) {
          throw new IOException(
              "sandbox backend returned an invalid file URI: " + e.getMessage(), e);
        }
        if (stagedPath.indexOf('\0') >= 0) {
          throw new IOException("sandbox backend returned a file URI with a NUL byte");
        }
        link(digest, received.getFileSystem().getPath(stagedPath), assembled, executable);
        received.delete();
        return;
      }
    }
    if (size != digest.getSizeBytes()) {
      throw new IOException(
          String.format(
              "sandbox backend returned %d bytes for %s/%d",
              size, digest.getHash(), digest.getSizeBytes()));
    }
    received.renameTo(assembled);
    assembled.setExecutable(executable);
  }

  private static boolean startsWithFileScheme(Path path) throws IOException {
    byte[] scheme = new byte[7];
    try (InputStream in = path.getInputStream()) {
      return in.readNBytes(scheme, 0, scheme.length) == scheme.length
          && ByteString.copyFrom(scheme).equals(ByteString.copyFromUtf8("file://"));
    }
  }

  private boolean hasDigest(Path path, Digest digest) throws IOException {
    return digestUtil.compute(path, path.stat()).getHash().equals(digest.getHash());
  }

  /**
   * Makes {@code assembled} a file with the contents of {@code staged} without moving it.
   *
   * <p>The staged file has to exist and to be a regular file of the size that the digest says. Its
   * contents are not hashed, which would read every output once more: the backend is trusted to
   * stage what it digested, and the size is the cheap check that catches a backend that names the
   * wrong file.
   */
  private static void link(Digest digest, Path staged, Path assembled, boolean executable)
      throws IOException {
    FileStatus status = staged.statIfFound();
    if (status == null) {
      throw new IOException(
          "sandbox backend returned the file URI of a file that doesn't exist: " + staged);
    }
    if (!status.isFile()) {
      throw new IOException(
          "sandbox backend returned the file URI of something that isn't a regular file: "
              + staged);
    }
    if (status.getSize() != digest.getSizeBytes()) {
      throw new IOException(
          String.format(
              "sandbox backend returned the file URI of %s, which has %d bytes instead of the %d of"
                  + " %s",
              staged, status.getSize(), digest.getSizeBytes(), digest.getHash()));
    }
    boolean stagedExecutable = (status.getPermissions() & 0100) != 0;
    // Changing the permissions of a hard link changes those of all names of the file.
    if (stagedExecutable == executable) {
      try {
        staged.createHardLink(assembled);
        return;
      } catch (IOException e) {
        // Possibly across file systems.
      }
    }
    FileSystemUtils.copyFile(staged, assembled);
    assembled.setExecutable(executable);
  }

  private static void waitFor(ListenableFuture<Void> future) throws IOException {
    try {
      future.get();
    } catch (ExecutionException e) {
      throw asIOException(e);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException(e);
    }
  }

  private static IOException asIOException(ExecutionException e) {
    return e.getCause() instanceof IOException io ? io : new IOException(e.getCause());
  }
}
