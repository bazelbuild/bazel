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
package com.google.devtools.build.lib.actions.cache;

import com.google.common.hash.HashingOutputStream;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.util.StreamWriter;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;

/**
 * An ActionInput that does not actually exist on the filesystem, but can still be written to an
 * OutputStream.
 */
public abstract class VirtualActionInput implements ActionInput, StreamWriter {
  /**
   * An empty virtual artifact <b>without</b> an execpath. This is used to denote empty files in
   * runfiles and filesets.
   */
  public static final VirtualActionInput EMPTY_MARKER = new EmptyActionInput();

  /**
   * Writes a virtual input file so that the final file is always consistent to all readers.
   *
   * <p>This function exists to aid dynamic scheduling. Param files are inputs, so they need to be
   * written without holding the output lock. When we have competing unsandboxed spawn runners (like
   * persistent workers), it's possible for them to clash in these writes, either encountering
   * missing file errors or encountering incomplete data. But given that we can assume both spawn
   * runners will write the same contents, we can write those as temporary files and then perform a
   * rename, which has atomic semantics on Unix, and thus keep all readers always seeing consistent
   * contents. This may cause a race condition on Windows.
   *
   * @param execRoot the path that this input should be written inside, typically the execroot
   * @param uniqueSuffix a filename extension that is different between the local spawn runners and
   *     the remote ones
   * @return digest of written virtual input
   * @throws IOException if we fail to write the virtual input file
   */
  @CanIgnoreReturnValue
  public byte[] atomicallyWriteRelativeTo(Path execRoot, String uniqueSuffix) throws IOException {
    Path outputPath = execRoot.getRelative(getExecPath());
    return atomicallyWriteTo(outputPath, uniqueSuffix);
  }

  /**
   * Like {@link #atomicallyWriteRelativeTo(Path, String)}, but takes the full path that the input
   * should be written to.
   */
  @CanIgnoreReturnValue
  protected byte[] atomicallyWriteTo(Path outputPath, String uniqueSuffix) throws IOException {
    Path tmpPath = outputPath.getFileSystem().getPath(outputPath.getPathString() + uniqueSuffix);
    tmpPath.getParentDirectory().createDirectoryAndParents();
    try {
      byte[] digest = writeTo(tmpPath);
      // We expect the following to replace the params file atomically in case we are using
      // the dynamic scheduler and we are racing the remote strategy writing this same file.
      tmpPath.renameTo(outputPath);
      tmpPath = null; // Avoid unnecessary deletion attempt.
      return digest;
    } finally {
      try {
        if (tmpPath != null) {
          // Make sure we don't leave temp files behind if we are interrupted.
          tmpPath.delete();
        }
      } catch (IOException e) {
        // Ignore.
      }
    }
  }

  @CanIgnoreReturnValue
  protected byte[] writeTo(Path target) throws IOException {
    byte[] digest;

    FileSystem fs = target.getFileSystem();
    try (OutputStream out = target.getOutputStream();
        HashingOutputStream hashingOut =
            new HashingOutputStream(fs.getDigestFunction().getHashFunction(), out)) {
      writeTo(hashingOut);
      digest = hashingOut.hash().asBytes();
    }
    // Some of the virtual inputs can be executed, e.g. embedded tools. Setting executable flag for
    // other is fine since that is only more permissive. Please note that for action outputs (e.g.
    // file write, where the user can specify executable flag), we will have artifacts which do not
    // go through this code path.
    target.setExecutable(true);
    return digest;
  }

  /**
   * Gets a {@link ByteString} representation of the fake file. Used to avoid copying if the fake
   * file is internally represented as a {@link ByteString}.
   */
  public abstract ByteString getBytes() throws IOException;

  /**
   * Returns the metadata for this input if available. Null otherwise.
   *
   * @throws IOException
   */
  public FileArtifactValue getMetadata() throws IOException {
    return null;
  }

  @Override
  public boolean isDirectory() {
    return false;
  }

  @Override
  public boolean isSymlink() {
    return false;
  }

  /**
   * In some cases, we want empty files in the runfiles tree that have no corresponding artifact. We
   * use instances of this class to represent those files.
   */
  public static final class EmptyActionInput extends VirtualActionInput {
    private static final byte[] emptyDigest = new byte[0];

    private EmptyActionInput() {}

    @Override
    public String getExecPathString() {
      throw new UnsupportedOperationException("empty virtual artifact doesn't have an execpath");
    }

    @Override
    public PathFragment getExecPath() {
      throw new UnsupportedOperationException("empty virtual artifact doesn't have an execpath");
    }

    @Override
    public byte[] atomicallyWriteRelativeTo(Path execRoot, String uniqueSuffix) {
      return emptyDigest;
    }

    @Override
    protected byte[] atomicallyWriteTo(Path outputPath, String uniqueSuffix) {
      return emptyDigest;
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      // Write no content - it's an empty file.
    }

    @Override
    public ByteString getBytes() throws IOException {
      return ByteString.EMPTY;
    }

    @Override
    public String toString() {
      return "EmptyActionInput";
    }
  }
}
