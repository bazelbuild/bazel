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

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.util.StreamWriter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;

/**
 * An ActionInput that does not actually exist on the filesystem, but can still be written to an
 * OutputStream.
 */
public interface VirtualActionInput extends ActionInput, StreamWriter {
  /**
   * An empty virtual artifact <b>without</b> an execpath. This is used to denote empty files in
   * runfiles and filesets.
   */
  public static final VirtualActionInput EMPTY_MARKER = new EmptyActionInput();

  /**
   * Gets a {@link ByteString} representation of the fake file. Used to avoid copying if the fake
   * file is internally represented as a {@link ByteString}.
   */
  ByteString getBytes() throws IOException;

  /**
   * Returns the metadata for this input if available. Null otherwise.
   *
   * @throws IOException
   */
  default FileArtifactValue getMetadata() throws IOException {
    return null;
  }

  /**
   * In some cases, we want empty files in the runfiles tree that have no corresponding artifact. We
   * use instances of this class to represent those files.
   */
  final class EmptyActionInput implements VirtualActionInput {
    private EmptyActionInput() {}

    @Override
    public boolean isSymlink() {
      return false;
    }

    @Override
    public String getExecPathString() {
      throw new UnsupportedOperationException("empty virtual artifact doesn't have an execpath");
    }

    @Override
    public PathFragment getExecPath() {
      throw new UnsupportedOperationException("empty virtual artifact doesn't have an execpath");
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
