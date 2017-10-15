// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;

/**
 * In some cases, we want empty files in the runfiles tree that have no corresponding artifact. We
 * use instances of this class to represent those files.
 */
final class EmptyActionInput implements VirtualActionInput {
  private final PathFragment execPath;

  public EmptyActionInput(PathFragment execPath) {
    this.execPath = Preconditions.checkNotNull(execPath);
  }

  @Override
  public String getExecPathString() {
    return execPath.getPathString();
  }

  @Override
  public PathFragment getExecPath() {
    return execPath;
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
    return "EmptyActionInput: " + execPath;
  }
}
