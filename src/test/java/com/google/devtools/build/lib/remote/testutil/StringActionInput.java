// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.testutil;

import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/** A virtual action input backed by a string */
public final class StringActionInput implements VirtualActionInput {
  private final String contents;
  private final PathFragment execPath;

  public StringActionInput(String contents, PathFragment execPath) {
    this.contents = contents;
    this.execPath = execPath;
  }

  @Override
  public void writeTo(OutputStream out) throws IOException {
    out.write(contents.getBytes(StandardCharsets.UTF_8));
  }

  @Override
  public ByteString getBytes() throws IOException {
    ByteString.Output out = ByteString.newOutput();
    writeTo(out);
    return out.toByteString();
  }

  @Override
  public String getExecPathString() {
    return execPath.getPathString();
  }

  @Override
  public PathFragment getExecPath() {
    return execPath;
  }
}
