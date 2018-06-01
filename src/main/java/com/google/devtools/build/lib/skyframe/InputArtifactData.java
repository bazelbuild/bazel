// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.util.HashMap;
import javax.annotation.Nullable;

/** A mapping from artifacts to metadata. */
interface InputArtifactData {

  boolean contains(ActionInput input);

  @Nullable
  FileArtifactValue get(ActionInput input);

  @Nullable
  FileArtifactValue get(PathFragment fragment);

  /**
   * This implementation has a privileged {@link put} method supporting mutations.
   *
   * <p>Action execution has distinct phases where this data can be read from multiple threads. It's
   * important that the underlying data is not modified during those phases.
   */
  final class MutableInputArtifactData implements InputArtifactData {
    private final HashMap<PathFragment, FileArtifactValue> inputs;

    public MutableInputArtifactData(int sizeHint) {
      this.inputs = new HashMap<>(sizeHint);
    }

    @Override
    public boolean contains(ActionInput input) {
      return inputs.containsKey(input.getExecPath());
    }

    @Override
    @Nullable
    public FileArtifactValue get(ActionInput input) {
      return inputs.get(input.getExecPath());
    }

    @Override
    @Nullable
    public FileArtifactValue get(PathFragment fragment) {
      return inputs.get(fragment);
    }

    public void put(Artifact artifact, FileArtifactValue value) {
      inputs.put(artifact.getExecPath(), value);
    }

    @Override
    public String toString() {
      return inputs.toString();
    }

    private static ByteString toByteString(byte[] digest) {
      return ByteString.copyFrom(
          BaseEncoding.base16().lowerCase().encode(digest).getBytes(US_ASCII));
    }
  }
}
