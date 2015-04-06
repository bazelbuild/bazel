// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import javax.annotation.Nullable;

/**
 * Cache provided by an {@link ActionExecutionFunction}, allowing Blaze to obtain artifact metadata
 * from the graph.
 *
 * <p>Data for the action's inputs is injected into this cache on construction, using the graph as
 * the source of truth.
 */
class PerActionFileCache implements ActionInputFileCache {
  private final Map<Artifact, FileArtifactValue> inputArtifactData;
  // Populated lazily, on calls to #getDigest.
  private final Map<ByteString, Artifact> reverseMap = new ConcurrentHashMap<>();

  private static final Interner<ByteString> BYTE_INTERNER = Interners.newWeakInterner();

  /**
   * @param inputArtifactData Map from artifact to metadata, used to return metadata upon request.
   */
  PerActionFileCache(Map<Artifact, FileArtifactValue> inputArtifactData) {
    this.inputArtifactData = Preconditions.checkNotNull(inputArtifactData);
  }

  @Nullable
  private FileArtifactValue getInputFileArtifactValue(ActionInput input) {
    if (!(input instanceof Artifact)) {
      return null;
    }
    return Preconditions.checkNotNull(inputArtifactData.get(input), input);
  }

  @Override
  public long getSizeInBytes(ActionInput input) throws IOException {
    FileArtifactValue metadata = getInputFileArtifactValue(input);
    if (metadata != null) {
      return metadata.getSize();
    }
    return -1;
  }

  @Nullable
  @Override
  public Artifact getInputFromDigest(ByteString digest) throws IOException {
    return reverseMap.get(digest);
  }

  @Override
  public Path getInputPath(ActionInput input) {
    return ((Artifact) input).getPath();
  }

  @Nullable
  @Override
  public ByteString getDigest(ActionInput input) throws IOException {
    FileArtifactValue value = getInputFileArtifactValue(input);
    if (value != null) {
      byte[] bytes = value.getDigest();
      if (bytes != null) {
        ByteString digest = ByteString.copyFrom(BaseEncoding.base16().lowerCase().encode(bytes)
            .getBytes(StandardCharsets.US_ASCII));
        reverseMap.put(BYTE_INTERNER.intern(digest), (Artifact) input);
        return digest;
      }
    }
    return null;
  }

  @Override
  public boolean contentsAvailableLocally(ByteString digest) {
    return reverseMap.containsKey(digest);
  }
}
