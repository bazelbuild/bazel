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

package com.google.devtools.build.lib.sandbox;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Manifest;
import com.google.protobuf.ByteString;
import com.google.protobuf.TextFormat;
import java.util.Map;

/**
 * Helpers for the {@code sandbox-backend} {@link Manifest}.
 *
 * <p>The input tree is hashed by {@link
 * com.google.devtools.build.lib.remote.merkletree.MerkleTreeComputer} (shared with remote
 * execution), which yields a content-addressed blob set already serialized as {@code byte[]}.
 */
final class SandboxBackendManifest {
  private SandboxBackendManifest() {}

  /** Renders a manifest as protobuf text format, for the {@code --sandbox_debug} dump. */
  static String toDebugString(Manifest manifest) {
    return TextFormat.printer().printToString(manifest);
  }

  /**
   * The tree's directory blobs keyed by digest hash. Each value is a serialized {@code
   * build.bazel.remote.execution.v2.Directory}, shipped verbatim: the hasher already produced these
   * bytes and their hash is their digest, so there is nothing to parse or re-serialize. The consumer
   * walks from {@code input_root_digest}, resolving each {@code DirectoryNode.digest} back through
   * this map.
   *
   * <p>In a built tree the only {@code byte[]} blobs are directories — file and virtual-input leaves
   * are retained as {@link com.google.devtools.build.lib.actions.ActionInput}s and travel as {@code
   * Push.content} (or are read from their default exec_root path), not here — so this is exactly the
   * set of directories in the tree.
   */
  static ImmutableMap<String, ByteString> directoryBlobs(MerkleTree.Uploadable built) {
    ImmutableMap.Builder<String, ByteString> blobs = ImmutableMap.builder();
    for (Map.Entry<Digest, Object> entry : built.blobs().entrySet()) {
      if (entry.getValue() instanceof byte[] bytes) {
        blobs.put(entry.getKey().getHash(), ByteString.copyFrom(bytes));
      }
    }
    return blobs.buildKeepingLast();
  }
}
