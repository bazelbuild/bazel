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

package com.google.devtools.build.lib.remote.merkletree;

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.NodeProperties;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;
import javax.annotation.Nullable;

interface DirectoryBuilder {
  void addFile(String name, Digest digest, @Nullable NodeProperties nodeProperties);

  void addFile(ActionInput file, Digest digest, @Nullable NodeProperties nodeProperties);

  void addFile(
      ActionInput file, FileArtifactValue metadata, @Nullable NodeProperties nodeProperties);

  void addSymlink(
      Artifact.SpecialArtifact symlink,
      FileArtifactValue metadata,
      @Nullable NodeProperties nodeProperties);

  void addDirectory(Artifact subTreeRoot, MerkleTree subTree);

  void addDirectory(String name, Digest digest);

  Object build();

  static DirectoryBuilder create(MerkleTreeComputer.BlobPolicy blobPolicy) {
    return new MessageDirectoryBuilder();
  }

  static void writeTo(OutputStream out, Object directory, Map<Object, Object> blobs)
      throws IOException {
    if (directory instanceof byte[] bytes) {
      out.write(bytes);
    } else {
      throw new IllegalArgumentException(
          "Expected Directory message, got: " + directory.getClass().getName());
    }
  }

  class MessageDirectoryBuilder implements DirectoryBuilder {
    private final Directory.Builder dirBuilder = Directory.newBuilder();

    @Override
    public void addFile(String name, Digest digest, @Nullable NodeProperties nodeProperties) {
      var fileBuilder =
          dirBuilder
              .addFilesBuilder()
              .setName(internalToUnicode(name))
              .setDigest(digest)
              // We always treat files as executable since Bazel will `chmod 555` on the output
              // files of an action within ActionOutputMetadataStore#getMetadata after action
              // execution if no metadata was injected. We can't use real executable bit of the
              // file until this behavior is changed. See
              // https://github.com/bazelbuild/bazel/issues/13262 for more details.
              .setIsExecutable(true);
      if (nodeProperties != null) {
        fileBuilder.setNodeProperties(nodeProperties);
      }
    }

    @Override
    public void addFile(ActionInput file, Digest digest, @Nullable NodeProperties nodeProperties) {
      addFile(file.getExecPath().getBaseName(), digest, nodeProperties);
    }

    @Override
    public void addFile(
        ActionInput file, FileArtifactValue metadata, @Nullable NodeProperties nodeProperties) {
      addFile(
          file, DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize()), nodeProperties);
    }

    @Override
    public void addSymlink(
        Artifact.SpecialArtifact symlink,
        FileArtifactValue metadata,
        @Nullable NodeProperties nodeProperties) {
      var symlinkBuilder =
          dirBuilder
              .addSymlinksBuilder()
              .setName(internalToUnicode(symlink.getFilename()))
              .setTarget(internalToUnicode(metadata.getUnresolvedSymlinkTarget()));
      if (nodeProperties != null) {
        symlinkBuilder.setNodeProperties(nodeProperties);
      }
    }

    @Override
    public void addDirectory(Artifact subTreeRoot, MerkleTree subtree) {
      dirBuilder
          .addDirectoriesBuilder()
          .setName(internalToUnicode(subTreeRoot.getFilename()))
          .setDigest(subtree.digest());
    }

    @Override
    public void addDirectory(String name, Digest digest) {
      dirBuilder.addDirectoriesBuilder().setName(internalToUnicode(name)).setDigest(digest);
    }

    @Override
    public Object build() {
      return dirBuilder.build().toByteArray();
    }
  }
}
