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

import static com.google.devtools.build.lib.remote.merkletree.MerkleTreeComputer.concat;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.NodeProperties;
import build.bazel.remote.execution.v2.SymlinkNode;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Objects;
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
    return switch (blobPolicy) {
      case DISCARD -> new MessageDirectoryBuilder();
      case KEEP, KEEP_AND_REUPLOAD -> new CompactDirectoryBuilder();
    };
  }

  static void writeTo(OutputStream out, Object directory) throws IOException {
    switch (directory) {
      case byte[] bytes -> MessageDirectoryBuilder.writeTo(out, bytes);
      case Object[] objects -> CompactDirectoryBuilder.writeTo(out, objects);
      default ->
          throw new IllegalArgumentException(
              "Unknown directory representation: " + directory.getClass());
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
    public byte[] build() {
      return dirBuilder.build().toByteArray();
    }

    private static void writeTo(OutputStream out, byte[] bytes) throws IOException {
      out.write(bytes);
    }
  }

  class CompactDirectoryBuilder implements DirectoryBuilder {
    private final ArrayList<Object> files = new ArrayList<>();
    @Nullable private NodeProperties fileNodeProperties;
    private final ArrayList<Object> symlinks = new ArrayList<>();
    @Nullable private NodeProperties symlinkNodeProperties;
    private final ArrayList<Object> directories = new ArrayList<>();

    @Override
    public void addFile(String name, Digest digest, @Nullable NodeProperties nodeProperties) {
      maybeUpdateFileNodeProperties(nodeProperties);
      files.add(name);
      files.add(digest);
    }

    @Override
    public void addFile(ActionInput file, Digest digest, @Nullable NodeProperties nodeProperties) {
      maybeUpdateFileNodeProperties(nodeProperties);
      files.add(file);
      files.add(digest);
    }

    @Override
    public void addFile(
        ActionInput file, FileArtifactValue metadata, @Nullable NodeProperties nodeProperties) {
      maybeUpdateFileNodeProperties(nodeProperties);
      files.add(file);
      files.add(metadata);
    }

    private void maybeUpdateFileNodeProperties(@Nullable NodeProperties newProperties) {
      if (!Objects.equals(newProperties, fileNodeProperties)) {
        files.add(newProperties);
        fileNodeProperties = newProperties;
      }
    }

    @Override
    public void addSymlink(
        Artifact.SpecialArtifact symlink,
        FileArtifactValue metadata,
        @Nullable NodeProperties nodeProperties) {
      maybeUpdateSymlinkNodeProperties(nodeProperties);
      symlinks.add(symlink);
      symlinks.add(metadata);
    }

    private void maybeUpdateSymlinkNodeProperties(@Nullable NodeProperties nodeProperties) {
      if (!Objects.equals(nodeProperties, symlinkNodeProperties)) {
        symlinks.add(nodeProperties);
        symlinkNodeProperties = nodeProperties;
      }
    }

    @Override
    public void addDirectory(Artifact subTreeRoot, MerkleTree subTree) {
      directories.add(subTree);
      directories.add(subTreeRoot);
    }

    @Override
    public void addDirectory(String name, Digest digest) {
      directories.add(name);
      directories.add(digest);
    }

    @Override
    public Object[] build() {
      maybeUpdateFileNodeProperties(null);
      maybeUpdateSymlinkNodeProperties(null);
      return concat(concat(files, directories), symlinks).toArray();
    }

    static void writeTo(OutputStream out, Object[] directory) throws IOException {
      var codedOut = CodedOutputStream.newInstance(out);
      NodeProperties nodeProperties = null;
      for (int i = 0; i < directory.length; i++) {
        var item = directory[i];
        switch (item) {
          case null -> nodeProperties = null;
          case NodeProperties newNodeProperties -> nodeProperties = newNodeProperties;
          case ActionInput symlink when symlink.isSymlink() -> {
            var metadata = (FileArtifactValue) directory[++i];
            codedOut.writeMessage(
                3,
                SymlinkNode.newBuilder()
                    .setName(internalToUnicode(symlink.getExecPath().getBaseName()))
                    .setTarget(internalToUnicode(metadata.getUnresolvedSymlinkTarget()))
                    .build());
          }
          case ActionInput input -> {
            var metadataObj = directory[++i];
            var digest =
                switch (metadataObj) {
                  case Digest d -> d;
                  case FileArtifactValue metadata ->
                      DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
                  default ->
                      throw new IllegalStateException("Unexpected item type: " + item.getClass());
                };
            var fileNode =
                FileNode.newBuilder()
                    .setName(internalToUnicode(input.getExecPath().getBaseName()))
                    .setDigest(digest)
                    // We always treat files as executable since Bazel will `chmod 555` on the
                    // output files of an action within ActionOutputMetadataStore#getMetadata after
                    // action execution if no metadata was injected. We can't use real executable
                    // bit of the file until this behavior is changed. See
                    // https://github.com/bazelbuild/bazel/issues/13262 for more details.
                    .setIsExecutable(true);
            if (nodeProperties != null) {
              fileNode.setNodeProperties(nodeProperties);
            }
            codedOut.writeMessage(1, fileNode.build());
          }
          case MerkleTree subTree -> {
            var subTreeRoot = (ActionInput) directory[++i];
            codedOut.writeMessage(
                2,
                DirectoryNode.newBuilder()
                    .setName(internalToUnicode(subTreeRoot.getExecPath().getBaseName()))
                    .setDigest(subTree.digest())
                    .build());
          }
          case String name -> {
            var digest = (Digest) directory[++i];
            codedOut.writeMessage(
                2,
                DirectoryNode.newBuilder()
                    .setName(internalToUnicode(name))
                    .setDigest(digest)
                    .build());
          }
          default -> throw new IllegalStateException("Unexpected value: " + item);
        }
      }
      codedOut.flush();
    }
  }
}
