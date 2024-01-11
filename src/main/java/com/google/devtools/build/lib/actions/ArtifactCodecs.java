// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.devtools.build.lib.actions.Artifact.OMITTED_FOR_SERIALIZATION;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactSerializationContext;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Codec implementations for {@link Artifact} subclasses. */
final class ArtifactCodecs {

  @SuppressWarnings("unused") // Codec used by reflection.
  private static final class DerivedArtifactCodec implements ObjectCodec<DerivedArtifact> {

    @Override
    public Class<DerivedArtifact> getEncodedClass() {
      return DerivedArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, DerivedArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
      context.serialize(getGeneratingActionKeyForSerialization(obj, context), codedOut);
    }

    @Override
    public DerivedArtifact deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ArtifactRoot root = context.deserialize(codedIn);
      PathFragment rootRelativePath = context.deserialize(codedIn);
      Object generatingActionKey = context.deserialize(codedIn);
      DerivedArtifact artifact =
          new DerivedArtifact(
              root,
              getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
              generatingActionKey);
      return context.getDependency(ArtifactSerializationContext.class).intern(artifact);
    }
  }

  private static Object getGeneratingActionKeyForSerialization(
      DerivedArtifact artifact, SerializationContext context) {
    return context
            .getDependency(ArtifactSerializationContext.class)
            .includeGeneratingActionKey(artifact)
        ? artifact.getGeneratingActionKey()
        : OMITTED_FOR_SERIALIZATION;
  }

  private static PathFragment getExecPathForDeserialization(
      ArtifactRoot root, PathFragment rootRelativePath, Object generatingActionKey) {
    Preconditions.checkArgument(
        !root.isSourceRoot(),
        "Root not derived: %s (rootRelativePath=%s, generatingActionKey=%s)",
        root,
        rootRelativePath,
        generatingActionKey);
    Preconditions.checkArgument(
        root.getRoot().isAbsolute() == rootRelativePath.isAbsolute(),
        "Illegal root relative path: %s (root=%s, generatingActionKey=%s)",
        rootRelativePath,
        root,
        generatingActionKey);
    return root.getExecPath().getRelative(rootRelativePath);
  }

  /** {@link ObjectCodec} for {@link SourceArtifact} */
  @SuppressWarnings("unused") // Used by reflection.
  private static final class SourceArtifactCodec implements ObjectCodec<SourceArtifact> {

    @Override
    public Class<SourceArtifact> getEncodedClass() {
      return SourceArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, SourceArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getExecPath(), codedOut);
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getArtifactOwner(), codedOut);
    }

    @Override
    public SourceArtifact deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      PathFragment execPath = context.deserialize(codedIn);
      ArtifactRoot artifactRoot = context.deserialize(codedIn);
      ArtifactOwner owner = context.deserialize(codedIn);
      return context
          .getDependency(ArtifactSerializationContext.class)
          .getSourceArtifact(execPath, artifactRoot.getRoot(), owner);
    }
  }

  // Keep in sync with DerivedArtifactCodec.
  @SuppressWarnings("unused") // Used by reflection.
  private static final class SpecialArtifactCodec implements ObjectCodec<SpecialArtifact> {

    @Override
    public Class<SpecialArtifact> getEncodedClass() {
      return SpecialArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, SpecialArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
      context.serialize(getGeneratingActionKeyForSerialization(obj, context), codedOut);
      context.serialize(obj.getSpecialArtifactType(), codedOut);
    }

    @Override
    public SpecialArtifact deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ArtifactRoot root = context.deserialize(codedIn);
      PathFragment rootRelativePath = context.deserialize(codedIn);
      Object generatingActionKey = context.deserialize(codedIn);
      SpecialArtifactType type = context.deserialize(codedIn);
      SpecialArtifact artifact =
          new SpecialArtifact(
              root,
              getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
              generatingActionKey,
              type);
      return (SpecialArtifact)
          context.getDependency(ArtifactSerializationContext.class).intern(artifact);
    }
  }

  @SuppressWarnings("unused") // Codec used by reflection.
  private static final class ArchivedTreeArtifactCodec
      implements ObjectCodec<ArchivedTreeArtifact> {

    @Override
    public Class<ArchivedTreeArtifact> getEncodedClass() {
      return ArchivedTreeArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ArchivedTreeArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      PathFragment derivedTreeRoot = obj.getRoot().getExecPath().subFragment(1, 2);

      context.serialize(obj.getParent(), codedOut);
      context.serialize(derivedTreeRoot, codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
    }

    @Override
    public ArchivedTreeArtifact deserialize(
        DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      SpecialArtifact treeArtifact = context.deserialize(codedIn);
      PathFragment derivedTreeRoot = context.deserialize(codedIn);
      PathFragment rootRelativePath = context.deserialize(codedIn);
      Object generatingActionKey =
          treeArtifact.hasGeneratingActionKey()
              ? treeArtifact.getGeneratingActionKey()
              : OMITTED_FOR_SERIALIZATION;

      return ArchivedTreeArtifact.createInternal(
          treeArtifact, derivedTreeRoot, rootRelativePath, generatingActionKey);
    }
  }

  @SuppressWarnings("unused") // Used by reflection.
  private static final class TreeFileArtifactCodec implements ObjectCodec<TreeFileArtifact> {

    @Override
    public Class<TreeFileArtifact> getEncodedClass() {
      return TreeFileArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, TreeFileArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getParent(), codedOut);
      context.serialize(obj.getParentRelativePath(), codedOut);
      context.serialize(getGeneratingActionKeyForSerialization(obj, context), codedOut);
    }

    @Override
    public TreeFileArtifact deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      SpecialArtifact parent = context.deserialize(codedIn);
      PathFragment parentRelativePath = context.deserialize(codedIn);
      Object generatingActionKey = context.deserialize(codedIn);
      return new TreeFileArtifact(parent, parentRelativePath, generatingActionKey);
    }
  }

  private ArtifactCodecs() {}
}
