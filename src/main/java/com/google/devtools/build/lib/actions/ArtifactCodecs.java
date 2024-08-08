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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactSerializationContext;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * Codec implementations for {@link Artifact} subclasses.
 *
 * <p>Each Artifact's codec implementation is split into two codecs: the main codec that handles the
 * individual fields, and a value-sharing codec.
 */
final class ArtifactCodecs {

  public static final ImmutableList<ObjectCodec<? extends Artifact>> VALUE_SHARING_CODECS =
      ImmutableList.of(
          new DerivedArtifactValueSharingCodec(),
          new SourceArtifactValueSharingCodec(),
          new SpecialArtifactValueSharingCodec());

  @Keep
  private static final class DerivedArtifactValueSharingCodec
      extends DeferredObjectCodec<DerivedArtifact> {

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public Class<DerivedArtifact> getEncodedClass() {
      return DerivedArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, DerivedArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          obj, /* distinguisher= */ null, DerivedArtifactCodec.INSTANCE, codedOut);
    }

    @Override
    public DeferredValue<DerivedArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      SimpleDeferredValue<DerivedArtifact> value = SimpleDeferredValue.create();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          DerivedArtifactCodec.INSTANCE,
          value,
          SimpleDeferredValue::set);
      return value;
    }
  }

  /**
   * {@link ObjectCodec} for {@link DerivedArtifact}.
   *
   * <p>To be kept in sync with {@link SpecialArtifactCodec}.
   */
  @Keep // Used by reflection.
  private static final class DerivedArtifactCodec extends DeferredObjectCodec<DerivedArtifact> {

    private static final DerivedArtifactCodec INSTANCE = new DerivedArtifactCodec();

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
    public DeferredValue<DerivedArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedDerivedArtifactBuilder builder =
          new DeserializedDerivedArtifactBuilder(
              context.getDependency(ArtifactSerializationContext.class));
      context.deserialize(codedIn, builder, DeserializedDerivedArtifactBuilder::setRoot);
      context.deserialize(
          codedIn, builder, DeserializedDerivedArtifactBuilder::setRootRelativePath);
      context.deserialize(
          codedIn, builder, DeserializedDerivedArtifactBuilder::setGeneratingActionKey);
      return builder;
    }
  }

  private static class DeserializedDerivedArtifactBuilder
      implements DeferredValue<DerivedArtifact> {
    private final ArtifactSerializationContext context;
    private ArtifactRoot root;
    private PathFragment rootRelativePath;
    private Object generatingActionKey;

    private DeserializedDerivedArtifactBuilder(ArtifactSerializationContext context) {
      this.context = context;
    }

    @Override
    public DerivedArtifact call() {
      return context.intern(
          new DerivedArtifact(
              root,
              getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
              generatingActionKey));
    }

    private static void setRoot(DeserializedDerivedArtifactBuilder builder, Object value) {
      builder.root = (ArtifactRoot) value;
    }

    private static void setRootRelativePath(
        DeserializedDerivedArtifactBuilder builder, Object value) {
      builder.rootRelativePath = (PathFragment) value;
    }

    private static void setGeneratingActionKey(
        DeserializedDerivedArtifactBuilder builder, Object value) {
      builder.generatingActionKey = value;
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

  @Keep
  private static final class SourceArtifactValueSharingCodec
      extends DeferredObjectCodec<SourceArtifact> {

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public Class<SourceArtifact> getEncodedClass() {
      return SourceArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, SourceArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          obj, /* distinguisher= */ null, SourceArtifactCodec.INSTANCE, codedOut);
    }

    @Override
    public DeferredValue<SourceArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      SimpleDeferredValue<SourceArtifact> value = SimpleDeferredValue.create();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          SourceArtifactCodec.INSTANCE,
          value,
          SimpleDeferredValue::set);
      return value;
    }
  }

  /** {@link ObjectCodec} for {@link SourceArtifact} */
  @Keep // Used by reflection.
  private static final class SourceArtifactCodec extends DeferredObjectCodec<SourceArtifact> {

    private static final SourceArtifactCodec INSTANCE = new SourceArtifactCodec();

    @Override
    public Class<SourceArtifact> getEncodedClass() {
      return SourceArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, SourceArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getExecPath(), codedOut);
      context.serialize(obj.getRoot().getRoot(), codedOut);
      context.serialize(obj.getArtifactOwner(), codedOut);
    }

    @Override
    public DeferredValue<SourceArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedSourceArtifactBuilder builder =
          new DeserializedSourceArtifactBuilder(
              context.getDependency(ArtifactSerializationContext.class));
      context.deserialize(codedIn, builder, DeserializedSourceArtifactBuilder::setExecPath);
      context.deserialize(codedIn, builder, DeserializedSourceArtifactBuilder::setRoot);
      context.deserialize(codedIn, builder, DeserializedSourceArtifactBuilder::setOwner);
      return builder;
    }
  }

  private static class DeserializedSourceArtifactBuilder implements DeferredValue<SourceArtifact> {
    private final ArtifactSerializationContext context;
    private PathFragment execPath;
    private Root root;
    private ArtifactOwner owner;

    private DeserializedSourceArtifactBuilder(ArtifactSerializationContext context) {
      this.context = context;
    }

    @Override
    public SourceArtifact call() {
      return context.getSourceArtifact(execPath, root, owner);
    }

    private static void setExecPath(DeserializedSourceArtifactBuilder builder, Object value) {
      builder.execPath = (PathFragment) value;
    }

    private static void setRoot(DeserializedSourceArtifactBuilder builder, Object value) {
      builder.root = (Root) value;
    }

    private static void setOwner(DeserializedSourceArtifactBuilder builder, Object value) {
      builder.owner = (ArtifactOwner) value;
    }
  }

  @Keep
  private static final class SpecialArtifactValueSharingCodec
      extends DeferredObjectCodec<SpecialArtifact> {

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public Class<SpecialArtifact> getEncodedClass() {
      return SpecialArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, SpecialArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          obj, /* distinguisher= */ null, SpecialArtifactCodec.INSTANCE, codedOut);
    }

    @Override
    public DeferredValue<SpecialArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      SimpleDeferredValue<SpecialArtifact> value = SimpleDeferredValue.create();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          SpecialArtifactCodec.INSTANCE,
          value,
          SimpleDeferredValue::set);
      return value;
    }
  }

  /**
   * {@link ObjectCodec} for {@link SpecialArtifact}.
   *
   * <p>To be kept in sync with {@link DerivedArtifactCodec}.
   */
  @Keep // Used by reflection.
  private static final class SpecialArtifactCodec extends DeferredObjectCodec<SpecialArtifact> {

    private static final SpecialArtifactCodec INSTANCE = new SpecialArtifactCodec();

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
    public DeferredValue<SpecialArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedSpecialArtifactBuilder builder =
          new DeserializedSpecialArtifactBuilder(
              context.getDependency(ArtifactSerializationContext.class));
      context.deserialize(codedIn, builder, DeserializedSpecialArtifactBuilder::setRoot);
      context.deserialize(
          codedIn, builder, DeserializedSpecialArtifactBuilder::setRootRelativePath);
      context.deserialize(
          codedIn, builder, DeserializedSpecialArtifactBuilder::setGeneratingActionKey);
      context.deserialize(codedIn, builder, DeserializedSpecialArtifactBuilder::setType);
      return builder;
    }
  }

  private static class DeserializedSpecialArtifactBuilder
      implements DeferredValue<SpecialArtifact> {
    private final ArtifactSerializationContext context;
    private ArtifactRoot root;
    private PathFragment rootRelativePath;
    private Object generatingActionKey;
    private SpecialArtifactType type;

    private DeserializedSpecialArtifactBuilder(ArtifactSerializationContext context) {
      this.context = context;
    }

    @Override
    public SpecialArtifact call() {
      return (SpecialArtifact)
          context.intern(
              new SpecialArtifact(
                  root,
                  getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
                  generatingActionKey,
                  type));
    }

    private static void setRoot(DeserializedSpecialArtifactBuilder builder, Object value) {
      builder.root = (ArtifactRoot) value;
    }

    private static void setRootRelativePath(
        DeserializedSpecialArtifactBuilder builder, Object value) {
      builder.rootRelativePath = (PathFragment) value;
    }

    private static void setGeneratingActionKey(
        DeserializedSpecialArtifactBuilder builder, Object value) {
      builder.generatingActionKey = value;
    }

    private static void setType(DeserializedSpecialArtifactBuilder builder, Object value) {
      builder.type = (SpecialArtifactType) value;
    }
  }

  @SuppressWarnings("unused") // Codec used by reflection.
  private static final class ArchivedTreeArtifactCodec
      extends DeferredObjectCodec<ArchivedTreeArtifact> {

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
    public DeferredValue<ArchivedTreeArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedArchivedTreeArtifactBuilder builder =
          new DeserializedArchivedTreeArtifactBuilder();
      context.deserialize(
          codedIn, builder, DeserializedArchivedTreeArtifactBuilder::setTreeArtifact);
      context.deserialize(
          codedIn, builder, DeserializedArchivedTreeArtifactBuilder::setDerivedTreeRoot);
      context.deserialize(
          codedIn, builder, DeserializedArchivedTreeArtifactBuilder::setRootRelativePath);
      return builder;
    }
  }

  private static class DeserializedArchivedTreeArtifactBuilder
      implements DeferredValue<ArchivedTreeArtifact> {
    private SpecialArtifact treeArtifact;
    private PathFragment derivedTreeRoot;
    private PathFragment rootRelativePath;

    @Override
    public ArchivedTreeArtifact call() {
      Object generatingActionKey =
          treeArtifact.hasGeneratingActionKey()
              ? treeArtifact.getGeneratingActionKey()
              : OMITTED_FOR_SERIALIZATION;
      return ArchivedTreeArtifact.createInternal(
          treeArtifact, derivedTreeRoot, rootRelativePath, generatingActionKey);
    }

    private static void setTreeArtifact(
        DeserializedArchivedTreeArtifactBuilder builder, Object value) {
      builder.treeArtifact = (SpecialArtifact) value;
    }

    private static void setDerivedTreeRoot(
        DeserializedArchivedTreeArtifactBuilder builder, Object value) {
      builder.derivedTreeRoot = (PathFragment) value;
    }

    private static void setRootRelativePath(
        DeserializedArchivedTreeArtifactBuilder builder, Object value) {
      builder.rootRelativePath = (PathFragment) value;
    }
  }

  @SuppressWarnings("unused") // Used by reflection.
  private static final class TreeFileArtifactCodec extends DeferredObjectCodec<TreeFileArtifact> {

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
    public DeferredValue<TreeFileArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedTreeFileArtifactBuilder builder = new DeserializedTreeFileArtifactBuilder();
      context.deserialize(codedIn, builder, DeserializedTreeFileArtifactBuilder::setParent);
      context.deserialize(
          codedIn, builder, DeserializedTreeFileArtifactBuilder::setParentRelativePath);
      context.deserialize(
          codedIn, builder, DeserializedTreeFileArtifactBuilder::setGeneratingActionKey);
      return builder;
    }
  }

  private static class DeserializedTreeFileArtifactBuilder
      implements DeferredValue<TreeFileArtifact> {
    private SpecialArtifact parent;
    private PathFragment parentRelativePath;
    private Object generatingActionKey;

    @Override
    public TreeFileArtifact call() {
      return new TreeFileArtifact(parent, parentRelativePath, generatingActionKey);
    }

    private static void setParent(DeserializedTreeFileArtifactBuilder builder, Object value) {
      builder.parent = (SpecialArtifact) value;
    }

    private static void setParentRelativePath(
        DeserializedTreeFileArtifactBuilder builder, Object value) {
      builder.parentRelativePath = (PathFragment) value;
    }

    private static void setGeneratingActionKey(
        DeserializedTreeFileArtifactBuilder builder, Object value) {
      builder.generatingActionKey = value;
    }
  }

  private ArtifactCodecs() {}
}
