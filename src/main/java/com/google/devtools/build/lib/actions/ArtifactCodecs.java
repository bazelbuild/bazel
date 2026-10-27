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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactSerializationContext;
import com.google.devtools.build.lib.actions.Artifact.ConstantMetadataArtifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.FilesetArtifact;
import com.google.devtools.build.lib.actions.Artifact.RunfilesArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.Artifact.UnresolvedSymlinkArtifact;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext.FieldSetter;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.ValueSharingAdapter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Codec implementations for {@link Artifact} subclasses.
 *
 * <p>Each Artifact's codec implementation is split into two codecs: the main codec that handles the
 * individual fields, and a value-sharing codec.
 */
public final class ArtifactCodecs {

  public static final ImmutableList<ObjectCodec<? extends Artifact>> VALUE_SHARING_CODECS =
      ImmutableList.of(
          new ValueSharingAdapter<>(DerivedArtifactCodec.INSTANCE),
          new ValueSharingAdapter<>(SourceArtifactCodec.INSTANCE),
          new ValueSharingAdapter<>(TreeArtifactCodec.INSTANCE),
          new ValueSharingAdapter<>(FilesetArtifactCodec.INSTANCE),
          new ValueSharingAdapter<>(RunfilesArtifactCodec.INSTANCE),
          new ValueSharingAdapter<>(UnresolvedSymlinkArtifactCodec.INSTANCE),
          new ValueSharingAdapter<>(ConstantMetadataArtifactCodec.INSTANCE));

  /**
   * {@link ObjectCodec} for {@link DerivedArtifact}.
   *
   * <p>To be kept in sync with {@link TreeArtifactCodec}, {@link FilesetArtifactCodec}, {@link
   * RunfilesArtifactCodec}, {@link UnresolvedSymlinkArtifactCodec}, and {@link
   * ConstantMetadataArtifactCodec}.
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
      serializeOrOmitGeneratingActionKey(context, obj, codedOut);
    }

    @Override
    public DeferredValue<DerivedArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedDerivedArtifactBuilder builder = new DeserializedDerivedArtifactBuilder(context);
      context.deserialize(codedIn, builder, DeserializedDerivedArtifactBuilder::setRoot);
      context.deserialize(
          codedIn, builder, DeserializedDerivedArtifactBuilder::setRootRelativePath);
      deserializeOrGetGeneratingActionKey(
          context, codedIn, builder, DeserializedDerivedArtifactBuilder::setGeneratingActionKey);
      return builder;
    }
  }

  private static class DeserializedDerivedArtifactBuilder
      implements DeferredValue<DerivedArtifact> {
    private final AsyncDeserializationContext context;
    private ArtifactRoot root;
    private PathFragment rootRelativePath;
    private ActionLookupData generatingActionKey;

    private DeserializedDerivedArtifactBuilder(AsyncDeserializationContext context) {
      this.context = context;
    }

    @Override
    public DerivedArtifact call() {
      return context
          .getDependency(ArtifactSerializationContext.class)
          .intern(
              new DerivedArtifact(
                  root,
                  getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
                  generatingActionKey),
              context);
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
      builder.generatingActionKey = (ActionLookupData) value;
    }
  }

  private static void serializeOrOmitGeneratingActionKey(
      SerializationContext context, DerivedArtifact obj, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    boolean include =
        context
            .getDependency(ArtifactSerializationContext.class)
            .includeGeneratingActionKey(obj, context);
    codedOut.writeBoolNoTag(include);
    if (include) {
      context.serialize(obj.getGeneratingActionKey(), codedOut);
    }
  }

  private static <T> void deserializeOrGetGeneratingActionKey(
      AsyncDeserializationContext context,
      CodedInputStream codedIn,
      T builder,
      FieldSetter<T> setter)
      throws IOException, SerializationException {
    boolean included = codedIn.readBool();
    if (included) {
      context.deserialize(codedIn, builder, setter);
    } else {
      ActionLookupData generatingActionKey =
          context
              .getDependency(ArtifactSerializationContext.class)
              .getOmittedGeneratingActionKey(context);
      setter.set(builder, generatingActionKey);
    }
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
      context.serialize(obj.getRoot(), codedOut);
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
    private ArtifactRoot root;
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
      builder.root = (ArtifactRoot) value;
    }

    private static void setOwner(DeserializedSourceArtifactBuilder builder, Object value) {
      builder.owner = (ArtifactOwner) value;
    }
  }

  /**
   * {@link ObjectCodec} for {@link TreeArtifact}.
   *
   * <p>To be kept in sync with {@link DerivedArtifactCodec}.
   */
  @Keep // Used by reflection.
  private static final class TreeArtifactCodec extends DeferredObjectCodec<TreeArtifact> {

    private static final TreeArtifactCodec INSTANCE = new TreeArtifactCodec();

    @Override
    public Class<TreeArtifact> getEncodedClass() {
      return TreeArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, TreeArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
      serializeOrOmitGeneratingActionKey(context, obj, codedOut);
      context.serialize(obj.getParent(), codedOut);
      context.serialize(obj.getParentRelativePath(), codedOut);
    }

    @Override
    public DeferredValue<TreeArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedTreeArtifactBuilder builder = new DeserializedTreeArtifactBuilder(context);
      context.deserialize(codedIn, builder, DeserializedTreeArtifactBuilder::setRoot);
      context.deserialize(codedIn, builder, DeserializedTreeArtifactBuilder::setRootRelativePath);
      deserializeOrGetGeneratingActionKey(
          context, codedIn, builder, DeserializedTreeArtifactBuilder::setGeneratingActionKey);
      context.deserialize(codedIn, builder, DeserializedTreeArtifactBuilder::setParent);
      context.deserialize(codedIn, builder, DeserializedTreeArtifactBuilder::setParentRelativePath);
      return builder;
    }
  }

  private static final class DeserializedTreeArtifactBuilder
      implements DeferredValue<TreeArtifact> {
    private final AsyncDeserializationContext context;
    private ArtifactRoot root;
    private PathFragment rootRelativePath;
    private ActionLookupData generatingActionKey;
    @Nullable private TreeArtifact parent;
    @Nullable private PathFragment parentRelativePath;

    private DeserializedTreeArtifactBuilder(AsyncDeserializationContext context) {
      this.context = context;
    }

    @Override
    public TreeArtifact call() {
      return (TreeArtifact)
          context
              .getDependency(ArtifactSerializationContext.class)
              .intern(
                  new TreeArtifact(
                      root,
                      getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
                      generatingActionKey,
                      parent,
                      parentRelativePath),
                  context);
    }

    private static void setRoot(DeserializedTreeArtifactBuilder builder, Object value) {
      builder.root = (ArtifactRoot) value;
    }

    private static void setRootRelativePath(DeserializedTreeArtifactBuilder builder, Object value) {
      builder.rootRelativePath = (PathFragment) value;
    }

    private static void setGeneratingActionKey(
        DeserializedTreeArtifactBuilder builder, Object value) {
      builder.generatingActionKey = (ActionLookupData) value;
    }

    private static void setParent(DeserializedTreeArtifactBuilder builder, Object value) {
      builder.parent = (TreeArtifact) value;
    }

    private static void setParentRelativePath(
        DeserializedTreeArtifactBuilder builder, Object value) {
      builder.parentRelativePath = (PathFragment) value;
    }
  }

  /**
   * {@link ObjectCodec} for {@link FilesetArtifact}.
   *
   * <p>To be kept in sync with {@link DerivedArtifactCodec}.
   */
  @Keep // Used by reflection.
  private static final class FilesetArtifactCodec extends DeferredObjectCodec<FilesetArtifact> {

    private static final FilesetArtifactCodec INSTANCE = new FilesetArtifactCodec();

    @Override
    public Class<FilesetArtifact> getEncodedClass() {
      return FilesetArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, FilesetArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
      serializeOrOmitGeneratingActionKey(context, obj, codedOut);
    }

    @Override
    public DeferredValue<FilesetArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedFilesetArtifactBuilder builder = new DeserializedFilesetArtifactBuilder(context);
      context.deserialize(codedIn, builder, DeserializedFilesetArtifactBuilder::setRoot);
      context.deserialize(
          codedIn, builder, DeserializedFilesetArtifactBuilder::setRootRelativePath);
      deserializeOrGetGeneratingActionKey(
          context, codedIn, builder, DeserializedFilesetArtifactBuilder::setGeneratingActionKey);
      return builder;
    }
  }

  private static final class DeserializedFilesetArtifactBuilder
      implements DeferredValue<FilesetArtifact> {
    private final AsyncDeserializationContext context;
    private ArtifactRoot root;
    private PathFragment rootRelativePath;
    private ActionLookupData generatingActionKey;

    private DeserializedFilesetArtifactBuilder(AsyncDeserializationContext context) {
      this.context = context;
    }

    @Override
    public FilesetArtifact call() {
      return (FilesetArtifact)
          context
              .getDependency(ArtifactSerializationContext.class)
              .intern(
                  new FilesetArtifact(
                      root,
                      getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
                      generatingActionKey),
                  context);
    }

    private static void setRoot(DeserializedFilesetArtifactBuilder builder, Object value) {
      builder.root = (ArtifactRoot) value;
    }

    private static void setRootRelativePath(
        DeserializedFilesetArtifactBuilder builder, Object value) {
      builder.rootRelativePath = (PathFragment) value;
    }

    private static void setGeneratingActionKey(
        DeserializedFilesetArtifactBuilder builder, Object value) {
      builder.generatingActionKey = (ActionLookupData) value;
    }
  }

  /**
   * {@link ObjectCodec} for {@link RunfilesArtifact}.
   *
   * <p>To be kept in sync with {@link DerivedArtifactCodec}.
   */
  @Keep // Used by reflection.
  private static final class RunfilesArtifactCodec extends DeferredObjectCodec<RunfilesArtifact> {

    private static final RunfilesArtifactCodec INSTANCE = new RunfilesArtifactCodec();

    @Override
    public Class<RunfilesArtifact> getEncodedClass() {
      return RunfilesArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, RunfilesArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
      serializeOrOmitGeneratingActionKey(context, obj, codedOut);
    }

    @Override
    public DeferredValue<RunfilesArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedRunfilesArtifactBuilder builder =
          new DeserializedRunfilesArtifactBuilder(context);
      context.deserialize(codedIn, builder, DeserializedRunfilesArtifactBuilder::setRoot);
      context.deserialize(
          codedIn, builder, DeserializedRunfilesArtifactBuilder::setRootRelativePath);
      deserializeOrGetGeneratingActionKey(
          context, codedIn, builder, DeserializedRunfilesArtifactBuilder::setGeneratingActionKey);
      return builder;
    }
  }

  private static final class DeserializedRunfilesArtifactBuilder
      implements DeferredValue<RunfilesArtifact> {
    private final AsyncDeserializationContext context;
    private ArtifactRoot root;
    private PathFragment rootRelativePath;
    private ActionLookupData generatingActionKey;

    private DeserializedRunfilesArtifactBuilder(AsyncDeserializationContext context) {
      this.context = context;
    }

    @Override
    public RunfilesArtifact call() {
      return (RunfilesArtifact)
          context
              .getDependency(ArtifactSerializationContext.class)
              .intern(
                  new RunfilesArtifact(
                      root,
                      getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
                      generatingActionKey),
                  context);
    }

    private static void setRoot(DeserializedRunfilesArtifactBuilder builder, Object value) {
      builder.root = (ArtifactRoot) value;
    }

    private static void setRootRelativePath(
        DeserializedRunfilesArtifactBuilder builder, Object value) {
      builder.rootRelativePath = (PathFragment) value;
    }

    private static void setGeneratingActionKey(
        DeserializedRunfilesArtifactBuilder builder, Object value) {
      builder.generatingActionKey = (ActionLookupData) value;
    }
  }

  /**
   * {@link ObjectCodec} for {@link UnresolvedSymlinkArtifact}.
   *
   * <p>To be kept in sync with {@link DerivedArtifactCodec}.
   */
  @Keep // Used by reflection.
  private static final class UnresolvedSymlinkArtifactCodec
      extends DeferredObjectCodec<UnresolvedSymlinkArtifact> {

    private static final UnresolvedSymlinkArtifactCodec INSTANCE =
        new UnresolvedSymlinkArtifactCodec();

    @Override
    public Class<UnresolvedSymlinkArtifact> getEncodedClass() {
      return UnresolvedSymlinkArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, UnresolvedSymlinkArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
      serializeOrOmitGeneratingActionKey(context, obj, codedOut);
    }

    @Override
    public DeferredValue<UnresolvedSymlinkArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedUnresolvedSymlinkArtifactBuilder builder =
          new DeserializedUnresolvedSymlinkArtifactBuilder(context);
      context.deserialize(codedIn, builder, DeserializedUnresolvedSymlinkArtifactBuilder::setRoot);
      context.deserialize(
          codedIn, builder, DeserializedUnresolvedSymlinkArtifactBuilder::setRootRelativePath);
      deserializeOrGetGeneratingActionKey(
          context,
          codedIn,
          builder,
          DeserializedUnresolvedSymlinkArtifactBuilder::setGeneratingActionKey);
      return builder;
    }
  }

  private static final class DeserializedUnresolvedSymlinkArtifactBuilder
      implements DeferredValue<UnresolvedSymlinkArtifact> {
    private final AsyncDeserializationContext context;
    private ArtifactRoot root;
    private PathFragment rootRelativePath;
    private ActionLookupData generatingActionKey;

    private DeserializedUnresolvedSymlinkArtifactBuilder(AsyncDeserializationContext context) {
      this.context = context;
    }

    @Override
    public UnresolvedSymlinkArtifact call() {
      return (UnresolvedSymlinkArtifact)
          context
              .getDependency(ArtifactSerializationContext.class)
              .intern(
                  new UnresolvedSymlinkArtifact(
                      root,
                      getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
                      generatingActionKey),
                  context);
    }

    private static void setRoot(
        DeserializedUnresolvedSymlinkArtifactBuilder builder, Object value) {
      builder.root = (ArtifactRoot) value;
    }

    private static void setRootRelativePath(
        DeserializedUnresolvedSymlinkArtifactBuilder builder, Object value) {
      builder.rootRelativePath = (PathFragment) value;
    }

    private static void setGeneratingActionKey(
        DeserializedUnresolvedSymlinkArtifactBuilder builder, Object value) {
      builder.generatingActionKey = (ActionLookupData) value;
    }
  }

  /**
   * {@link ObjectCodec} for {@link ConstantMetadataArtifact}.
   *
   * <p>To be kept in sync with {@link DerivedArtifactCodec}.
   */
  @Keep // Used by reflection.
  private static final class ConstantMetadataArtifactCodec
      extends DeferredObjectCodec<ConstantMetadataArtifact> {

    private static final ConstantMetadataArtifactCodec INSTANCE =
        new ConstantMetadataArtifactCodec();

    @Override
    public Class<ConstantMetadataArtifact> getEncodedClass() {
      return ConstantMetadataArtifact.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ConstantMetadataArtifact obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getRoot(), codedOut);
      context.serialize(obj.getRootRelativePath(), codedOut);
      serializeOrOmitGeneratingActionKey(context, obj, codedOut);
    }

    @Override
    public DeferredValue<ConstantMetadataArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedConstantMetadataArtifactBuilder builder =
          new DeserializedConstantMetadataArtifactBuilder(context);
      context.deserialize(codedIn, builder, DeserializedConstantMetadataArtifactBuilder::setRoot);
      context.deserialize(
          codedIn, builder, DeserializedConstantMetadataArtifactBuilder::setRootRelativePath);
      deserializeOrGetGeneratingActionKey(
          context,
          codedIn,
          builder,
          DeserializedConstantMetadataArtifactBuilder::setGeneratingActionKey);
      return builder;
    }
  }

  private static final class DeserializedConstantMetadataArtifactBuilder
      implements DeferredValue<ConstantMetadataArtifact> {
    private final AsyncDeserializationContext context;
    private ArtifactRoot root;
    private PathFragment rootRelativePath;
    private ActionLookupData generatingActionKey;

    private DeserializedConstantMetadataArtifactBuilder(AsyncDeserializationContext context) {
      this.context = context;
    }

    @Override
    public ConstantMetadataArtifact call() {
      return (ConstantMetadataArtifact)
          context
              .getDependency(ArtifactSerializationContext.class)
              .intern(
                  new ConstantMetadataArtifact(
                      root,
                      getExecPathForDeserialization(root, rootRelativePath, generatingActionKey),
                      generatingActionKey),
                  context);
    }

    private static void setRoot(DeserializedConstantMetadataArtifactBuilder builder, Object value) {
      builder.root = (ArtifactRoot) value;
    }

    private static void setRootRelativePath(
        DeserializedConstantMetadataArtifactBuilder builder, Object value) {
      builder.rootRelativePath = (PathFragment) value;
    }

    private static void setGeneratingActionKey(
        DeserializedConstantMetadataArtifactBuilder builder, Object value) {
      builder.generatingActionKey = (ActionLookupData) value;
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
    private TreeArtifact treeArtifact;
    private PathFragment derivedTreeRoot;
    private PathFragment rootRelativePath;

    @Override
    public ArchivedTreeArtifact call() {
      return ArchivedTreeArtifact.createWithCustomDerivedTreeRoot(
          treeArtifact, derivedTreeRoot, rootRelativePath);
    }

    private static void setTreeArtifact(
        DeserializedArchivedTreeArtifactBuilder builder, Object value) {
      builder.treeArtifact = (TreeArtifact) value;
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
      serializeOrOmitGeneratingActionKey(context, obj, codedOut);
    }

    @Override
    public DeferredValue<TreeFileArtifact> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      DeserializedTreeFileArtifactBuilder builder = new DeserializedTreeFileArtifactBuilder();
      context.deserialize(codedIn, builder, DeserializedTreeFileArtifactBuilder::setParent);
      context.deserialize(
          codedIn, builder, DeserializedTreeFileArtifactBuilder::setParentRelativePath);
      deserializeOrGetGeneratingActionKey(
          context, codedIn, builder, DeserializedTreeFileArtifactBuilder::setGeneratingActionKey);
      return builder;
    }
  }

  private static class DeserializedTreeFileArtifactBuilder
      implements DeferredValue<TreeFileArtifact> {
    private TreeArtifact parent;
    private PathFragment parentRelativePath;
    private ActionLookupData generatingActionKey;

    @Override
    public TreeFileArtifact call() {
      return new TreeFileArtifact(parent, parentRelativePath, generatingActionKey);
    }

    private static void setParent(DeserializedTreeFileArtifactBuilder builder, Object value) {
      builder.parent = (TreeArtifact) value;
    }

    private static void setParentRelativePath(
        DeserializedTreeFileArtifactBuilder builder, Object value) {
      builder.parentRelativePath = (PathFragment) value;
    }

    private static void setGeneratingActionKey(
        DeserializedTreeFileArtifactBuilder builder, Object value) {
      builder.generatingActionKey = (ActionLookupData) value;
    }
  }

  private ArtifactCodecs() {}
}
