// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext.HeaderInfo;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * A codec for {@link HeaderInfo} with intra-value memoization.
 *
 * <p>{@link HeaderInfo} recursively refers to other {@link HeaderInfo} instances that must be
 * serialized in memoized fashion to avoid quadratic storage costs.
 *
 * <p>This codec is only used via delegation.
 */
public final class HeaderInfoCodec extends DeferredObjectCodec<HeaderInfo> {
  private static final HeaderInfoCodec INSTANCE = new HeaderInfoCodec();

  static HeaderInfoCodec headerInfoCodec() {
    return INSTANCE;
  }

  @Override
  public Class<HeaderInfo> getEncodedClass() {
    return HeaderInfo.class;
  }

  @Override
  public boolean autoRegister() {
    return false; // Used only by delegation from CcCompilationContextCodec.
  }

  @Override
  public void serialize(
      SerializationContext context, HeaderInfo headerInfo, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    context.serialize(headerInfo.headerModule, codedOut);
    context.serialize(headerInfo.picHeaderModule, codedOut);
    context.serialize(headerInfo.modularPublicHeaders, codedOut);
    context.serialize(headerInfo.modularPrivateHeaders, codedOut);
    context.serialize(headerInfo.textualHeaders, codedOut);
    context.serialize(headerInfo.separateModuleHeaders, codedOut);
    context.serialize(headerInfo.separateModule, codedOut);
    context.serialize(headerInfo.separatePicModule, codedOut);

    ImmutableList<HeaderInfo> deps = headerInfo.deps;
    codedOut.writeInt32NoTag(deps.size());
    for (HeaderInfo dep : deps) {
      context.putSharedValue(dep, /* distinguisher= */ null, this, codedOut);
    }
  }

  @Override
  public DeferredValue<HeaderInfo> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Builder builder = new Builder();

    context.deserialize(codedIn, builder, Builder::setHeaderModule);
    context.deserialize(codedIn, builder, Builder::setPicHeaderModule);
    context.deserialize(codedIn, builder, Builder::setModularPublicHeaders);
    context.deserialize(codedIn, builder, Builder::setModularPrivateHeaders);
    context.deserialize(codedIn, builder, Builder::setTextualHeaders);
    context.deserialize(codedIn, builder, Builder::setSeparateModuleHeaders);
    context.deserialize(codedIn, builder, Builder::setSeparateModule);
    context.deserialize(codedIn, builder, Builder::setSeparatePicModule);

    int depsCount = codedIn.readInt32();
    builder.deps = new HeaderInfo[depsCount];
    for (int i = 0; i < depsCount; i++) {
      final int indexForCapture = i;
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          this,
          builder,
          (b, obj) -> b.deps[indexForCapture] = (HeaderInfo) obj);
    }
    return builder;
  }

  private static class Builder implements DeferredValue<HeaderInfo> {
    private DerivedArtifact headerModule;
    private DerivedArtifact picHeaderModule;
    private ImmutableList<Artifact> modularPublicHeaders;
    private ImmutableList<Artifact> modularPrivateHeaders;
    private ImmutableList<Artifact> textualHeaders;
    private ImmutableList<Artifact> separateModuleHeaders;
    private DerivedArtifact separateModule;
    private DerivedArtifact separatePicModule;
    private HeaderInfo[] deps;

    @Override
    public HeaderInfo call() {
      return new HeaderInfo(
          headerModule,
          picHeaderModule,
          modularPublicHeaders,
          modularPrivateHeaders,
          textualHeaders,
          separateModuleHeaders,
          separateModule,
          separatePicModule,
          ImmutableList.copyOf(deps));
    }

    private static void setHeaderModule(Builder builder, Object obj) {
      builder.headerModule = (DerivedArtifact) obj;
    }

    private static void setPicHeaderModule(Builder builder, Object obj) {
      builder.picHeaderModule = (DerivedArtifact) obj;
    }

    @SuppressWarnings("unchecked")
    private static void setModularPublicHeaders(Builder builder, Object obj) {
      builder.modularPublicHeaders = (ImmutableList<Artifact>) obj;
    }

    @SuppressWarnings("unchecked")
    private static void setModularPrivateHeaders(Builder builder, Object obj) {
      builder.modularPrivateHeaders = (ImmutableList<Artifact>) obj;
    }

    @SuppressWarnings("unchecked")
    private static void setTextualHeaders(Builder builder, Object obj) {
      builder.textualHeaders = (ImmutableList<Artifact>) obj;
    }

    @SuppressWarnings("unchecked")
    private static void setSeparateModuleHeaders(Builder builder, Object obj) {
      builder.separateModuleHeaders = (ImmutableList<Artifact>) obj;
    }

    private static void setSeparateModule(Builder builder, Object obj) {
      builder.separateModule = (DerivedArtifact) obj;
    }

    private static void setSeparatePicModule(Builder builder, Object obj) {
      builder.separatePicModule = (DerivedArtifact) obj;
    }
  }

  private HeaderInfoCodec() {}
}
