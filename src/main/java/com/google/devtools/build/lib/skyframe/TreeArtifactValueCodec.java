// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.cache.MetadataDigestUtils;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.DeserializedSkyValue;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * A wrapper around the AutoCodec-generated codec for {@link TreeArtifactValue} that makes sure the
 * {@link TreeArtifactValue#empty()} constant is deserialized into a different constant ({@link
 * #EMPTY_DESERIALIZED}) that implements the {@link DeserializedSkyValue} marker interface.
 */
@Keep // Accessed reflectively.
class TreeArtifactValueCodec extends DeferredObjectCodec<TreeArtifactValue> {
  private static final TreeArtifactValue EMPTY_DESERIALIZED =
      new TreeArtifactValue_AutoCodec.Deserialized(
          MetadataDigestUtils.fromMetadata(ImmutableMap.of()),
          TreeArtifactValue.EMPTY_MAP,
          0L,
          /* archivedRepresentation= */ null,
          /* resolvedPath= */ null,
          /* entirelyRemote= */ false);

  private static final DeferredObjectCodec<TreeArtifactValue> AUTOCODEC =
      new TreeArtifactValue_AutoCodec();

  @Override
  public Class<TreeArtifactValue> getEncodedClass() {
    return TreeArtifactValue.class;
  }

  @Override
  public ImmutableSet<Class<? extends TreeArtifactValue>> additionalEncodedClasses() {
    return ImmutableSet.of(TreeArtifactValue_AutoCodec.Deserialized.class);
  }

  @Override
  public void serialize(
      SerializationContext context, TreeArtifactValue obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (obj.equals(TreeArtifactValue.empty())) {
      codedOut.writeBoolNoTag(true);
    } else {
      codedOut.writeBoolNoTag(false);
      AUTOCODEC.serialize(context, obj, codedOut);
    }
  }

  @Override
  public DeferredValue<? extends TreeArtifactValue> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    if (codedIn.readBool()) {
      return () -> EMPTY_DESERIALIZED;
    }
    return AUTOCODEC.deserializeDeferred(context, codedIn);
  }
}
