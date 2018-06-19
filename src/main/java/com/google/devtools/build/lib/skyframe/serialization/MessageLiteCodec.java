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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.collect.ImmutableList;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageLite;
import com.google.protobuf.UnknownFieldSet;
import java.io.IOException;
import java.util.function.Supplier;

/** Codec for protos. */
public class MessageLiteCodec implements ObjectCodec<MessageLite> {

  private final Supplier<MessageLite.Builder> builderSupplier;
  private final Class<? extends MessageLite> type;

  /**
   * Constructor.
   *
   * @param builderSupplier reference to a proto's newBuilder method
   */
  public MessageLiteCodec(Supplier<MessageLite.Builder> builderSupplier) {
    this.builderSupplier = builderSupplier;
    this.type = builderSupplier.get().buildPartial().getClass();
  }

  @Override
  public Class<? extends MessageLite> getEncodedClass() {
    return type;
  }

  @Override
  public void serialize(
      SerializationContext unusedContext, MessageLite message, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    codedOut.writeMessageNoTag(message);
  }

  @Override
  public MessageLite deserialize(DeserializationContext unusedContext, CodedInputStream codedIn)
      throws IOException, SerializationException {
    // Don't hold on to full byte array when constructing this proto.
    codedIn.enableAliasing(false);
    try {
      MessageLite.Builder builder = builderSupplier.get();
      codedIn.readMessage(builder, ExtensionRegistryLite.getEmptyRegistry());
      return builder.build();
    } catch (InvalidProtocolBufferException e) {
      throw new SerializationException("Failed to parse proto of type " + type, e);
    } finally {
      codedIn.enableAliasing(true);
    }
  }

  private static class MessageLiteCodecRegisterer implements CodecRegisterer<MessageLiteCodec> {
    @Override
    public ImmutableList<MessageLiteCodec> getCodecsToRegister() {
      return ImmutableList.of(new MessageLiteCodec(UnknownFieldSet::newBuilder));
    }
  }
}
