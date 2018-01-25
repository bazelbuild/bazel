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

package com.google.devtools.build.lib.skyframe.serialization.serializers;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoException;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.google.protobuf.AbstractMessage;
import com.google.protobuf.InvalidProtocolBufferException;

/**
 * Serializer for protos.
 *
 * <p>A separate instance must be registered for each distinct proto.
 */
public class ProtoSerializer<T extends AbstractMessage> extends Serializer<T> {

  private final ParseFromHandle<T> handle;

  /** Wrapper for {@code parseFrom} references. */
  @FunctionalInterface
  public static interface ParseFromHandle<T> {
    T parseFrom(byte[] bytes) throws InvalidProtocolBufferException;
  }

  /**
   * Constructor.
   *
   * @param handle reference to T.parseFrom
   */
  public ProtoSerializer(ParseFromHandle<T> handle) {
    setImmutable(true);
    this.handle = handle;
  }

  @Override
  public void write(Kryo kryo, Output output, T message) {
    byte[] bytes = message.toByteArray();
    output.writeInt(bytes.length, true);
    output.writeBytes(bytes);
  }

  @Override
  public T read(Kryo kryo, Input input, Class<T> type) {
    try {
      return handle.parseFrom(input.readBytes(input.readInt(true)));
    } catch (InvalidProtocolBufferException e) {
      throw new KryoException("Failed to parse " + type.getCanonicalName(), e);
    }
  }
}
