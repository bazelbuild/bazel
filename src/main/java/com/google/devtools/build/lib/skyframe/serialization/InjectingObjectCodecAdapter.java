// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Adapts an InjectingObjectCodec to an ObjectCodec. */
public class InjectingObjectCodecAdapter<T, D> implements ObjectCodec<T> {

  private final InjectingObjectCodec<T, D> codec;
  private final D dependency;

  /**
   * Creates an ObjectCodec from an InjectingObjectCodec.
   *
   * @param codec underlying codec to delegate to
   * @param dependency object forwarded to {@code codec.deserialize}
   */
  public InjectingObjectCodecAdapter(InjectingObjectCodec<T, D> codec, D dependency) {
    this.codec = codec;
    this.dependency = dependency;
  }

  @Override
  public Class<T> getEncodedClass() {
    return codec.getEncodedClass();
  }

  @Override
  public void serialize(T obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codec.serialize(dependency, obj, codedOut);
  }

  @Override
  public T deserialize(CodedInputStream codedIn) throws SerializationException, IOException {
    return codec.deserialize(dependency, codedIn);
  }
}
