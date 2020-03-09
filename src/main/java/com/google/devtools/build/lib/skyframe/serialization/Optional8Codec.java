// Copyright 2020 The Bazel Authors. All rights reserved.
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
import java.util.Optional;

/** {@link ObjectCodec} for {@link Optional} in Java 8. */
class Optional8Codec implements ObjectCodec<Optional<?>> {
  @SuppressWarnings("unchecked")
  @Override
  public Class<Optional<?>> getEncodedClass() {
    return (Class<Optional<?>>) (Class<?>) Optional.class;
  }

  @Override
  public void serialize(SerializationContext context, Optional<?> obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeBoolNoTag(obj.isPresent());
    if (obj.isPresent()) {
      context.serialize(obj.get(), codedOut);
    }
  }

  @Override
  public Optional<?> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    if (!codedIn.readBool()) {
      return Optional.empty();
    }
    return Optional.of(context.deserialize(codedIn));
  }
}
