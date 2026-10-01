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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Optional;

/** Custom codec for {@link Optional} to support JEP 401 value classes safely. */
@SuppressWarnings("rawtypes")
final class OptionalCodec extends DeferredObjectCodec<Optional> {

  @Override
  public Class<Optional> getEncodedClass() {
    return Optional.class;
  }

  @Override
  public void serialize(SerializationContext context, Optional obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (obj.isPresent()) {
      codedOut.writeBoolNoTag(true);
      context.serialize(obj.get(), codedOut);
    } else {
      codedOut.writeBoolNoTag(false);
    }
  }

  @Override
  public DeferredValue<Optional> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    if (!codedIn.readBool()) {
      return Optional::empty;
    }
    var builder = new OptionalBuilder();
    context.deserialize(codedIn, builder, OptionalBuilder::setValue);
    return builder;
  }

  private static final class OptionalBuilder implements DeferredValue<Optional> {
    private Object value;

    @Override
    public Optional call() {
      return Optional.of(value);
    }

    private static final void setValue(OptionalBuilder builder, Object value) {
      builder.value = value;
    }
  }
}
