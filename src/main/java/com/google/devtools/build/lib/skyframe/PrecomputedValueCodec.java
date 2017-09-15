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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.function.Supplier;

/**
 * {@link ObjectCodec} for {@link PrecomputedValue} objects. Because {@link PrecomputedValue}
 * objects can theoretically contain any kind of value object as their value, an {@link
 * ObjectCodecs} instance must be accessible to this codec during serialization/deserialization, so
 * that it can handle the arbitrary objects it's given.
 */
public class PrecomputedValueCodec implements ObjectCodec<PrecomputedValue> {
  private final Supplier<ObjectCodecs> objectCodecsSupplier;

  public PrecomputedValueCodec(Supplier<ObjectCodecs> objectCodecsSupplier) {
    this.objectCodecsSupplier = objectCodecsSupplier;
  }

  @Override
  public Class<PrecomputedValue> getEncodedClass() {
    return PrecomputedValue.class;
  }

  @Override
  public void serialize(PrecomputedValue obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    ObjectCodecs objectCodecs = objectCodecsSupplier.get();
    Object val = obj.get();
    Preconditions.checkState(!(val instanceof PrecomputedValue), "recursive precomputed: %s", obj);
    // TODO(janakr): this assumes the classifier is the class of the object. This should be enforced
    // by the ObjectCodecs instance.
    String classifier = val.getClass().getName();
    codedOut.writeStringNoTag(classifier);
    objectCodecs.serialize(classifier, val, codedOut);
  }

  @Override
  public PrecomputedValue deserialize(CodedInputStream codedIn)
      throws SerializationException, IOException {
    ObjectCodecs objectCodecs = objectCodecsSupplier.get();
    Object val = objectCodecs.deserialize(codedIn.readBytes(), codedIn);
    return new PrecomputedValue(val);
  }
}
