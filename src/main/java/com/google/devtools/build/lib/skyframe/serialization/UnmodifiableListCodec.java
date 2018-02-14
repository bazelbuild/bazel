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

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/** A {@link ObjectCodec} for objects produced by {@link Collections#unmodifiableList}. */
class UnmodifiableListCodec implements ObjectCodec<List<Object>> {

  private static final Class<?> RANDOM_ACCESS_TYPE =
      Collections.unmodifiableList(new ArrayList<Object>()).getClass();
  private static final Class<?> SEQUENTIAL_ACCESS_TYPE =
      Collections.unmodifiableList(new LinkedList<Object>()).getClass();

  @Override
  public Class<List<Object>> getEncodedClass() {
    return null; // No reasonable value here.
  }

  @Override
  public void serialize(
      SerializationContext context, List<Object> object, CodedOutputStream output) {
    // TODO(shahan): Stub. Replace with actual implementation, which requires the registry to be
    // added to the context.
  }

  @Override
  public List<Object> deserialize(DeserializationContext context, CodedInputStream input) {
    // TODO(shahan): Stub. Replace with actual implementation, which requires the registry to be
    // added to the context.
    return null;
  }

  static class UnmodifiableListCodecRegisterer implements CodecRegisterer<UnmodifiableListCodec> {
    @SuppressWarnings({"rawtypes", "unchecked"})
    @Override
    public void register(ObjectCodecRegistry.Builder builder) {
      UnmodifiableListCodec codec = new UnmodifiableListCodec();
      builder.add((Class) RANDOM_ACCESS_TYPE, (ObjectCodec) codec);
      builder.add((Class) SEQUENTIAL_ACCESS_TYPE, (ObjectCodec) codec);
    }
  }
}
