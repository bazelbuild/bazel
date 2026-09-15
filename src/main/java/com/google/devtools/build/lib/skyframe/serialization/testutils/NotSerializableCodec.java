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

package com.google.devtools.build.lib.skyframe.serialization.testutils;

import com.google.devtools.build.lib.skyframe.serialization.LeafDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.LeafObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.LeafSerializationContext;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.NotSerializableException;

/** A testing helper to force serialization errors. */
public class NotSerializableCodec extends LeafObjectCodec<Object> {
  private final Class<?> type;

  public NotSerializableCodec(Class<?> type) {
    this.type = type;
  }

  @Override
  public Class<?> getEncodedClass() {
    return type;
  }

  @Override
  public void serialize(
      LeafSerializationContext context, Object unusedObj, CodedOutputStream unusedCodedOut)
      throws NotSerializableException {
    throw new NotSerializableException(type + " marked not serializable");
  }

  @Override
  public Object deserialize(LeafDeserializationContext context, CodedInputStream unusedCodedIn)
      throws NotSerializableException {
    throw new NotSerializableException(type + " marked not serializable");
  }
}
