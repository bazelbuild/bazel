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

class LongArrayCodec extends LeafObjectCodec<long[]> {
  @Override
  public Class<? extends long[]> getEncodedClass() {
    return long[].class;
  }

  @Override
  public void serialize(LeafSerializationContext context, long[] obj, CodedOutputStream codedOut)
      throws IOException {
    codedOut.writeInt32NoTag(obj.length);
    for (long l : obj) {
      codedOut.writeInt64NoTag(l);
    }
  }

  @Override
  public long[] deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
      throws IOException {
    long[] result = new long[codedIn.readInt32()];
    for (int i = 0; i < result.length; i++) {
      result[i] = codedIn.readInt64();
    }
    return result;
  }
}
