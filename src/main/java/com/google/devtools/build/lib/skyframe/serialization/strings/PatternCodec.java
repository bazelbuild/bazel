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
package com.google.devtools.build.lib.skyframe.serialization.strings;

import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;

import com.google.devtools.build.lib.skyframe.serialization.LeafDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.LeafObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.LeafSerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.regex.Pattern;

class PatternCodec extends LeafObjectCodec<Pattern> {

  @Override
  public Class<Pattern> getEncodedClass() {
    return Pattern.class;
  }

  @Override
  public void serialize(
      LeafSerializationContext context, Pattern pattern, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    context.serializeLeaf(pattern.pattern(), stringCodec(), codedOut);
    codedOut.writeInt32NoTag(pattern.flags());
  }

  @Override
  public Pattern deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return Pattern.compile(context.deserializeLeaf(codedIn, stringCodec()), codedIn.readInt32());
  }
}
