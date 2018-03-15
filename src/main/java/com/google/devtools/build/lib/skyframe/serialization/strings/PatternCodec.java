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

import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.regex.Pattern;

class PatternCodec implements ObjectCodec<Pattern> {

  @Override
  public Class<Pattern> getEncodedClass() {
    return Pattern.class;
  }

  @Override
  public void serialize(SerializationContext context, Pattern pattern, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    context.serialize(pattern.pattern(), codedOut);
    codedOut.writeInt32NoTag(pattern.flags());
  }

  @Override
  public Pattern deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return Pattern.compile(context.deserialize(codedIn), codedIn.readInt32());
  }
}
