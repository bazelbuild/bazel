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

import com.google.devtools.build.lib.skyframe.serialization.LeafObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationDependencyProvider;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;

class CharsetCodec extends LeafObjectCodec<Charset> {

  @Override
  public Class<Charset> getEncodedClass() {
    return Charset.class;
  }

  @Override
  public void serialize(
      SerializationDependencyProvider dependencies, Charset charset, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    stringCodec().serialize(dependencies, charset.name(), codedOut);
  }

  @Override
  public Charset deserialize(SerializationDependencyProvider dependencies, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return Charset.forName(stringCodec().deserialize(dependencies, codedIn));
  }
}
