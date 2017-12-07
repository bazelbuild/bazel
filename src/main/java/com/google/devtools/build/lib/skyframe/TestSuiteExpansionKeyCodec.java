// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.TestSuiteExpansionValue.TestSuiteExpansionKey;
import com.google.devtools.build.lib.skyframe.serialization.LabelCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Comparator;

/** Custom serialization for {@link TestSuiteExpansionKey}. */
class TestSuiteExpansionKeyCodec implements ObjectCodec<TestSuiteExpansionKey> {
  public static final TestSuiteExpansionKeyCodec INSTANCE = new TestSuiteExpansionKeyCodec();

  private TestSuiteExpansionKeyCodec() {}

  @Override
  public Class<TestSuiteExpansionKey> getEncodedClass() {
    return TestSuiteExpansionKey.class;
  }

  @Override
  public void serialize(TestSuiteExpansionKey key, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    // Writes the target count to the stream so deserialization knows when to stop.
    codedOut.writeInt32NoTag(key.getTargets().size());
    for (Label label : key.getTargets()) {
      LabelCodec.INSTANCE.serialize(label, codedOut);
    }
  }

  @Override
  public TestSuiteExpansionKey deserialize(CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    ImmutableSortedSet.Builder<Label> builder =
        new ImmutableSortedSet.Builder<>(Comparator.naturalOrder());
    for (int i = 0; i < size; ++i) {
      builder.add(LabelCodec.INSTANCE.deserialize(codedIn));
    }
    return new TestSuiteExpansionKey(builder.build());
  }
}
