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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Custom serialization for {@link PathFragment}s. */
class PathFragmentCodec implements ObjectCodec<PathFragment> {

  private final ObjectCodec<String> stringCodec = new FastStringCodec();

  @Override
  public Class<PathFragment> getEncodedClass() {
    return PathFragment.class;
  }

  @Override
  public void serialize(PathFragment pathFragment, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    codedOut.writeInt32NoTag(pathFragment.getDriveLetter());
    codedOut.writeBoolNoTag(pathFragment.isAbsolute());
    codedOut.writeInt32NoTag(pathFragment.segmentCount());
    for (int i = 0; i < pathFragment.segmentCount(); i++) {
      stringCodec.serialize(pathFragment.getSegment(i), codedOut);
    }
  }

  @Override
  public PathFragment deserialize(CodedInputStream codedIn)
      throws IOException, SerializationException {
    char driveLetter = (char) codedIn.readInt32();
    boolean isAbsolute = codedIn.readBool();
    int segmentCount = codedIn.readInt32();
    String[] segments = new String[segmentCount];
    for (int i = 0; i < segmentCount; i++) {
      segments[i] = stringCodec.deserialize(codedIn);
    }
    return PathFragment.create(driveLetter, isAbsolute, segments);
  }
}
