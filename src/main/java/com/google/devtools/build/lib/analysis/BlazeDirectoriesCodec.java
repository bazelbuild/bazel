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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.PathCodec;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * {@link ObjectCodec} for {@link BlazeDirectories}. Delegates to {@link BlazeDirectories} for
 * everything.
 */
public class BlazeDirectoriesCodec implements ObjectCodec<BlazeDirectories> {
  private final PathCodec pathCodec;

  public BlazeDirectoriesCodec(PathCodec pathCodec) {
    this.pathCodec = pathCodec;
  }

  @Override
  public Class<BlazeDirectories> getEncodedClass() {
    return BlazeDirectories.class;
  }

  @Override
  public void serialize(BlazeDirectories obj, CodedOutputStream codedOut) throws IOException {
    obj.serialize(codedOut, pathCodec);
  }

  @Override
  public BlazeDirectories deserialize(CodedInputStream codedIn) throws IOException {
    return BlazeDirectories.deserialize(codedIn, pathCodec);
  }
}
