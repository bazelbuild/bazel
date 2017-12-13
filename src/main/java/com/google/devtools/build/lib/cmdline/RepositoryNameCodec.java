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

package com.google.devtools.build.lib.cmdline;

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Custom serialization for {@link RepositoryName}. */
public class RepositoryNameCodec implements ObjectCodec<RepositoryName> {

  @Override
  public Class<RepositoryName> getEncodedClass() {
    return RepositoryName.class;
  }

  @Override
  public void serialize(RepositoryName repoName, CodedOutputStream codedOut) throws IOException {
    boolean isMain = repoName.isMain();
    // Main is by far the most common. Use boolean to short-circuit string encoding on
    // serialization and byte[]/ByteString creation on deserialization.
    codedOut.writeBoolNoTag(isMain);
    if (!isMain) {
      codedOut.writeStringNoTag(repoName.getName());
    }
  }

  @Override
  public RepositoryName deserialize(CodedInputStream codedIn)
      throws SerializationException, IOException {
    boolean isMain = codedIn.readBool();
    if (isMain) {
      return RepositoryName.MAIN;
    }
    try {
      // We can read the string we wrote back as bytes to avoid string decoding/copying.
      return deserializeRepoName(codedIn.readBytes());
    } catch (LabelSyntaxException e) {
      throw new SerializationException("Failed to deserialize RepositoryName", e);
    }
  }

  private static final ByteString DEFAULT_REPOSITORY =
      ByteString.copyFromUtf8(RepositoryName.DEFAULT.getName());
  private static final ByteString MAIN_REPOSITORY =
      ByteString.copyFromUtf8(RepositoryName.MAIN.getName());

  public static RepositoryName deserializeRepoName(ByteString repoNameBytes)
      throws LabelSyntaxException {
    // We expect MAIN_REPOSITORY the vast majority of the time, so check for it first.
    if (repoNameBytes.equals(MAIN_REPOSITORY)) {
      return RepositoryName.MAIN;
    } else if (repoNameBytes.equals(DEFAULT_REPOSITORY)) {
      return RepositoryName.DEFAULT;
    } else {
      return RepositoryName.create(repoNameBytes.toStringUtf8());
    }
  }
}
