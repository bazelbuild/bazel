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

import static org.junit.Assert.fail;

import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RepositoryNameCodec}. */
@RunWith(JUnit4.class)
public class RepositoryNameCodecTest extends AbstractObjectCodecTest<RepositoryName> {

  // Set 0th byte (isMain) false, so that we'll try to read a string from the rest of the
  // data and fail.
  public static final byte[] INVALID_ENCODED_REPOSITORY_NAME = new byte[] {0, 10, 9, 8, 7};

  public RepositoryNameCodecTest() throws LabelSyntaxException {
    super(
        new RepositoryNameCodec(),
        RepositoryName.create(RepositoryName.DEFAULT.getName()),
        RepositoryName.create(RepositoryName.MAIN.getName()),
        RepositoryName.create("@externalandshouldntexistinthisworkspace"));
  }

  // The default bad data test from AbstractObjectCodecTest doesn't play nice with boolean prefixed
  // encodings.
  @Override
  @Test
  public void testDeserializeBadDataThrowsSerializationException() {
    try {
      fromBytes(INVALID_ENCODED_REPOSITORY_NAME);
      fail("Expected exception");
    } catch (SerializationException | IOException e) {
      // Expected.
    }
  }
}
