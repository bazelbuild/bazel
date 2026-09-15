// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.Instantiator;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link Path}. */
@RunWith(JUnit4.class)
public final class PathTest {

  @Test
  public void serialization_roundTripsEqualObject() throws Exception {
    FileSystem fileSystem = SerializableMockFileSystem.INSTANCE;
    Path path = fileSystem.getPath("/file");

    new SerializationTester(path).runTests();
  }

  @AutoCodec
  static final class SerializableMockFileSystem extends DelegateFileSystem {
    private static final SerializableMockFileSystem INSTANCE = new SerializableMockFileSystem();

    private SerializableMockFileSystem() {
      super(createMock());
    }

    private static FileSystem createMock() {
      FileSystem fileSystem = mock(FileSystem.class);
      when(fileSystem.getDigestFunction()).thenReturn(DigestHashFunction.SHA256);
      return fileSystem;
    }

    @SuppressWarnings("unused") // Used by (de)serialization.
    @Instantiator
    static SerializableMockFileSystem getInstance() {
      return INSTANCE;
    }
  }
}
