// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.singlejar;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * FileSystem for testing. FileSystem supports exactly one one OutputStream for filename
 * specified in constructor.
 * Workflow for using this class in tests are following:
 * <ul>
 *   <li> Construct with exactly one outputFile. </li>
 *   <li> add some input files using method addFile </li>
 *   <li> check content of outputFile calling toByteArray </li>
 * </ul>
 */
public final class MockSimpleFileSystem implements SimpleFileSystem {

  private final String outputFileName;
  private ByteArrayOutputStream out;
  private final Map<String, byte[]> files = new HashMap<>();

  public MockSimpleFileSystem(String outputFileName) {
    this.outputFileName = outputFileName;
  }

  public void addFile(String name, byte[] content) {
    files.put(name, content);
  }

  public void addFile(String name, String content) {
    files.put(name, content.getBytes(UTF_8));
  }

  @Override
  public OutputStream getOutputStream(String filename) {
    assertEquals(outputFileName, filename);
    assertNull(out);
    out = new ByteArrayOutputStream();
    return out;
  }

  @Override
  public InputStream getInputStream(String filename) throws IOException {
    byte[] data = files.get(filename);
    if (data == null) {
      throw new FileNotFoundException();
    }
    return new ByteArrayInputStream(data);
  }

  @Override
  public boolean delete(String filename) {
    assertEquals(outputFileName, filename);
    assertNotNull(out);
    out = null;
    return true;
  }

  public byte[] toByteArray() {
    assertNotNull(out);
    return out.toByteArray();
  }
}
