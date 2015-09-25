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
package com.google.devtools.build.android.ziputils;

import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENTIM;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Unit tests for {@link ZipOut}.
 */
@RunWith(JUnit4.class)
public class ZipOutTest {

  private static final FakeFileSystem fileSystem = new FakeFileSystem();

  @Test
  public void testNextEntry() {
    try {
      String filename = "out.zip";
      ZipOut instance = new ZipOut(fileSystem.getOutputChannel(filename, false), filename);

      instance.nextEntry(DirectoryEntry.allocate("pgk/a.class", null, null))
          .set(CENTIM, DosTime.EPOCH.time);

      instance.nextEntry(DirectoryEntry.allocate("pgk/b.class", null, null))
          .set(CENTIM, DosTime.EPOCH.time);

      instance.close();
    } catch (IOException ex) {
      Logger.getLogger(ZipOutTest.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
}
