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
package com.google.devtools.build.android.dexer;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.inOrder;

import com.android.dex.Dex;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.InOrder;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/** Tests for {@link DexFileArchive}. */
@RunWith(JUnit4.class)
public class DexFileArchiveTest {

  @Mock private ZipOutputStream out;

  @Before
  public void setUp() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testAddDex() throws Exception {
    ZipEntry entry = new ZipEntry("test.dex");
    Dex dex = new Dex(1);
    try (DexFileArchive archive = new DexFileArchive(out)) {
      archive.addFile(entry, dex);
    }
    assertThat(entry.getSize()).isEqualTo(1L);
    InOrder order = inOrder(out);
    order.verify(out).putNextEntry(entry);
    order.verify(out).write(any(byte[].class), eq(0), eq(1));
    order.verify(out).close();
  }
}
