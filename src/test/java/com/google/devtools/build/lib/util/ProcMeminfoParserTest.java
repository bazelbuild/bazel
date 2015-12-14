// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.vfs.util.FsApparatus;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

/**
 * Tests for ProcMeminfoParser.
 */
@RunWith(JUnit4.class)
public class ProcMeminfoParserTest {

  private FsApparatus scratch = FsApparatus.newNative();

  @Test
  public void memInfo() throws IOException {
    String meminfoContent = StringUtilities.joinLines(
        "MemTotal:      3091732 kB",
        "MemFree:       2167344 kB",
        "Buffers:         60644 kB",
        "Cached:         509940 kB",
        "SwapCached:          0 kB",
        "Active:         636892 kB",
        "Inactive:       212760 kB",
        "HighTotal:           0 kB",
        "HighFree:            0 kB",
        "LowTotal:      3091732 kB",
        "LowFree:       2167344 kB",
        "SwapTotal:     9124880 kB",
        "SwapFree:      9124880 kB",
        "Dirty:               0 kB",
        "Writeback:           0 kB",
        "AnonPages:      279028 kB",
        "Mapped:          54404 kB",
        "Slab:            42820 kB",
        "PageTables:       5184 kB",
        "NFS_Unstable:        0 kB",
        "Bounce:              0 kB",
        "CommitLimit:  10670744 kB",
        "Committed_AS:   665840 kB",
        "VmallocTotal: 34359738367 kB",
        "VmallocUsed:    300484 kB",
        "VmallocChunk: 34359437307 kB",
        "HugePages_Total:     0",
        "HugePages_Free:      0",
        "HugePages_Rsvd:      0",
        "Hugepagesize:     2048 kB",
        "Bogus: not_a_number",
        "Bogus2: 1000000000000000000000000000000000000000000000000 kB",
        "Not even a valid line"
    );

    String meminfoFile = scratch.file("test_meminfo", meminfoContent).getPathString();
    ProcMeminfoParser memInfo = new ProcMeminfoParser(meminfoFile);

    assertEquals(2356756, memInfo.getFreeRamKb());
    assertEquals(509940, memInfo.getRamKb("Cached"));
    assertEquals(3091732, memInfo.getTotalKb());
    assertNotAvailable("Bogus", memInfo);
    assertNotAvailable("Bogus2", memInfo);
  }

  private static void assertNotAvailable(String field, ProcMeminfoParser memInfo) {
    try {
      memInfo.getRamKb(field);
      fail();
    } catch (IllegalArgumentException e) {
      // Expected.
    }
  }

}
