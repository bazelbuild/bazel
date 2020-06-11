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
package com.google.devtools.build.lib.unix;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.util.FsApparatus;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for ProcMeminfoParser.
 */
@RunWith(JUnit4.class)
public class ProcMeminfoParserTest {

  private FsApparatus scratch = FsApparatus.newNative();

  @Test
  public void memInfo() throws Exception {
    String meminfoContent =
        StringUtilities.joinLines(
            "MemTotal:      3091732 kB",
            "MemFree:       2167344 kB",
            "MemAvailable:   14717640 kB",
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
            "Writeback:         123 kB",
            "Not even a valid line");

    String meminfoFile = scratch.file("test_meminfo", meminfoContent).getPathString();
    ProcMeminfoParser memInfo = new ProcMeminfoParser(meminfoFile);

    assertThat(memInfo.getFreeRamKb()).isEqualTo(14717640);
    assertThat(memInfo.getRamKb("Cached")).isEqualTo(509940);
    assertThat(memInfo.getTotalKb()).isEqualTo(3091732);
    assertThat(memInfo.getRamKb("Writeback")).isEqualTo(0);
    assertThrows(ProcMeminfoParser.KeywordNotFoundException.class,
        () -> memInfo.getRamKb("Bogus"));
    assertThrows(ProcMeminfoParser.KeywordNotFoundException.class,
        () -> memInfo.getRamKb("Bogus2"));
  }

  @Test
  public void testOldKernelFallback() throws Exception {
    String meminfoContent =
        StringUtilities.joinLines(
            "MemTotal:      3091732 kB",
            "Active:         636892 kB",
            "Inactive:       212760 kB",
            "Slab:            42820 kB");

    String meminfoFile = scratch.file("test_meminfo", meminfoContent).getPathString();
    ProcMeminfoParser memInfo = new ProcMeminfoParser(meminfoFile);
    assertThat(memInfo.getFreeRamKb()).isEqualTo(2356756);
  }
}
