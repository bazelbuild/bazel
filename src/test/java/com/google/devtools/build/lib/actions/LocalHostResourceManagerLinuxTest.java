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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.util.FsApparatus;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for LocalHostResourceManagerLinux.
 */
@RunWith(JUnit4.class)
public class LocalHostResourceManagerLinuxTest {

  private final FsApparatus scratch = FsApparatus.newNative();

  @Test
  public void testNonHyperthreadedMachine() throws Exception {
    String meminfoContent =
        StringUtilities.joinLines(
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
            "Hugepagesize:     2048 kB");
    String meminfoFile = scratch.file("test_meminfo_nonht", meminfoContent).getPathString();
    // +/- 0.1MB
    assertThat(LocalHostResourceManagerLinux.getMemoryInMbHelper(meminfoFile))
        .isWithin(0.1)
        .of(3091732 >> 10); // kibis to meibis
  }

  @Test
  public void testHyperthreadedMachine() throws Exception {
    String meminfoContent =
        StringUtilities.joinLines(
            "MemTotal:      3092004 kB",
            "MemFree:         26124 kB",
            "Buffers:          3836 kB",
            "Cached:          52400 kB",
            "SwapCached:      68204 kB",
            "Active:        2281464 kB",
            "Inactive:       260908 kB",
            "HighTotal:           0 kB",
            "HighFree:            0 kB",
            "LowTotal:      3092004 kB",
            "LowFree:         26124 kB",
            "SwapTotal:     9124880 kB",
            "SwapFree:      8264920 kB",
            "Dirty:             616 kB",
            "Writeback:           0 kB",
            "AnonPages:     2466336 kB",
            "Mapped:          37576 kB",
            "Slab:           483004 kB",
            "PageTables:      11912 kB",
            "NFS_Unstable:        0 kB",
            "Bounce:              0 kB",
            "CommitLimit:  10670880 kB",
            "Committed_AS:  3627984 kB",
            "VmallocTotal: 34359738367 kB",
            "VmallocUsed:    300460 kB",
            "VmallocChunk: 34359437307 kB",
            "HugePages_Total:     0",
            "HugePages_Free:      0",
            "HugePages_Rsvd:      0",
            "Hugepagesize:     2048 kB");
    String meminfoFile = scratch.file("test_meminfo_ht", meminfoContent).getPathString();
    // +/- 0.1MB
    assertThat(LocalHostResourceManagerLinux.getMemoryInMbHelper(meminfoFile))
        .isWithin(0.1)
        .of(3092004 >> 10); // kibis to meibis
  }

  @Test
  public void testAMDMachine() throws Exception {
    String meminfoContent =
        StringUtilities.joinLines(
            "MemTotal:      8223956 kB",
            "MemFree:       3670396 kB",
            "Buffers:        374068 kB",
            "Cached:        3366980 kB",
            "SwapCached:          0 kB",
            "Active:        3275860 kB",
            "Inactive:       737816 kB",
            "HighTotal:           0 kB",
            "HighFree:            0 kB",
            "LowTotal:      8223956 kB",
            "LowFree:       3670396 kB",
            "SwapTotal:     6024332 kB",
            "SwapFree:      6024332 kB",
            "Dirty:              84 kB",
            "Writeback:           0 kB",
            "AnonPages:      272308 kB",
            "Mapped:          62604 kB",
            "Slab:           506140 kB",
            "PageTables:       4608 kB",
            "NFS_Unstable:        0 kB",
            "Bounce:              0 kB",
            "CommitLimit:  10136308 kB",
            "Committed_AS:   600672 kB",
            "VmallocTotal: 34359738367 kB",
            "VmallocUsed:    299068 kB",
            "VmallocChunk: 34359438843 kB",
            "HugePages_Total:     0",
            "HugePages_Free:      0",
            "HugePages_Rsvd:      0",
            "Hugepagesize:     2048 kB");
    String meminfoFile = scratch.file("test_meminfo_amd", meminfoContent).getPathString();
    // +/- 0.1MB
    assertThat(LocalHostResourceManagerLinux.getMemoryInMbHelper(meminfoFile))
        .isWithin(0.1)
        .of(8223956 >> 10); // kibis to meibis
  }
}
