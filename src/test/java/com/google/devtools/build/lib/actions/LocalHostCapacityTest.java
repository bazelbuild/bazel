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
package com.google.devtools.build.lib.actions;


import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.util.FsApparatus;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class LocalHostCapacityTest {

  private FsApparatus scratch = FsApparatus.newNative();

  @Test
  public void testNonHyperthreadedMachine() throws Exception {
    String cpuinfoContent = StringUtilities.joinLines(
        "processor\t: 0",
        "vendor_id\t: GenuineIntel",
        "cpu family\t: 15",
        "model\t\t: 4",
        "model name\t:               Intel(R) Pentium(R) 4 CPU 3.40GHz",
        "stepping\t: 10",
        "cpu MHz\t\t: 3400.000",
        "cache size\t: 2048 KB",
        "fpu\t\t: yes",
        "fpu_exception\t: yes",
        "cpuid level\t: 5",
        "wp\t\t: yes",
        "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca "
            + "cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm "
            + "syscall nx lm constant_tsc up pni monitor ds_cpl est cid cx16 "
            + "xtpr lahf_lm",
        "bogomips\t: 6803.83",
        "clflush size\t: 64",
        "cache_alignment\t: 128",
        "address sizes\t: 36 bits physical, 48 bits virtual",
        "power management:"
        );
    String cpuinfoFile =
        scratch.file("test_cpuinfo_nonht", cpuinfoContent).getPathString();
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
        "Hugepagesize:     2048 kB"
        );
    String stat1Content = StringUtilities.joinLines(
        "cpu 29793342 260290 3479274 636259369 6683218 656426 714057 0",
        "cpu0 29793342 260290 3479274 636259369 6683218 656426 714057 0",
        "intr 2870488853 2486517107 3 0 0 2 0 5 0 0 0 0 0 3 0 74363716 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " +
        "0 0 52483586 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 973315 0 0 0 0 0 0 0 46 " +
        "0 0 0 0 0 0 0 98792358 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 114339590 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 43019122 0 0 0 0 0",
        "ctxt 15053799843",
        "btime 1199289688",
        "processes 25799993",
        "procs_running 1",
        "procs_blocked 0"
        );
    String stat2Content = StringUtilities.joinLines(
        "cpu 29794509 260290 3479474 636287862 6683283 656450 714087 0 0",
        "cpu0 29794509 260290 3479474 636287862 6683283 656450 714087 0 0",
        "intr 2870488853 2486517107 3 0 0 2 0 5 0 0 0 0 0 3 0 74363716 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " +
        "0 0 52483586 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 973315 0 0 0 0 0 0 0 46 " +
        "0 0 0 0 0 0 0 98792358 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 114339590 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " +
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 43019122 0 0 0 0 0",
        "ctxt 15053799843",
        "btime 1199289688",
        "processes 25799993",
        "procs_running 1",
        "procs_blocked 0"
        );
    String meminfoFile =
        scratch.file("test_meminfo_nonht", meminfoContent).getPathString();
    String stat1File =
      scratch.file("proc_stat_1", stat1Content).getPathString();
    String stat2File =
      scratch.file("proc_stat_2", stat2Content).getPathString();
    assertEquals(1, LocalHostCapacity.getLogicalCpuCount(cpuinfoContent));
    assertEquals(1, LocalHostCapacity.getPhysicalCpuCount(cpuinfoContent, 1));
    assertEquals(1, LocalHostCapacity.getCoresPerCpu(cpuinfoContent));
    ResourceSet capacity =
        LocalHostCapacity.getLocalHostCapacity(cpuinfoFile, meminfoFile);
    assertEquals(1.0, capacity.getCpuUsage(), 0.01);
    assertEquals(3091.732, capacity.getMemoryMb(), 0.1); // +/- 0.1MB
    LocalHostCapacity.setLocalHostCapacity(capacity);
    assertSame(capacity, LocalHostCapacity.getLocalHostCapacity());
    Clock mockedClock = new Clock() {
      private int callCount = 0;

      @Override
      public long currentTimeMillis() {
        throw new AssertionError("unexpected method call");
      }

      @Override
      public long nanoTime() {
        callCount++;
        if (callCount == 1) {
          return 0;
        } else if (callCount == 2) {
          return 100 * 1000000;
        } else if (callCount == 3) {
          return 200 * 1000000;
        } else {
          throw new AssertionError("unexpected method call");
        }
      }
    };
    LocalHostCapacity.FreeResources freeStats =
      LocalHostCapacity.getFreeResources(mockedClock, meminfoFile, stat1File, null);
    assertNotNull(freeStats);
    assertEquals(2356.756, freeStats.getFreeMb(), 0.001);
    assertEquals(0.0, freeStats.getAvgFreeCpu(), 0);
    // The next call to the mock clock returns a timestamp as if 100 ms have passed.
    assertTrue(freeStats.getReadingAge() > 50);
    // Fake another 100 ms going by for the next call.
    freeStats = LocalHostCapacity.getFreeResources(mockedClock, meminfoFile, stat2File, freeStats);
    assertNotNull(freeStats);
    assertEquals(2356.756, freeStats.getFreeMb(), 0.001);
    assertTrue(freeStats.getInterval() > 100);
    assertEquals(0.95, freeStats.getAvgFreeCpu(), 0.001);
  }

  @Test
  public void testHyperthreadedMachine() throws Exception {
    String cpuinfoContent = StringUtilities.joinLines(
        "processor\t: 0",
        "vendor_id\t: GenuineIntel",
        "cpu family\t: 15",
        "model\t\t: 4",
        "model name\t:               Intel(R) Pentium(R) 4 CPU 3.40GHz",
        "stepping\t: 1",
        "cpu MHz\t\t: 3400.245",
        "cache size\t: 1024 KB",
        "physical id\t: 0",
        "siblings\t: 2",
        "core id\t\t: 0",
        "cpu cores\t: 1",
        "fpu\t\t: yes",
        "fpu_exception\t: yes",
        "cpuid level\t: 5",
        "wp\t\t: yes",
        "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge "
            + "mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm "
            + "syscall lm constant_tsc pni monitor ds_cpl cid cx16 xtpr",
        "bogomips\t: 6806.31",
        "clflush size\t: 64",
        "cache_alignment\t: 128",
        "address sizes\t: 36 bits physical, 48 bits virtual",
        "power management:",
        "",
        "processor\t: 1",
        "vendor_id\t: GenuineIntel",
        "cpu family\t: 15",
        "model\t\t: 4",
        "model name\t:               Intel(R) Pentium(R) 4 CPU 3.40GHz",
        "stepping\t: 1",
        "cpu MHz\t\t: 3400.245",
        "cache size\t: 1024 KB",
        "physical id\t: 0",
        "siblings\t: 2",
        "core id\t\t: 0",
        "cpu cores\t: 1",
        "fpu\t\t: yes",
        "fpu_exception\t: yes",
        "cpuid level\t: 5",
        "wp\t\t: yes",
        "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge "
            + "mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm "
            + "syscall lm constant_tsc pni monitor ds_cpl cid cx16 xtpr",
        "bogomips\t: 6800.76",
        "clflush size\t: 64",
        "cache_alignment\t: 128",
        "address sizes\t: 36 bits physical, 48 bits virtual",
        "power management:",
        ""
        );
    String cpuinfoFile =
        scratch.file("test_cpuinfo_ht", cpuinfoContent).getPathString();
    String meminfoContent = StringUtilities.joinLines(
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
        "Hugepagesize:     2048 kB"
        );
    String meminfoFile =
        scratch.file("test_meminfo_ht", meminfoContent).getPathString();
    assertEquals(2, LocalHostCapacity.getLogicalCpuCount(cpuinfoContent));
    assertEquals(1, LocalHostCapacity.getPhysicalCpuCount(cpuinfoContent, 2));
    assertEquals(1, LocalHostCapacity.getCoresPerCpu(cpuinfoContent));
    ResourceSet capacity =
        LocalHostCapacity.getLocalHostCapacity(cpuinfoFile, meminfoFile);
    assertEquals(1.2, capacity.getCpuUsage(), .0001);
    assertEquals(3092.004, capacity.getMemoryMb(), 0.1); // +/- 0.1MB
  }

  @Test
  public void testAMDMachine() throws Exception {
    String cpuinfoContent = StringUtilities.joinLines(
        "processor\t: 0",
        "vendor_id\t: AuthenticAMD",
        "cpu family\t: 15",
        "model\t\t: 65",
        "model name\t: Dual-Core AMD Opteron(tm) Processor 8214 HE",
        "stepping\t: 2",
        "cpu MHz\t\t: 2200.000",
        "cache size\t: 1024 KB",
        "physical id\t: 0",
        "siblings\t: 2",
        "core id\t\t: 0",
        "cpu cores\t: 2",
        "fpu\t\t: yes",
        "fpu_exception\t: yes",
        "cpuid level\t: 1",
        "wp\t\t: yes",
        "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr "
            + "pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall "
            + "nx mmxext fxsr_opt rdtscp lm 3dnowext 3dnow pni cx16 lahf_lm "
            + "cmp_legacy svm cr8_legacy",
        "bogomips\t: 4425.84",
        "TLB size\t: 1024 4K pages",
        "clflush size\t: 64",
        "cache_alignment\t: 64",
        "address sizes\t: 40 bits physical, 48 bits virtual",
        "power management: ts fid vid ttp tm stc",
        "",
        "processor\t: 1",
        "vendor_id\t: AuthenticAMD",
        "cpu family\t: 15",
        "model\t\t: 65",
        "model name\t: Dual-Core AMD Opteron(tm) Processor 8214 HE",
        "stepping\t: 2",
        "cpu MHz\t\t: 2200.000",
        "cache size\t: 1024 KB",
        "physical id\t: 0",
        "siblings\t: 2",
        "core id\t\t: 1",
        "cpu cores\t: 2",
        "fpu\t\t: yes",
        "fpu_exception\t: yes",
        "cpuid level\t: 1",
        "wp\t\t: yes",
        "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr "
            + "pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall "
            + "nx mmxext fxsr_opt rdtscp lm 3dnowext 3dnow pni cx16 lahf_lm "
            + "cmp_legacy svm cr8_legacy",
        "bogomips\t: 4460.61",
        "TLB size\t: 1024 4K pages",
        "clflush size\t: 64",
        "cache_alignment\t: 64",
        "address sizes\t: 40 bits physical, 48 bits virtual",
        "power management: ts fid vid ttp tm stc",
        "",
        "processor\t: 2",
        "vendor_id\t: AuthenticAMD",
        "cpu family\t: 15",
        "model\t\t: 65",
        "model name\t: Dual-Core AMD Opteron(tm) Processor 8214 HE",
        "stepping\t: 2",
        "cpu MHz\t\t: 2200.000",
        "cache size\t: 1024 KB",
        "physical id\t: 1",
        "siblings\t: 2",
        "core id\t\t: 0",
        "cpu cores\t: 2",
        "fpu\t\t: yes",
        "fpu_exception\t: yes",
        "cpuid level\t: 1",
        "wp\t\t: yes",
        "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr "
            + "pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall "
            + "nx mmxext fxsr_opt rdtscp lm 3dnowext 3dnow pni cx16 lahf_lm "
            + "cmp_legacy svm cr8_legacy",
        "bogomips\t: 4420.45",
        "TLB size\t: 1024 4K pages",
        "clflush size\t: 64",
        "cache_alignment\t: 64",
        "address sizes\t: 40 bits physical, 48 bits virtual",
        "power management: ts fid vid ttp tm stc",
        "",
        "processor\t: 3",
        "vendor_id\t: AuthenticAMD",
        "cpu family\t: 15",
        "model\t\t: 65",
        "model name\t: Dual-Core AMD Opteron(tm) Processor 8214 HE",
        "stepping\t: 2",
        "cpu MHz\t\t: 2200.000",
        "cache size\t: 1024 KB",
        "physical id\t: 1",
        "siblings\t: 2",
        "core id\t\t: 1",
        "cpu cores\t: 2",
        "fpu\t\t: yes",
        "fpu_exception\t: yes",
        "cpuid level\t: 1",
        "wp\t\t: yes",
        "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr "
            + "pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall "
            + "nx mmxext fxsr_opt rdtscp lm 3dnowext 3dnow pni cx16 lahf_lm "
            + "cmp_legacy svm cr8_legacy",
        "bogomips\t: 4460.39",
        "TLB size\t: 1024 4K pages",
        "clflush size\t: 64",
        "cache_alignment\t: 64",
        "address sizes\t: 40 bits physical, 48 bits virtual",
        "power management: ts fid vid ttp tm stc",
        ""
        );
    String cpuinfoFile =
        scratch.file("test_cpuinfo_amd", cpuinfoContent).getPathString();
    String meminfoContent = StringUtilities.joinLines(
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
    String meminfoFile =
        scratch.file("test_meminfo_amd", meminfoContent).getPathString();
    assertEquals(4, LocalHostCapacity.getLogicalCpuCount(cpuinfoContent));
    assertEquals(2, LocalHostCapacity.getPhysicalCpuCount(cpuinfoContent, 4));
    assertEquals(2, LocalHostCapacity.getCoresPerCpu(cpuinfoContent));
    ResourceSet capacity =
        LocalHostCapacity.getLocalHostCapacity(cpuinfoFile, meminfoFile);
    assertEquals(capacity.getCpuUsage(), 4.0, 0.01);
    assertEquals(8223.956, capacity.getMemoryMb(), 0.1); // +/- 0.1MB
  }
}
