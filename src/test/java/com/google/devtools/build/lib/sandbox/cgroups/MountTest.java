package com.google.devtools.build.lib.sandbox.cgroups;

import com.google.devtools.build.lib.vfs.util.FsApparatus;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.google.common.truth.Truth.assertThat;

public class MountTest {
    private final FsApparatus scratch = FsApparatus.newNative();

    @Test
    public void testParse() throws IOException {
        List<Mount> mounts = Mount.parse(
            scratch.file(
                "proc/self/mounts",
                "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
                "cgroup /dev/cgroup/cpu cgroup rw,cpu,cpuacct 0 0",
                "cgroup /dev/cgroup/unified cgroup2 ro 0 0",
                "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0").getPathFile());

        assertThat(mounts).hasSize(2);
        assertThat(mounts.get(0).isV2()).isFalse();
        assertThat(mounts.get(1).isV2()).isTrue();
        assertThat(mounts.get(0).opts()).contains("cpu");
        assertThat(mounts.get(0).path()).isEqualTo(Path.of("/dev/cgroup/cpu"));
        assertThat(mounts.get(1).path()).isEqualTo(Path.of("/dev/cgroup/unified"));
    }
}
