package com.google.devtools.build.lib.sandbox.cgroups;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.util.FsApparatus;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.Test;

public class HierarchyTest {
    private final FsApparatus scratch = FsApparatus.newNative();

    @Test
    public void testParse() throws IOException {
        List<Hierarchy> hierarchies = Hierarchy.parse(
            scratch.file(
                "proc/self/cgroup",
                "1:cpu,cpuacct:/user.slice",
                "0::/user.slice/session-1.scope").getPathFile());

        assertThat(hierarchies).hasSize(2);
        assertThat(hierarchies.get(0).isV2()).isFalse();
        assertThat(hierarchies.get(1).isV2()).isTrue();
        assertThat(hierarchies.get(0).controllers()).containsExactly("cpu", "cpuacct");
        assertThat(hierarchies.get(1).controllers()).containsExactly("");
        assertThat(hierarchies.get(0).path()).isEqualTo(Path.of("/user.slice"));
        assertThat(hierarchies.get(1).path()).isEqualTo(Path.of("/user.slice/session-1.scope"));
    }
}
