package com.google.devtools.build.lib.sandbox.cgroups;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.util.FsApparatus;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.List;
import java.util.Optional;

import static com.google.common.truth.Truth.assertThat;

public class VirtualCgroupFactoryTest {
    private final FsApparatus scratch = FsApparatus.newNative();

    private VirtualCGroup root;

    @Before
    public void setup() throws Exception {
        scratch.dir("cpu/cpu");
        scratch.dir("mem/mem");
        List<Mount> mounts = List.of(
            Mount.create(scratch.path("cpu").getPathFile().toPath(), "cgroup", List.of("cpu")),
            Mount.create(scratch.path("mem").getPathFile().toPath(), "cgroup", List.of("memory")));
        List<Hierarchy> hierarchies = List.of(
            Hierarchy.create(1, List.of("cpu"), scratch.path("/cpu").getPathFile().toPath()),
            Hierarchy.create(2, List.of("memory"), scratch.path("/mem").getPathFile().toPath()));

        root = VirtualCGroup.create(mounts, hierarchies);
        assertThat(root.cpu()).isNotNull();
        assertThat(root.memory()).isNotNull();
    }

    @Test
    public void testCreateNoLimits() throws IOException {
        ImmutableMap<String, Double> defaults = ImmutableMap.of();
        VirtualCGroupFactory factory = new VirtualCGroupFactory("nolimits", root, defaults, false);

        Optional<VirtualCGroup> vcg = factory.create(1, ImmutableMap.of());

        assertThat(vcg).isEmpty();
    }

    @Test
    public void testForceCreateNoLimits() throws IOException {
        ImmutableMap<String, Double> defaults = ImmutableMap.of();
        VirtualCGroupFactory factory = new VirtualCGroupFactory("nolimits", root, defaults, true);

        Optional<VirtualCGroup> vcg = factory.create(1, ImmutableMap.of());

        assertThat(vcg).isPresent();
        assertThat(vcg.get().cpu()).isNotNull();
        assertThat(vcg.get().memory()).isNotNull();
    }

    @Test
    public void testCreateWithDefaultLimits() throws IOException {
        ImmutableMap<String, Double> defaults = ImmutableMap.of("memory", 100.0);
        VirtualCGroupFactory factory = new VirtualCGroupFactory("defaults", root, defaults, false);

        Optional<VirtualCGroup> vcg = factory.create(1, ImmutableMap.of());

        assertThat(vcg).isPresent();
        assertThat(vcg.get().cpu()).isNotNull();
        assertThat(vcg.get().memory()).isNotNull();
        assertThat(vcg.get().memory().getMaxBytes()).isEqualTo(100 * 1024 * 1024);
    }

    @Test
    public void testCreateWithCustomLimits() throws IOException {
        scratch.file("cpu/cpu/custom1.scope/cpu.cfs_period_us", "1000");
        ImmutableMap<String, Double> defaults = ImmutableMap.of("memory", 100.0, "cpu", 1.0);
        VirtualCGroupFactory factory = new VirtualCGroupFactory("custom", root, defaults, false);

        Optional<VirtualCGroup> vcg = factory.create(1, ImmutableMap.of("memory", 200.0));

        assertThat(vcg).isPresent();
        assertThat(vcg.get().cpu()).isNotNull();
        assertThat(vcg.get().memory()).isNotNull();
        assertThat(vcg.get().cpu().getCpus()).isEqualTo(1);
        assertThat(vcg.get().memory().getMaxBytes()).isEqualTo(200 * 1024 * 1024);
    }

    @Test
    public void testCreateNull() throws IOException {
        ImmutableMap<String, Double> defaults = ImmutableMap.of("memory", 100.0, "cpu", 1.0);
        VirtualCGroupFactory factory =
            new VirtualCGroupFactory("null", VirtualCGroup.NULL, defaults, false);

        Optional<VirtualCGroup> vcg = factory.create(1, ImmutableMap.of());

        assertThat(vcg).isPresent();
        assertThat(vcg.get().cpu()).isNull();
        assertThat(vcg.get().memory()).isNull();
    }

    @Test
    public void testGet() throws IOException {
        ImmutableMap<String, Double> defaults = ImmutableMap.of("memory", 100.0);
        VirtualCGroupFactory factory = new VirtualCGroupFactory("get", root, defaults, false);

        Optional<VirtualCGroup> vcg = factory.create(1, ImmutableMap.of());

        assertThat(factory.get(1)).isEqualTo(vcg);
    }


    @Test
    public void testRemove() throws IOException {
        ImmutableMap<String, Double> defaults = ImmutableMap.of();
        VirtualCGroupFactory factory = new VirtualCGroupFactory("get", root, defaults, true);

        Optional<VirtualCGroup> vcg = factory.create(1, ImmutableMap.of());

        assertThat(factory.remove(1)).isEqualTo(vcg);
        assertThat(factory.get(1)).isEmpty();
    }
}
