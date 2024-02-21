package com.google.devtools.build.lib.sandbox.cgroups;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Streams;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.CharSink;
import com.google.common.io.FileWriteMode;
import com.google.common.io.Files;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;
import com.google.devtools.build.lib.sandbox.cgroups.controller.v1.LegacyCpu;
import com.google.devtools.build.lib.sandbox.cgroups.controller.v1.LegacyMemory;
import com.google.devtools.build.lib.sandbox.cgroups.controller.v2.UnifiedCpu;
import com.google.devtools.build.lib.sandbox.cgroups.controller.v2.UnifiedMemory;

import javax.annotation.Nullable;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Scanner;
import java.util.concurrent.ConcurrentLinkedQueue;


/**
 * This class creates and exposes a virtual cgroup for the bazel process and allows creating
 * child cgroups. Resources are exposed as {@link Controller}s, each representing a
 * subsystem within the virtual cgroup and that could in theory belong to different real cgroups.
 */
@AutoValue
public abstract class VirtualCGroup {
    private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
    private static final File PROC_SELF_MOUNTS_PATH = new File("/proc/self/mounts");
    private static final File PROC_SELF_CGROUP_PATH = new File("/proc/self/cgroup");

    @Nullable
    private static volatile VirtualCGroup instance;

    @Nullable
    public abstract Controller.Cpu cpu();
    @Nullable
    public abstract Controller.Memory memory();
    public abstract ImmutableSet<Path> paths();

    private final Queue<VirtualCGroup> children = new ConcurrentLinkedQueue<>();

    public static VirtualCGroup getInstance() {
        if (instance == null) {
            synchronized (VirtualCGroup.class) {
                if (instance == null) {
                    try {
                        instance = create().child("bazel_" + ProcessHandle.current().pid() + ".slice");
                    } catch (IOException e) {
                        logger.atInfo().withCause(e).log("Failed to create root cgroup");
                        instance = NULL;
                    }
                    Runtime.getRuntime().addShutdownHook(new Thread(VirtualCGroup::deleteInstance));
                }
            }
        }
        return instance;
    }

    public static void deleteInstance() {
        if (instance != null) {
            synchronized (VirtualCGroup.class) {
                if (instance != null) {
                    instance.delete();
                    instance = null;
                }
            }
        }
    }

    public static VirtualCGroup create() throws IOException {
        return create(PROC_SELF_MOUNTS_PATH, PROC_SELF_CGROUP_PATH);
    }

    static public VirtualCGroup NULL = new AutoValue_VirtualCGroup(null, null, ImmutableSet.of());

    public static VirtualCGroup create(File procMounts, File procCgroup) throws IOException {
        final List<Mount> mounts = Mount.parse(procMounts);
        final Map<String, Hierarchy> hierarchies = Hierarchy.parse(procCgroup)
            .stream()
            .flatMap(h -> h.controllers().stream().map(c -> Map.entry(c, h)))
            // For cgroup v2, there are no controllers specified in the proc/pid/cgroup file
            // So the keep will be empty and unique. For cgroup v1, there could potentially
            // be multiple mounting points for the same controller, but they represent a
            // "view of the same hierarchy" so it is ok to just keep one.
            // Ref. https://man7.org/linux/man-pages/man7/cgroups.7.html
            .collect(ImmutableMap.toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));
        return create(mounts, hierarchies);
    }

    public static VirtualCGroup create(List<Mount> mounts, Map<String, Hierarchy> hierarchies) throws IOException {
        Controller.Memory memory = null;
        Controller.Cpu cpu = null;
        ImmutableSet.Builder<Path> paths = ImmutableSet.builder();

        for (Mount m: mounts) {
            if (memory != null && cpu != null) break;

            if (m.isV2()) {
                Hierarchy h = hierarchies.get("");
                if (h == null) continue;
                Path cgroup = m.path().resolve(Paths.get("/").relativize(h.path()));
                if (!cgroup.equals(m.path())) {
                    // Because of the "no internal processes" rule, it is not possible to
                    // create a non-empty child cgroups on non-root cgroups with member processes
                    // Instead, we go up one level in the hierarchy and declare a sibling.
                    cgroup = cgroup.getParent();
                }
                if (!cgroup.toFile().canWrite()) {
                    logger.atInfo().log("Found non-writable cgroup v2 at %s", cgroup);
                    continue;
                }
                paths.add(cgroup);

                try (Scanner scanner = new Scanner(cgroup.resolve("cgroup.controllers").toFile())) {
                    while (scanner.hasNext()) {
                        switch (scanner.next()) {
                            case "memory":
                                if (memory != null) continue;
                                logger.atFine().log("Found v2 memory controller at %s", cgroup);
                                memory = new UnifiedMemory(cgroup);
                                break;
                            case "cpu":
                                if (cpu != null) continue;
                                logger.atFine().log("Found v2 cpu controller at %s", cgroup);
                                cpu = new UnifiedCpu(cgroup);
                                break;
                        }
                  }
                }
            } else {
                for (String opt : m.opts()) {
                    Hierarchy h = hierarchies.get(opt);
                    if (h == null) continue;
                    Path cgroup = m.path().resolve(Paths.get("/").relativize(h.path()));
                    if (!cgroup.toFile().canWrite()) {
                        logger.atInfo().log("Found non-writable cgroup v1 at %s", cgroup);
                        continue;
                    }
                    paths.add(cgroup);

                    switch (opt) {
                        case "memory":
                            if (memory != null) continue;
                            logger.atFine().log("Found v1 memory controller at %s", cgroup);
                            memory = new LegacyMemory(cgroup);
                            break;
                        case "cpu":
                            if (cpu != null) continue;
                            logger.atFine().log("Found v1 cpu controller at %s", cgroup);
                            cpu = new LegacyCpu(cgroup);
                            break;
                    }
                }
            }
        }

        VirtualCGroup vcgroup = new AutoValue_VirtualCGroup(cpu, memory, paths.build());
        return vcgroup;
    }

    public void delete() {
        this.children.forEach(VirtualCGroup::delete);
        this.paths().stream().map(Path::toFile).filter(File::exists).forEach(File::delete);
    }

    public VirtualCGroup child(String name) throws IOException {
        Controller.Cpu cpu = null;
        Controller.Memory memory = null;
        ImmutableSet.Builder<Path> paths = ImmutableSet.builder();
        if (memory() != null) {
            memory = memory().child(name);
            paths.add(memory.getPath());
        }
        if (cpu() != null) {
            cpu = cpu().child(name);
            paths.add(cpu.getPath());
        }
        VirtualCGroup child = new AutoValue_VirtualCGroup(cpu, memory, paths.build());
        this.children.add(child);
        return child;
    }

    public void addProcess(long pid) throws IOException {
        String pidStr = Long.toString(pid);
        for (Path path : paths()) {
            File procs = path.resolve("cgroup.procs").toFile();
            CharSink sink = Files.asCharSink(procs, StandardCharsets.UTF_8);
            sink.write(pidStr);
        }
    }
}
