package com.google.devtools.build.lib.sandbox.cgroups;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;

import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

public class VirtualCGroupFactory {
    private final String name;
    private final ImmutableMap<String, Double> defaultLimits;
    private final VirtualCGroup root;
    private final ConcurrentHashMap<Integer, VirtualCGroup> cgroups;
    private final boolean alwaysCreate;

    public VirtualCGroupFactory(String name, VirtualCGroup root, ImmutableMap<String, Double> defaultLimits, boolean alwaysCreate) {
        this.name = Preconditions.checkNotNull(name);
        this.defaultLimits = Preconditions.checkNotNull(defaultLimits);
        this.root = Preconditions.checkNotNull(root);
        this.alwaysCreate = alwaysCreate;
        this.cgroups = new ConcurrentHashMap<>();
    }

    public Optional<VirtualCGroup> create(Integer id, ImmutableMap<String, Double> limits) throws IOException {
        if (!alwaysCreate && defaultLimits.isEmpty() && limits.isEmpty())
            return Optional.empty();

        VirtualCGroup cgroup = null;

        Double cpuLimit = limits.getOrDefault("cpu", defaultLimits.getOrDefault("cpu", 0.0));
        Double memoryLimit =
            limits.getOrDefault("memory", defaultLimits.getOrDefault("memory", 0.0)) * 1024 * 1024;

        if (alwaysCreate || cpuLimit != 0 || memoryLimit != 0) {
            cgroup = root.child(this.name + id + ".scope");
            cgroups.put(id, cgroup);
        }
        if (memoryLimit > 0 && cgroup.memory() != null)
            cgroup.memory().setMaxBytes(memoryLimit.longValue());
        if (cpuLimit > 0 && cgroup.cpu() != null)
            cgroup.cpu().setCpus(cpuLimit);

        return Optional.ofNullable(cgroup);
    }

    public Optional<VirtualCGroup> get(Integer id) {
        return Optional.ofNullable(cgroups.get(id));
    }

    public Optional<VirtualCGroup> remove(Integer id) {
        return Optional.ofNullable(cgroups.remove(id));
    }
}
