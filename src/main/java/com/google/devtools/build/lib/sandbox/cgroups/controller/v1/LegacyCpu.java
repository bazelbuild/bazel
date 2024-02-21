package com.google.devtools.build.lib.sandbox.cgroups.controller.v1;

import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class LegacyCpu implements Controller.Cpu {
    private final Path path;

    public LegacyCpu(Path path) {
        this.path = path;
    }

    @Override
    public Path getPath() {
        return path;
    }

    @Override
    public void setCpus(double cpus) throws IOException {
        long period = Long.parseLong(Files.readString(path.resolve("cpu.cfs_period_us")).trim());
        long quota = Math.round(cpus * period);
        Files.writeString(path.resolve("cpu.cfs_quota_us"), Long.toString(quota));
    }

    @Override
    public long getCpus() throws IOException {
        long quota = Long.parseLong(Files.readString(path.resolve("cpu.cfs_quota_us")).trim());
        long period = Long.parseLong(Files.readString(path.resolve("cpu.cfs_period_us")).trim());
        return quota / period;
    }
}
