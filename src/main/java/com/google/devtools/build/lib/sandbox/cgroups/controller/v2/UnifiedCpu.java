package com.google.devtools.build.lib.sandbox.cgroups.controller.v2;

import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Scanner;

public class UnifiedCpu implements Controller.Cpu {
    private final Path path;

    public UnifiedCpu(Path path) {
        this.path = path;
    }

    @Override
    public Path getPath() {
        return path;
    }

    @Override
    public void setCpus(double cpus) throws IOException {
        long period;
        try (Scanner scanner = new Scanner(Files.newBufferedReader(path.resolve("cpu.max")))) {
            period = scanner.skip(".*\\s").nextInt();
        }
        long quota = Math.round(period * cpus);
        String limit = String.format("%d %d", quota, period);
        Files.writeString(path.resolve("cpu.max"), limit);
    }

    @Override
    public long getCpus() throws IOException {
        try (Scanner scanner = new Scanner(Files.newBufferedReader(path.resolve("cpu.max")))) {
          long quota = scanner.nextLong();
          long period = scanner.nextLong();
          return quota / period;
        }
    }
}
