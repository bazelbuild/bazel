package com.google.devtools.build.lib.sandbox.cgroups.controller.v1;

import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class LegacyMemory extends LegacyController implements Controller.Memory {
    private final Path path;

    @Override
    public Path getPath() {
        return path;
    }

    public LegacyMemory(Path path) {
        this.path = path;
    }

    @Override
    public Memory child(String name) throws IOException {
        return new LegacyMemory(getChild(name));
    }

    @Override
    public void setMaxBytes(long bytes) throws IOException {
        Files.writeString(path.resolve("memory.limit_in_bytes"), Long.toString(bytes));
    }

    @Override
    public long getMaxBytes() throws IOException {
        return Long.parseLong(Files.readString(path.resolve("memory.limit_in_bytes")).trim());
    }

    @Override
    public long getUsageInBytes() throws IOException {
        return Long.parseLong(Files.readString(path.resolve("memory.usage_in_bytes")).trim());
    }
}
