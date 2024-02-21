package com.google.devtools.build.lib.sandbox.cgroups.controller.v2;

import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class UnifiedMemory extends UnifiedController implements Controller.Memory {
    private final Path path;
    public UnifiedMemory(Path path) {
        this.path = path;
    }

    @Override
    public Path getPath() {
        return path;
    }

    @Override
    public Memory child(String name) throws IOException {
        return new UnifiedMemory(getChild(name));
    }

    @Override
    public void setMaxBytes(long bytes) throws IOException {
        Files.writeString(path.resolve("memory.max"), Long.toString(bytes));
    }

    @Override
    public long getMaxBytes() throws IOException {
        return Long.parseLong(Files.readString(path.resolve("memory.max")).trim());
    }
}
