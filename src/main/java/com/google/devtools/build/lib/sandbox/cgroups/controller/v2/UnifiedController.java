package com.google.devtools.build.lib.sandbox.cgroups.controller.v2;

import com.google.common.collect.Streams;
import com.google.common.io.CharSink;
import com.google.common.io.Files;
import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Scanner;

abstract class UnifiedController implements Controller {
    @Override
    public boolean isLegacy() {
        return false;
    }

    protected Path getChild(String name) throws IOException {
        File subtree = getPath().resolve("cgroup.subtree_control").toFile();
        File controllers = getPath().resolve("cgroup.controllers").toFile();
        if (subtree.canWrite() && controllers.canRead()) {
            CharSink sink = Files.asCharSink(subtree, StandardCharsets.UTF_8);
            try (Scanner scanner = new Scanner(controllers)) {
                sink.writeLines(Streams.stream(scanner).map(c -> "+" + c), " ");
            }
        }
        Path path = getPath().resolve(name);
        path.toFile().mkdirs();
        return path;
    }
}
