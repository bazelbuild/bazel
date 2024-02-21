package com.google.devtools.build.lib.sandbox.cgroups.controller.v1;

import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;

import java.io.IOException;
import java.nio.file.Path;

abstract class LegacyController implements Controller {
    protected Path getChild(String name) throws IOException {
        Path path = getPath().resolve(name);
        path.toFile().mkdirs();
        return path;
    }
}
