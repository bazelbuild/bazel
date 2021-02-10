package com.google.devtools.build.lib.bazel.repository.downloader;

import net.starlark.java.syntax.Location;

public class UrlRewriterParseException extends Exception {
    private Location location;

    public UrlRewriterParseException(String message, Location location) {
        super(message);
        this.location = location;
    }

    public Location getLocation() {
        return location;
    }
}
