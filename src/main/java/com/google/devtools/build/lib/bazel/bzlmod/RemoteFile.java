package com.google.devtools.build.lib.bazel.bzlmod;
import java.net.URL;
import java.util.List;

/**
 * For use with IndexRegistry and associated files. A simple pojo to track remote files that are
 * offered at multiple urls (mirrors) with a single integrity.
 * We split up the file here to simplify the dependency.
 */
record RemoteFile(String integrity, List<URL> urls) {
}

