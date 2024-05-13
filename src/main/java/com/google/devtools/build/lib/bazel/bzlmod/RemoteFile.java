package com.google.devtools.build.lib.bazel.bzlmod;
import java.net.URL;
import java.util.List;

/**
 * For use with IndexRegistry and associated files. A simple pojo to track remote files that are
 * offered at multiple urls (mirrors) with a single integrity.
 * We split up the file here to simplify the dependency.
 */
class RemoteFile {

  RemoteFile(String integrity, List<URL> urls) {
    this.integrity = integrity;
    this.urls = urls;
  }

  List<URL> urls;
  String integrity;
}

