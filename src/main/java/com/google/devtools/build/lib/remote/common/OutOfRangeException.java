package com.google.devtools.build.lib.remote.common;

import build.bazel.remote.execution.v2.Digest;

import java.io.IOException;

/**
 * An exception to indicate the digest size exceeds the accepted limit set by the remote-cache.
 */
public final class OutOfRangeException extends IOException {

  public OutOfRangeException(String resourceName) {
    super(String.format("Resource %s size exceeds the limit by remote-cache.", resourceName));
  }
}
