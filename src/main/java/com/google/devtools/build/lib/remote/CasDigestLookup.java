package com.google.devtools.build.lib.remote;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableSet;
import java.io.IOException;

/** A simple interface for lookup of CAS digests. */
public interface CasDigestLookup {
  ImmutableSet<Digest> getMissingDigests(Iterable<Digest> digests)
      throws IOException, InterruptedException;
}
