package com.google.devtools.build.lib.util;

import java.lang.ref.Cleaner;

/** Common {@link Cleaner} for Bazel. */
public final class BazelCleaner {
  public static final Cleaner CLEANER = Cleaner.create();

  private BazelCleaner() {}
}
