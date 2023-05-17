package com.google.devtools.build.lib.hash;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.hash.Hasher;
import com.google.errorprone.annotations.Immutable;

@Immutable
public final class Blake3HashFunction extends AbstractHashFunction {
  @Override
  public Hasher newHasher(int expectedInputSize) {
    checkArgument(
        expectedInputSize >= 0, "expectedInputSize must be >= 0 but was %s", expectedInputSize);
    return newHasher();
  }

  @Override
  public int bits() {
    return 256;
  }

  @Override
  public Hasher newHasher() {
    return new Blake3Hasher();
  }
}
