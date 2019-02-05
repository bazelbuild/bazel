// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.util;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Wrapper for calculating a BigInteger fingerprint for an object. This BigInteger has a maximum of
 * 128 bits (16 bytes).
 *
 * <p>We avoid blindly digesting {@link BigInteger} objects because they are likely already
 * appropriately smeared across our fingerprint range, and therefore composable more cheaply than
 * md5.
 */
public class BigIntegerFingerprint {
  private final Fingerprint fingerprint = new Fingerprint(DigestHashFunction.MD5);
  private final List<BigInteger> alreadySmearedFingerprints = new ArrayList<>();
  private boolean seenNull = false;

  public BigIntegerFingerprint addLong(long addition) {
    fingerprint.addLong(addition);
    return this;
  }

  public BigIntegerFingerprint addString(String string) {
    fingerprint.addString(string);
    return this;
  }

  public BigIntegerFingerprint addDigestedBytes(@Nullable byte[] bytes) {
    if (bytes == null) {
      seenNull = true;
      return this;
    }
    if (bytes.length == 32) {
      return addBigIntegerOrdered(new BigInteger(1, bytes));
    }
    fingerprint.addBytes(bytes);
    return this;
  }

  public BigIntegerFingerprint addBoolean(boolean bool) {
    fingerprint.addBoolean(bool);
    return this;
  }

  public BigIntegerFingerprint addPath(PathFragment pathFragment) {
    fingerprint.addPath(pathFragment);
    return this;
  }

  public BigIntegerFingerprint addBigIntegerOrdered(BigInteger bigInteger) {
    alreadySmearedFingerprints.add(bigInteger);
    // Make sure the ordering of this BigInteger with respect to the items added to the fingerprint
    // is reflected in the output. Because no other method calls #addSInt, we can use it as a
    // marker.
    fingerprint.addSInt(1);
    return this;
  }

  public BigIntegerFingerprint addNullableBigIntegerOrdered(@Nullable BigInteger bigInteger) {
    if (bigInteger == null) {
      seenNull = true;
      return this;
    }
    return addBigIntegerOrdered(bigInteger);
  }

  public BigInteger getFingerprint() {
    if (seenNull) {
      return null;
    }
    BigInteger fp = new BigInteger(1, fingerprint.digestAndReset());
    for (BigInteger bigInteger : alreadySmearedFingerprints) {
      fp = BigIntegerFingerprintUtils.composeOrdered(fp, bigInteger);
    }
    return fp;
  }

  public void reset() {
    fingerprint.digestAndReset();
    seenNull = false;
    alreadySmearedFingerprints.clear();
  }
}
