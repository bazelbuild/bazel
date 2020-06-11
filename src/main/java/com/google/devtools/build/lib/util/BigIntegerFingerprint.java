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

import com.google.devtools.build.lib.vfs.PathFragment;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * Wrapper for calculating a BigInteger fingerprint for an object.
 *
 * <p>We avoid blindly digesting {@link BigInteger} objects because they are likely already
 * appropriately smeared across our fingerprint range, and therefore composable more cheaply than by
 * hashing.
 */
// TODO(b/150308424): Deprecate BigIntegerFingerprint
public class BigIntegerFingerprint {
  private static final UUID MARKER = UUID.fromString("28481318-fe19-454e-a60f-47922b398bca");
  private final Fingerprint fingerprint = new Fingerprint();
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
    // is reflected in the output. Use a UUID as a marker since extremely unlikely for there to
    // be a collision.
    // TODO(b/150308424): This class should just add a boolean in each add call:
    //   true here and false for all others. OR, this add is entirely unnecessary if the location
    //   of BigInteger adds are not data-dependent.
    fingerprint.addUUID(MARKER);
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
    // TODO(b/150312032): Is this still actually faster than hashing?
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
