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

package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterators;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Iterator;
import javax.annotation.Nullable;

/** List of fingerprints, each of which is a nullable {@link BigInteger}. Ordered and immutable. */
public class DepFingerprintList implements Iterable<BigInteger> {
  /**
   * Marker object indicating that dep fingerprints are not being tracked, and so there should be no
   * attempts to use this dep fingerprints object. Clients can compare their {@code
   * DepFingerprintList} object to this one using reference equality to see if they are tracking dep
   * fingerprints. This object should never be passed in as a valid {@code DepFingerprintList}
   * object.
   */
  public static final DepFingerprintList NOT_TRACKING_MARKER = new NoDepGroupsFingerprintMarker();

  private static final DepFingerprintList EMPTY_DEP_GROUPS_FINGERPRINT =
      new DepFingerprintList(new BigInteger[0]);

  private final BigInteger[] fingerprints;

  private DepFingerprintList(BigInteger[] fingerprints) {
    this.fingerprints = fingerprints;
  }

  /**
   * Returns the {@link BigInteger} for the dep group at {@code index}. A null value indicates that
   * a fingerprint could not be computed for this group, so that equality-by-dep-group checking
   * cannot be performed for this group.
   */
  @Nullable
  public BigInteger get(int index) {
    return fingerprints[index];
  }

  @Override
  public Iterator<BigInteger> iterator() {
    return Iterators.forArray(fingerprints);
  }

  @Override
  public String toString() {
    return "DepFingerprintList{" + Arrays.toString(fingerprints) + "}";
  }

  private static class NoDepGroupsFingerprintMarker extends DepFingerprintList {
    private NoDepGroupsFingerprintMarker() {
      super(null);
    }

    @Override
    public BigInteger get(int index) {
      throw new UnsupportedOperationException("No dep groups fingerprint (" + index + ")");
    }

    @Override
    public Iterator<BigInteger> iterator() {
      throw new UnsupportedOperationException("No dep groups fingerprint");
    }

    @Override
    public String toString() {
      return "NoDepGroupsFingerprintMarker";
    }
  }

  /** Builder for {@code DepFingerprintList}. */
  public static class Builder {
    private int curIndex = 0;
    private final BigInteger[] fingerprints;

    public Builder(int size) {
      fingerprints = new BigInteger[size];
    }

    public void add(BigInteger fingerprint) {
      fingerprints[curIndex++] = fingerprint;
    }

    public DepFingerprintList build() {
      Preconditions.checkState(
          curIndex == fingerprints.length, "%s %s", curIndex, fingerprints.length);
      if (fingerprints.length == 0) {
        return EMPTY_DEP_GROUPS_FINGERPRINT;
      }
      return new DepFingerprintList(fingerprints);
    }
  }
}
