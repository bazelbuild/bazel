// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.Fingerprint;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link AdvertisedProviderSet}. */
@RunWith(JUnit4.class)
public class AdvertisedProviderSetTest {
  @Test
  public void fingerprintsMatchExactly() {
    // Demonstrates that fingerprints of some choice AdvertisedProviderSet instances are exactly we
    // expect them to be.
    //
    // If this test fails because the implementation of AdvertisedProviderSet#fingerprint has been
    // intentionally changed in a manner compatible with the properties listed in that method's
    // javadoc, then simply update the expectations here.

    assertThat(getFingerprint(AdvertisedProviderSet.ANY))
        .isEqualTo(
            HashCode.fromString(
                "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a"));
    assertThat(getFingerprint(AdvertisedProviderSet.EMPTY))
        .isEqualTo(
            HashCode.fromString(
                "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"));
    assertThat(getFingerprint(AdvertisedProviderSet.builder().addBuiltin(String.class).build()))
        .isEqualTo(
            HashCode.fromString(
                "d93c0c71b65c3af6dd7fb823680bb768ede3857abb3046ebb4762f5a3e8793dc"));
    assertThat(getFingerprint(AdvertisedProviderSet.builder().addBuiltin(Integer.class).build()))
        .isEqualTo(
            HashCode.fromString(
                "979d311543b348225618bd436c874e24cfe4b7a35c891cbfaf89ce69885981b4"));
    assertThat(
            getFingerprint(
                AdvertisedProviderSet.builder()
                    .addStarlark(StarlarkProviderIdentifier.forLegacy("legacyProvider1"))
                    .build()))
        .isEqualTo(
            HashCode.fromString(
                "332f6fc411cfe63c5aaf1ed0fb06f979b6ce947239aede9b6e61649f9c543747"));
    assertThat(
            getFingerprint(
                AdvertisedProviderSet.builder()
                    .addStarlark(StarlarkProviderIdentifier.forLegacy("legacyProvider2"))
                    .build()))
        .isEqualTo(
            HashCode.fromString(
                "ba2f3a9271b4398d8c6bd8ae846446f13b6549be9d3fbce985358b709b343f80"));
    assertThat(
            getFingerprint(
                AdvertisedProviderSet.builder()
                    .addStarlark(
                        StarlarkProviderIdentifier.forKey(
                            new StarlarkProvider.Key(
                                Label.parseAbsoluteUnchecked("//my:label1.bzl"), "exportedName1")))
                    .build()))
        .isEqualTo(
            HashCode.fromString(
                "a17a99d19f39ae61e927da9169d42274b7121fa0fbc79bbb85cce7b199576a42"));
    assertThat(
            getFingerprint(
                AdvertisedProviderSet.builder()
                    .addStarlark(
                        StarlarkProviderIdentifier.forKey(
                            new StarlarkProvider.Key(
                                Label.parseAbsoluteUnchecked("//my:label1.bzl"), "exportedName1")))
                    .addStarlark(
                        StarlarkProviderIdentifier.forKey(
                            new StarlarkProvider.Key(
                                Label.parseAbsoluteUnchecked("//my:label2.bzl"), "exportedName2")))
                    .build()))
        .isEqualTo(
            HashCode.fromString(
                "2909013264231d50a3550d41e2eaf0fbd501bfffacf6303019cc8a93e187bf68"));
    assertThat(
            getFingerprint(
                AdvertisedProviderSet.builder()
                    .addStarlark(
                        StarlarkProviderIdentifier.forKey(
                            new StarlarkProvider.Key(
                                Label.parseAbsoluteUnchecked("//my:label2.bzl"), "exportedName2")))
                    .build()))
        .isEqualTo(
            HashCode.fromString(
                "4d5efc81e801feaeef6b5549d2c4b9d943ae673c76456f85631dc61cae698e4c"));
  }

  private static HashCode getFingerprint(AdvertisedProviderSet advertisedProviderSet) {
    Fingerprint fp = new Fingerprint();
    advertisedProviderSet.fingerprint(fp);
    return HashCode.fromBytes(fp.digestAndReset());
  }
}
