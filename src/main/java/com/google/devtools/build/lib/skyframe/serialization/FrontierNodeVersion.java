// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.MoreObjects;
import com.google.common.hash.HashCode;
import com.google.common.primitives.Bytes;
import com.google.common.primitives.Longs;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId.SnapshotClientId;
import com.google.devtools.build.skyframe.IntVersion;
import java.util.Arrays;
import java.util.Optional;

/** A tuple representing the version of a cached SkyValue in the frontier. */
public final class FrontierNodeVersion {
  public static final FrontierNodeVersion CONSTANT_FOR_TESTING =
      new FrontierNodeVersion(
          "123",
          HashCode.fromInt(42),
          IntVersion.of(9000),
          "distinguisher",
          /* useFakeStampData= */ true,
          Optional.of(new SnapshotClientId("for_testing", 123)));

  /**
   * The checksum of the top-level configuration (trimmed of test options).
   *
   * <p>The configuration of any node in the build graph includes a mnemonic (e.g. k8-opt) that is
   * part of its output path. If a transition is applied, this mnemonic is computed based on the
   * delta between the node's configuration and the top-level configuration producing an ST-hash.
   *
   * <p>If the top-level configuration changes, the output paths of artifacts may change even if the
   * node's configuration remains distinct and identical (e.g. in a transition). Including this
   * checksum ensures that we do not reuse nodes that would produce artifacts at incorrect paths
   * relative to the current build's top-level configuration.
   *
   * <p>See b/360073915.
   */
  private final String topLevelConfigChecksum;

  private final byte[] topLevelConfigFingerprint;

  /**
   * The MD5 hash of the Bazel installation.
   *
   * <p>Ensures that cache entries are invalid if the Bazel binary itself changes (e.g. updated
   * version or locally modified binary). Different Bazel versions may produce different analysis
   * graphs from the same source code.
   */
  private final HashCode blazeInstallMD5;

  private final byte[] blazeInstallMD5Fingerprint;

  /**
   * The version of the source code (workspace) being evaluated.
   *
   * <p>This corresponds to the state of the BUILD files, .bzl files, and source files. Any change
   * to the source code likely changes the analysis graph, so this version is critical for
   * correctness.
   */
  private final long evaluatingVersion;

  private final byte[] evaluatingVersionFingerprint;

  /**
   * A distinguisher used to separate cache entries for different test cases or scenarios.
   *
   * <p>Allows integration tests to share a single cache backend without collision, or to force
   * specific cache keys for testing purposes.
   */
  private final byte[] distinguisherBytesForTesting;

  /** Whether this invocations use fake data for stamping (volatile) information. */
  private final boolean useFakeStampData;

  /**
   * The precomputed fingerprint of this node version.
   *
   * <p>This is the concatenation of the fingerprints of the other fields, providing a single hash
   * value for the entire version.
   */
  private final byte[] precomputedFingerprint;

  /**
   * A pointer to the specific workspace snapshot in the remote system.
   *
   * <p>This is NOT part of the cache key identity (hash/equals). It is used to retrieve
   * invalidation data or metadata associated with the specific state corresponding to {@link
   * #evaluatingVersion}.
   */
  private final Optional<ClientId> clientId;

  public FrontierNodeVersion(
      String topLevelConfigChecksum,
      HashCode blazeInstallMD5,
      IntVersion evaluatingVersion,
      String distinguisherBytesForTesting,
      boolean useFakeStampData,
      Optional<ClientId> clientId) {
    this.topLevelConfigChecksum = topLevelConfigChecksum;
    this.topLevelConfigFingerprint = topLevelConfigChecksum.getBytes(UTF_8);
    this.blazeInstallMD5 = blazeInstallMD5;
    this.blazeInstallMD5Fingerprint = blazeInstallMD5.asBytes();
    this.evaluatingVersion = evaluatingVersion.getVal();
    this.evaluatingVersionFingerprint = Longs.toByteArray(evaluatingVersion.getVal());
    this.distinguisherBytesForTesting = distinguisherBytesForTesting.getBytes(UTF_8);
    this.useFakeStampData = useFakeStampData;
    this.precomputedFingerprint =
        Bytes.concat(
            this.topLevelConfigFingerprint,
            this.blazeInstallMD5Fingerprint,
            this.evaluatingVersionFingerprint,
            this.distinguisherBytesForTesting,
            this.useFakeStampData ? new byte[] {1} : new byte[] {0});

    // This is undigested.
    this.clientId = clientId;
  }

  /**
   * Returns the snapshot of the workspace.
   *
   * <p>Can be empty if snapshots are not supported by the workspace.
   */
  @SuppressWarnings("unused") // to be integrated
  public Optional<ClientId> getClientId() {
    return clientId;
  }

  public byte[] getTopLevelConfigFingerprint() {
    return topLevelConfigFingerprint;
  }

  public byte[] getPrecomputedFingerprint() {
    return precomputedFingerprint;
  }

  public byte[] concat(byte[] input) {
    return Bytes.concat(precomputedFingerprint, input);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("topLevelConfig", Arrays.hashCode(topLevelConfigFingerprint))
        .add("blazeInstall", Arrays.hashCode(blazeInstallMD5Fingerprint))
        .add("evaluatingVersion", Arrays.hashCode(evaluatingVersionFingerprint))
        .add("distinguisherBytesForTesting", Arrays.hashCode(distinguisherBytesForTesting))
        .add("useFakeStampData", useFakeStampData)
        .add("precomputed", hashCode())
        .toString();
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(precomputedFingerprint);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof FrontierNodeVersion that)) {
      return false;
    }
    return Arrays.equals(precomputedFingerprint, that.precomputedFingerprint);
  }

  public HashCode getBlazeInstallMD5() {
    return blazeInstallMD5;
  }

  public long getEvaluatingVersion() {
    return evaluatingVersion;
  }

  public boolean getUseFakeStampData() {
    return useFakeStampData;
  }

  public String getTopLevelConfigChecksum() {
    return topLevelConfigChecksum;
  }
}
