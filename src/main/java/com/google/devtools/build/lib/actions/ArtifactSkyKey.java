// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * A {@link SkyKey} coming from an {@link Artifact} that is not mandatory: absence of the Artifact
 * does not imply any error.
 *
 * <p>Since {@link Artifact} is already a {@link SkyKey}, this wrapper is only needed when the
 * {@link Artifact} is not mandatory (discovered during include scanning).
 */
// TODO(janakr): pull mandatory/non-mandatory handling up to consumers and get rid of this wrapper.
@AutoCodec
public class ArtifactSkyKey implements SkyKey {
  private static final Interner<ArtifactSkyKey> INTERNER = BlazeInterners.newWeakInterner();

  private final Artifact.SourceArtifact artifact;

  private ArtifactSkyKey(Artifact.SourceArtifact sourceArtifact) {
    this.artifact = Preconditions.checkNotNull(sourceArtifact);
  }

  @ThreadSafety.ThreadSafe
  public static SkyKey key(Artifact artifact, boolean isMandatory) {
    if (isMandatory || !artifact.isSourceArtifact()) {
      return artifact;
    }
    return create(((Artifact.SourceArtifact) artifact));
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static ArtifactSkyKey create(Artifact.SourceArtifact artifact) {
    return INTERNER.intern(new ArtifactSkyKey(artifact));
  }

  public static Artifact artifact(SkyKey key) {
    return key instanceof Artifact ? (Artifact) key : ((ArtifactSkyKey) key).artifact;
  }

  public static boolean isMandatory(SkyKey key) {
    return key instanceof Artifact;
  }

  @Override
  public SkyFunctionName functionName() {
    return Artifact.ARTIFACT;
  }

  @Override
  public int hashCode() {
    return artifact.hashCode();
  }

  @Override
  public boolean equals(Object that) {
    if (this == that) {
      return true;
    }
    if (!(that instanceof ArtifactSkyKey)) {
      return false;
    }
    ArtifactSkyKey thatArtifactSkyKey = ((ArtifactSkyKey) that);
    return artifact.equals(thatArtifactSkyKey.artifact);
  }

  public Artifact getArtifact() {
    return artifact;
  }

  @Override
  public String toString() {
    return "ArtifactSkyKey:" + artifact.prettyPrint() + " " + artifact.getArtifactOwner();
  }
}
