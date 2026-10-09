// Copyright 2026 The Bazel Authors. All rights reserved.
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
// See the License for the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.actions.MetadataDigestUtils;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/** Computes digests of mandatory action inputs for split action cache checking. */
public final class MandatoryInputsDigestUtils {
  private MandatoryInputsDigestUtils() {}

  public static byte[] fromMetadataMap(Map<String, FileArtifactValue> inputMap) {
    // Persistent cache entries require a full-length mandatory digest. Keep the shared metadata
    // digest's zero identity unchanged: the input/output digest reuse optimization relies on it.
    return inputMap.isEmpty()
        ? new Fingerprint().digestAndReset()
        : MetadataDigestUtils.fromMetadata(inputMap);
  }

  public static byte[] fromMandatoryArtifacts(
      InputMetadataProvider inputMetadataProvider, Iterable<Artifact> mandatoryInputs)
      throws IOException {
    Map<String, FileArtifactValue> inputMap = new HashMap<>();
    for (Artifact artifact : mandatoryInputs) {
      inputMap.put(
          artifact.getExecPathString(), getInputMetadataMaybe(inputMetadataProvider, artifact));
    }
    return fromMetadataMap(inputMap);
  }

  @Nullable
  public static FileArtifactValue getInputMetadataMaybe(
      InputMetadataProvider inputMetadataProvider, Artifact artifact) {
    try {
      FileArtifactValue metadata = inputMetadataProvider.getInputMetadata(artifact);
      return (metadata != null && artifact.isConstantMetadata())
          ? FileArtifactValue.ConstantMetadataValue.INSTANCE
          : metadata;
    } catch (IOException e) {
      return null;
    }
  }
}
