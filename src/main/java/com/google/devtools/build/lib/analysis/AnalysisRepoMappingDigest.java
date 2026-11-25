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
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;

/** Internal hash input for apparent repository names resolved during rule or aspect analysis. */
@AutoCodec
public record AnalysisRepoMappingDigest(String digest) implements TransitiveInfoProvider {
  static AnalysisRepoMappingDigest create(Label.SimpleRepoMappingRecorder recorder) {
    var mappings = recorder.recordedEntries();
    Fingerprint fingerprint = new Fingerprint().addInt(mappings.size());
    for (var mapping : mappings.cellSet()) {
      fingerprint
          .addString(mapping.getRowKey().getName())
          .addString(mapping.getColumnKey())
          .addString(mapping.getValue().getNameWithAt());
    }
    return new AnalysisRepoMappingDigest(fingerprint.hexDigestAndReset());
  }
}
