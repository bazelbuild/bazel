// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import com.google.common.annotations.VisibleForTesting;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

/**
 * Represents a link mapping that acts as input to {@link RuleLinkExpander} and {@link
 * SourceUrlMapper}.
 */
public class DocLinkMap {
  // For RuleLinkExpander
  final String beRoot;
  final Map<String, String> beReferences; // Gson#fromJson ensures the map is of an ordered type

  // For SourceUrlMapper
  final String sourceUrlRoot;
  final Map<String, String> repoPathRewrites; // Gson#fromJson ensures the map is of an ordered type

  @VisibleForTesting
  DocLinkMap(
      String beRoot,
      Map<String, String> beReferences,
      String sourceUrlRoot,
      Map<String, String> repoPathRewrites) {
    this.beRoot = beRoot;
    this.beReferences = beReferences;
    this.sourceUrlRoot = sourceUrlRoot;
    this.repoPathRewrites = repoPathRewrites;
  }

  public static DocLinkMap createFromFile(String filePath) {
    try {
      return new Gson().fromJson(Files.readString(Paths.get(filePath)), DocLinkMap.class);
    } catch (IOException | JsonSyntaxException ex) {
      throw new IllegalArgumentException("Failed to read link map from " + filePath, ex);
    }
  }
}
