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

package com.google.devtools.build.lib.rules.repository;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Computes the value of output hashes for the repositories specified in the resolved file
 * designated for this purpose.
 */
public class ResolvedHashesFunction implements SkyFunction {
  public static final String ORIGINAL_RULE_CLASS = "original_rule_class";
  public static final String ORIGINAL_ATTRIBUTES = "original_attributes";
  public static final String DEFINITION_INFORMATION = "definition_information";
  public static final String RULE_CLASS = "rule_class";
  public static final String ATTRIBUTES = "attributes";
  public static final String OUTPUT_TREE_HASH = "output_tree_hash";
  public static final String REPOSITORIES = "repositories";
  public static final String NATIVE = "native";

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, SkyFunctionException {

    Optional<RootedPath> resolvedFile =
        RepositoryDelegatorFunction.RESOLVED_FILE_FOR_VERIFICATION.get(env);
    if (resolvedFile == null) {
      return null;
    }
    if (!resolvedFile.isPresent()) {
      return new ResolvedHashesValue(ImmutableMap.<String, String>of());
    }
    ResolvedFileValue resolvedValue =
        (ResolvedFileValue) env.getValue(ResolvedFileValue.key(resolvedFile.get()));
    if (resolvedValue == null) {
      return null;
    }
    List<Map<String, Object>> resolved = resolvedValue.getResolvedValue();

    // Collect the hashes in a mutable map, to be able to detect duplicates and
    // only take the first entry, following the "maybe pattern" of external repositories,
    // adding a repository only if not already present.
    Map<String, String> hashes = new LinkedHashMap<String, String>();
    for (Map<String, Object> entry :  resolved) {
      Object repositories = entry.get(REPOSITORIES);
      if (repositories instanceof List) {
        for (Object repo : (List) repositories) {
          if (repo instanceof Map) {
            Object hash = ((Map) repo).get(OUTPUT_TREE_HASH);
            Object attributes = ((Map) repo).get(ATTRIBUTES);
            if (attributes instanceof Map) {
              Object name = ((Map) attributes).get("name");
              if ((name instanceof String) && (hash instanceof String)) {
                if (!hashes.containsKey((String) name)) {
                  hashes.put((String) name, (String) hash);
                }
              }
            }
          }
        }
      }
    }
    return new ResolvedHashesValue(ImmutableMap.copyOf(hashes));
  }

  @Override
  @Nullable
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
