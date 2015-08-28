// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules.test;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

import java.util.Map;

/**
 * An implementation class for the InstrumentedFilesProvider interface.
 */
public final class InstrumentedFilesProviderImpl implements InstrumentedFilesProvider {
  public static final InstrumentedFilesProvider EMPTY = new InstrumentedFilesProviderImpl(
      NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
      NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
      ImmutableMap.<String, String>of());

  private final NestedSet<Artifact> instrumentedFiles;
  private final NestedSet<Artifact> instrumentationMetadataFiles;
  private final ImmutableMap<String, String> extraEnv;

  public InstrumentedFilesProviderImpl(NestedSet<Artifact> instrumentedFiles,
      NestedSet<Artifact> instrumentationMetadataFiles, Map<String, String> extraEnv) {
    this.instrumentedFiles = instrumentedFiles;
    this.instrumentationMetadataFiles = instrumentationMetadataFiles;
    this.extraEnv = ImmutableMap.copyOf(extraEnv);
  }

  @Override
  public NestedSet<Artifact> getInstrumentedFiles() {
    return instrumentedFiles;
  }

  @Override
  public NestedSet<Artifact> getInstrumentationMetadataFiles() {
    return instrumentationMetadataFiles;
  }

  @Override
  public Map<String, String> getExtraEnv() {
    return extraEnv;
  }
}
