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

package com.google.devtools.build.lib.cmdline;

import static java.util.Comparator.naturalOrder;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Table;
import java.util.Comparator;

/**
 * A {@link RepoMappingRecorder} that records forward lookups in a {@link Table} keyed by the
 * context repo of the mapping and the apparent name.
 */
public final class SimpleRepoMappingRecorder implements RepoMappingRecorder {
  private final Table<RepositoryName, String, RepositoryName> entries = HashBasedTable.create();

  @Override
  public void record(
      RepositoryMapping mapping, String apparentName, RepositoryName canonicalName) {
    entries.put(mapping.contextRepo(), apparentName, canonicalName);
  }

  /** Records all the given entries, as returned by {@link #recordedEntries}. */
  public void recordAll(Table<RepositoryName, String, RepositoryName> entries) {
    this.entries.putAll(entries);
  }

  public ImmutableTable<RepositoryName, String, RepositoryName> recordedEntries() {
    return ImmutableTable.<RepositoryName, String, RepositoryName>builder()
        .orderRowsBy(Comparator.comparing(RepositoryName::getName))
        .orderColumnsBy(naturalOrder())
        .putAll(entries)
        .buildOrThrow();
  }
}
