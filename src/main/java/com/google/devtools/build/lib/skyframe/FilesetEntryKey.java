// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.FilesetTraversalParams;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.stream.StreamSupport;

/** A {@link SkyKey} for the {@link FilesetEntryFunction} */
@AutoValue
public abstract class FilesetEntryKey implements SkyKey {
  private static final Interner<FilesetEntryKey> INTERNER = BlazeInterners.newWeakInterner();

  abstract FilesetTraversalParams params();

  @Override
  public FilesetTraversalParams argument() {
    return params();
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.FILESET_ENTRY;
  }

  public static FilesetEntryKey key(FilesetTraversalParams param) {
    return INTERNER.intern(new AutoValue_FilesetEntryKey(param));
  }

  public static ImmutableList<FilesetEntryKey> keys(Iterable<FilesetTraversalParams> params) {
    return StreamSupport.stream(params.spliterator(), /*parallel=*/ false)
        .map(FilesetEntryKey::key)
        .collect(ImmutableList.toImmutableList());
  }
}
