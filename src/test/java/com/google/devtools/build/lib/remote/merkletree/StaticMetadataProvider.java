// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.merkletree;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import java.util.Map;
import javax.annotation.Nullable;

/** A {@link MetadataProvider} backed by static data */
class StaticMetadataProvider implements MetadataProvider {

  private final ImmutableMap<ActionInput, FileArtifactValue> metadata;

  public StaticMetadataProvider(Map<ActionInput, FileArtifactValue> metadata) {
    this.metadata = ImmutableMap.copyOf(metadata);
  }

  @Nullable
  @Override
  public FileArtifactValue getMetadata(ActionInput input) {
    return metadata.get(input);
  }

  @Nullable
  @Override
  public ActionInput getInput(String execPath) {
    throw new UnsupportedOperationException();
  }
}
