// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec.util;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesSupplier.RunfilesTree;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/** A fake implementation of the {@link InputMetadataProvider} interface. */
public final class FakeActionInputFileCache implements InputMetadataProvider {
  private final Map<ActionInput, FileArtifactValue> inputs = new HashMap<>();

  public FakeActionInputFileCache() {}

  public void put(ActionInput artifact, FileArtifactValue metadata) {
    inputs.put(artifact, metadata);
  }

  @Override
  public FileArtifactValue getInputMetadata(ActionInput input) throws IOException {
    return Preconditions.checkNotNull(inputs.get(input));
  }

  @Override
  @Nullable
  public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
    throw new UnsupportedOperationException();
  }

  @Override
  public ImmutableList<RunfilesTree> getRunfilesTrees() {
    throw new UnsupportedOperationException();
  }

  @Override
  public ActionInput getInput(String execPath) {
    throw new UnsupportedOperationException();
  }
}
