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
package com.google.devtools.build.lib.rules.cpp;

import static java.lang.Math.max;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStream;//?
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

@Immutable
public final class CppHeaderMapAction extends AbstractFileWriteAction {

  private static final String GUID = "3f9eb099-e62d-4d3b-a08a-c478e188993d";

  // C++ header map of the current target
  private final CppHeaderMap cppHeaderMap;
  // Data required to build the actual header map
  // NOTE: If you add a field here, you'll likely need to add it to the cache key in computeKey().
  private final ImmutableList<CppHeaderMap> dependencies;
  private final String includePrefix = "";

  public CppHeaderMapAction(
      ActionOwner owner,
      CppHeaderMap cppHeaderMap,
      Iterable<CppHeaderMap> dependencies
  ) {
    super(
        owner,
        ImmutableList.<Artifact>builder()
        .addAll(Iterables.filter(privateHeaders, Artifact::isTreeArtifact))
        .addAll(Iterables.filter(publicHeaders, Artifact::isTreeArtifact))
        .build(),
        cppHeaderMap.getArtifact(),
        /*makeExecutable=*/ false);
    this.cppHeaderMap = cppHeaderMap;
    this.privateHeaders = ImmutableList.copyOf(privateHeaders);
    this.publicHeaders = ImmutableList.copyOf(publicHeaders);
    this.dependencies = ImmutableList.copyOf(dependencies);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext context) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        ByteBuffer buffer = serializeHeaderMap(headerMap);
        WritableByteChannel channel = Channels.newChannel(out);
        buffer.flip();
        channel.write(buffer);
        out.flush();
        out.close();
      }
    };
  }

  @Override
  public String getMnemonic() {
    return "CppHeaderMap";
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint f) {
    f.addString(GUID);
    for (Map.Entry<String, String> entry : headerMap.entrySet()) {
      String key = entry.getKey();
      String path = entry.getValue();
      f.addString(key + path);
    }
  }

