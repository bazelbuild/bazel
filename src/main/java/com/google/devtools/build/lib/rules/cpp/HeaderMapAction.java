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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import java.io.OutputStream;//?
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;
import java.util.Map;

@Immutable
public final class HeaderMapAction extends AbstractFileWriteAction {

  private static final String GUID = "b9d8aba5-5dab-481d-a2e0-937589da336e";

  // C++ header map of the current target
  private final ImmutableMap <String, String> headerMap;

  public HeaderMapAction(
      ActionOwner owner,
      ImmutableMap <String, String> headerMap,
      Artifact output
      ) {
    super(
        owner,
        ImmutableList.of(),
        output,
        /* makeExecutable= */ false);
    this.headerMap = headerMap;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext context)  {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        ClangHeaderMap headerMap1 = new ClangHeaderMap(headerMap);
        WritableByteChannel channel = Channels.newChannel(out);
        headerMap1.writeToChannel(channel);
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
    for(Map.Entry<String, String> entry: headerMap.entrySet()){
      String key = entry.getKey();
      String path = entry.getValue();
      f.addString(key + path);
    }
  }
}
