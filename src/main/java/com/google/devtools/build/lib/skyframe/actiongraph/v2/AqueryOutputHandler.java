// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.actiongraph.v2;

import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Action;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.AspectDescriptor;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Configuration;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.DepSetOfFiles;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.PathFragment;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.RuleClass;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Target;
import java.io.IOException;

/** Outputs various messages of analysis_v2.proto. */
public interface AqueryOutputHandler extends AutoCloseable {
  /** Defines the types of proto output this class can handle. */
  enum OutputType {
    BINARY("proto"),
    DELIMITED_BINARY("streamed_proto"),
    TEXT("textproto"),
    JSON("jsonproto");

    private final String formatName;

    OutputType(String formatName) {
      this.formatName = formatName;
    }

    public String formatName() {
      return formatName;
    }

    public static OutputType fromString(String string) throws InvalidAqueryOutputFormatException {
      switch (string) {
        case "proto":
          return BINARY;
        case "streamed_proto":
          return DELIMITED_BINARY;
        case "textproto":
          return TEXT;
        case "jsonproto":
          return JSON;
        default: // fall out
      }
      throw new InvalidAqueryOutputFormatException("Invalid aquery output format: " + string);
    }
  }

  void outputArtifact(Artifact message) throws IOException;

  void outputAction(Action message) throws IOException;

  void outputTarget(Target message) throws IOException;

  void outputDepSetOfFiles(DepSetOfFiles message) throws IOException;

  void outputConfiguration(Configuration message) throws IOException;

  void outputAspectDescriptor(AspectDescriptor message) throws IOException;

  void outputRuleClass(RuleClass message) throws IOException;

  void outputPathFragment(PathFragment message) throws IOException;

  /** Called at the end of the query process. */
  @Override
  void close() throws IOException;
}
