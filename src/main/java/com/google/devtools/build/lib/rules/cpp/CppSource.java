// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;

import java.util.Map;

/** A source file that is an input to a c++ compilation. */
public abstract class CppSource {
  private final Artifact source;
  private final Label label;
  private final Map<String, String> buildVariables;

  /**
   * Types of sources.
   */
  public enum Type {
    SOURCE,
    HEADER,
    CLIF_INPUT_PROTO,
  }

  public CppSource(Artifact source, Label label, Map<String, String> buildVariables) {
    this.source = source;
    this.label = label;
    this.buildVariables = buildVariables;
  }

  /**
   * Returns the actual source file.
   */
  public Artifact getSource() {
    return source;
  }

  /**
   * Returns the label from which this source arises in the build graph.
   */
  public Label getLabel() {
    return label;
  }

  /**
   * Returns build variables to be used specifically in the compilation of this source.
   */
  public Map<String, String> getBuildVariables() {
    return buildVariables;
  }

  /**
   * Returns the type of this source.
   */
  abstract Type getType();

  private static class SourceCppSource extends CppSource {

    protected SourceCppSource(Artifact source, Label label, Map<String, String> buildVariables) {
      super(source, label, buildVariables);
    }

    @Override
    public Type getType() {
      return Type.SOURCE;
    }
  }

  private static class HeaderCppSource extends CppSource {
    protected HeaderCppSource(Artifact source, Label label, Map<String, String> buildVariables) {
      super(source, label, buildVariables);
    }

    @Override
    public Type getType() {
      return Type.HEADER;
    }
  }

  private static class ClifProtoCppSource extends CppSource {
    protected ClifProtoCppSource(Artifact source, Label label, Map<String, String> buildVariables) {
      super(source, label, buildVariables);
    }

    @Override
    public Type getType() {
      return Type.CLIF_INPUT_PROTO;
    }
  }



  /**
   * Creates a {@code CppSource}.
   * @param source  the actual source file
   * @param label  the label from which this source arises in the build graph
   * @param buildVariables  build variables that should be used specifically in the compilation
   *     of this source
   * @param type type of the source file.
   */
  static CppSource create(Artifact source, Label label, Map<String, String> buildVariables,
      Type type) {
    switch (type) {
        case SOURCE:
            return new SourceCppSource(source, label, buildVariables);
        case HEADER:
           return new HeaderCppSource(source, label, buildVariables);
        case CLIF_INPUT_PROTO:
           return new ClifProtoCppSource(source, label, buildVariables);
        default:
           throw new IllegalStateException("Unhandled CppSource type: " + type);
    }
  }
}
