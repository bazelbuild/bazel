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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;

import java.util.Map;

/** A source file that is an input to a c++ compilation. */
@AutoValue
public abstract class CppSource {

  /**
   * Types of sources.
   */
  public enum Type {
    SOURCE,
    HEADER,
    CLIF_INPUT_PROTO,
  }

  /**
   * Creates a {@code CppSource}.
   * 
   * @param source  the actual source file
   * @param label  the label from which this source arises in the build graph
   * @param buildVariables  build variables that should be used specifically in the compilation
   *     of this source
   * @param type type of the source file.
   */
  static CppSource create(Artifact source, Label label, Map<String, String> buildVariables,
      Type type) {
    return new AutoValue_CppSource(source, label, buildVariables, type);
  }

  /**
   * Returns the actual source file.
   */
  abstract Artifact getSource();

  /**
   * Returns the label from which this source arises in the build graph.
   */
  abstract Label getLabel();

  /**
   * Returns build variables to be used specifically in the compilation of this source.
   */
  abstract Map<String, String> getBuildVariables();

  /**
   * Returns the type of this source.
   */
  abstract Type getType();
}
