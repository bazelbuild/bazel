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
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;


/** A source file that is an input to a c++ compilation. */
public abstract class CppSource implements StarlarkValue {
  private final Artifact source;
  private final Label label;

  /**
   * Types of sources.
   */
  public enum Type {
    SOURCE,
    HEADER,
    CLIF_INPUT_PROTO,
  }

  public CppSource(Artifact source, Label label) {
    this.source = source;
    this.label = label;
  }

  /** Returns the actual source file. */
  @StarlarkMethod(name = "file", documented = false, structField = true)
  public Artifact getSource() {
    return source;
  }

  /** Returns the label from which this source arises in the build graph. */
  @StarlarkMethod(name = "label", documented = false, structField = true)
  public Label getLabel() {
    return label;
  }

  /**
   * Returns the type of this source.
   */
  abstract Type getType();

  @StarlarkMethod(name = "type", documented = false, structField = true)
  public String getTypeForStarlark() {
    return getType().name();
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("CppSource(type=");
    printer.append(getType().name());
    printer.append(", label=");
    printer.append(label.toString());
    printer.append(", source=");
    printer.append(source.getExecPathString());
    printer.append(")");
  }

  private static class SourceCppSource extends CppSource {

    protected SourceCppSource(Artifact source, Label label) {
      super(source, label);
    }

    @Override
    public Type getType() {
      return Type.SOURCE;
    }
  }

  private static class HeaderCppSource extends CppSource {
    protected HeaderCppSource(Artifact source, Label label) {
      super(source, label);
    }

    @Override
    public Type getType() {
      return Type.HEADER;
    }
  }

  private static class ClifProtoCppSource extends CppSource {
    protected ClifProtoCppSource(Artifact source, Label label) {
      super(source, label);
    }

    @Override
    public Type getType() {
      return Type.CLIF_INPUT_PROTO;
    }
  }

  /**
   * Creates a {@code CppSource}.
   *
   * @param source the actual source file
   * @param label the label from which this source arises in the build graph
   * @param type type of the source file.
   */
  static CppSource create(Artifact source, Label label, Type type) {
    switch (type) {
      case SOURCE:
        return new SourceCppSource(source, label);
      case HEADER:
        return new HeaderCppSource(source, label);
      case CLIF_INPUT_PROTO:
        return new ClifProtoCppSource(source, label);
        default:
           throw new IllegalStateException("Unhandled CppSource type: " + type);
    }
  }
}
