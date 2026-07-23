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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.io.OutputStream;
import javax.annotation.Nullable;

/** An in-memory param file virtual action input. */
public final class ParamFileActionInput extends VirtualActionInput {
  private final PathFragment paramFileExecPath;
  @Nullable private final String paramFileArg;
  private final Iterable<String> arguments;
  private final ParameterFileType type;

  /**
   * Creates a new {@link ParamFileActionInput}.
   *
   * <p>Prefer {@link #ParamFileActionInput(PathFragment, String, Iterable, ParameterFileType)} if
   * this param file is referenced by a known command line argument.
   *
   * @param paramFileExecPath the exec path to the param file
   * @param arguments the arguments to write to the param file
   * @param type the param file type
   */
  public ParamFileActionInput(
      PathFragment paramFileExecPath, Iterable<String> arguments, ParameterFileType type) {
    this(paramFileExecPath, /* paramFileArg= */ null, arguments, type);
  }

  /**
   * Creates a new {@link ParamFileActionInput}.
   *
   * @param paramFileExecPath the exec path to the param file
   * @param paramFileArg the argument that references this param file on the command line
   * @param arguments the arguments to write to the param file
   * @param type the param file type
   */
  public ParamFileActionInput(
      PathFragment paramFileExecPath,
      String paramFileArg,
      Iterable<String> arguments,
      ParameterFileType type) {
    this.paramFileExecPath = paramFileExecPath;
    this.paramFileArg = paramFileArg;
    this.arguments = arguments;
    this.type = type;
  }

  @Override
  public void writeTo(OutputStream out) throws IOException {
    ParameterFile.writeParameterFile(out, arguments, type);
  }

  @Override
  @CanIgnoreReturnValue
  public byte[] atomicallyWriteTo(Path outputPath) throws IOException {
    // This is needed for internal path wrangling reasons :(
    return super.atomicallyWriteTo(outputPath);
  }

  @Override
  public String getExecPathString() {
    return paramFileExecPath.getPathString();
  }

  /**
   * Returns the argument that references this param file on the command line, or null if unknown.
   */
  @Nullable
  public String getParamFileArg() {
    return paramFileArg;
  }

  @Override
  public PathFragment getExecPath() {
    return paramFileExecPath;
  }

  public Iterable<String> getArguments() {
    return arguments;
  }

  public ParameterFileType getType() {
    return type;
  }
}
