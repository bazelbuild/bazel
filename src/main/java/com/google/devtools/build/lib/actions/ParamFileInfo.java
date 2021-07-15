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

package com.google.devtools.build.lib.actions;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import java.nio.charset.Charset;
import java.util.Objects;
import javax.annotation.concurrent.Immutable;

/**
 * An object that encapsulates how a params file should be constructed: what is the filetype, what
 * charset to use and what prefix (typically "@") to use.
 */
@Immutable
public final class ParamFileInfo {
  private final ParameterFileType fileType;
  private final Charset charset;
  private final String flagFormatString;
  private final boolean always;
  private final boolean flagsOnly;

  private static final Interner<ParamFileInfo> paramFileInfoInterner =
      BlazeInterners.newWeakInterner();

  private ParamFileInfo(Builder builder) {
    this.fileType = Preconditions.checkNotNull(builder.fileType);
    this.charset = Preconditions.checkNotNull(builder.charset);
    this.flagFormatString = Preconditions.checkNotNull(builder.flagFormatString);
    this.always = builder.always;
    this.flagsOnly = builder.flagsOnly;
  }

  /** Returns the file type. */
  public ParameterFileType getFileType() {
    return fileType;
  }

  /** Returns the charset. */
  public Charset getCharset() {
    return charset;
  }

  /** Returns the format string for the params filename on the command line (typically "@%s"). */
  public String getFlagFormatString() {
    return flagFormatString;
  }

  /** Returns true if a params file should always be used. */
  public boolean always() {
    return always;
  }

  /**
   * If true, only the flags will be spilled to the file, leaving positional args on the command
   * line.
   */
  public boolean flagsOnly() {
    return flagsOnly;
  }

  @Override
  public int hashCode() {
    return Objects.hash(charset, flagFormatString, fileType, always);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ParamFileInfo)) {
      return false;
    }
    ParamFileInfo other = (ParamFileInfo) obj;
    return fileType.equals(other.fileType)
        && charset.equals(other.charset)
        && flagFormatString.equals(other.flagFormatString)
        && always == other.always
        && flagsOnly == other.flagsOnly;
  }

  public static Builder builder(ParameterFileType parameterFileType) {
    return new Builder(parameterFileType);
  }

  /** Builder for a ParamFileInfo. */
  public static class Builder {
    private final ParameterFileType fileType;
    private Charset charset = ISO_8859_1;
    private String flagFormatString = "@%s";
    private boolean always;
    private boolean flagsOnly;

    private Builder(ParameterFileType fileType) {
      this.fileType = fileType;
    }

    /** Sets the encoding to write the parameter file with. */
    public Builder setCharset(Charset charset) {
      this.charset = charset;
      return this;
    }

    /**
     * Sets a format string to use for the flag that is passed to original command.
     *
     * <p>The format string must have a single "%s" that will be replaced by the execution path to
     * the param file.
     */
    public Builder setFlagFormatString(String flagFormatString) {
      this.flagFormatString = flagFormatString;
      return this;
    }

    /** Set whether the parameter file is always used, regardless of parameter file length. */
    public Builder setUseAlways(boolean always) {
      this.always = always;
      return this;
    }

    /**
     * If true, only the flags will be spilled to the file, leaving positional args on the command
     * line. (Default is false.)
     */
    public Builder setFlagsOnly(boolean flagsOnly) {
      this.flagsOnly = flagsOnly;
      return this;
    }

    public ParamFileInfo build() {
      return paramFileInfoInterner.intern(new ParamFileInfo(this));
    }
  }
}
