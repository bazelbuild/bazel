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

package com.google.devtools.build.lib.analysis.actions;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;

import java.nio.charset.Charset;
import java.util.Objects;

import javax.annotation.concurrent.Immutable;

/**
 * An object that encapsulates how a params file should be constructed: what is the filetype,
 * what charset to use and what prefix (typically "@") to use.
 */
@Immutable
public final class ParamFileInfo {
  private final ParameterFileType fileType;
  private final Charset charset;
  private final String flag;

  public ParamFileInfo(ParameterFileType fileType, Charset charset, String flag) {
    this.fileType = Preconditions.checkNotNull(fileType);
    this.charset = Preconditions.checkNotNull(charset);
    this.flag = Preconditions.checkNotNull(flag);
  }

  /**
   * Returns the file type.
   */
  public ParameterFileType getFileType() {
    return fileType;
  }

  /**
   * Returns the charset.
   */
  public Charset getCharset() {
    return charset;
  }

  /**
   * Returns the prefix for the params filename on the command line (typically "@").
   */
  public String getFlag() {
    return flag;
  }

  @Override
  public int hashCode() {
    return Objects.hash(charset, flag, fileType);
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
    return fileType.equals(other.fileType) && charset.equals(other.charset)
        && flag.equals(other.flag);
  }
}
