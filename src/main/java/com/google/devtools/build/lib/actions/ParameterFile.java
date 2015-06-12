// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;


/**
 * Support for parameter file generation (as used by gcc and other tools, e.g.
 * {@code gcc @param_file}. Note that the parameter file needs to be explicitly
 * deleted after use. Different tools require different parameter file formats,
 * which can be selected via the {@link ParameterFileType} enum.
 *
 * <p>The default charset is ISO-8859-1 (latin1). This also has to match the
 * expectation of the tool.
 *
 * <p>Don't use this class for new code. Use the ParameterFileWriteAction
 * instead!
 */
public class ParameterFile {

  /**
   * Different styles of parameter files.
   */
  public static enum ParameterFileType {
    /**
     * A parameter file with every parameter on a separate line. This format
     * cannot handle newlines in parameters. It is currently used for most
     * tools, but may not be interpreted correctly if parameters contain
     * white space or other special characters. It should be avoided for new
     * development.
     */
    UNQUOTED,

    /**
     * A parameter file where each parameter is correctly quoted for shell
     * use, and separated by white space (space, tab, newline). This format is
     * safe for all characters, but must be specially supported by the tool. In
     * particular, it must not be used with gcc and related tools, which do not
     * support this format as it is.
     */
    SHELL_QUOTED;
  }

  @VisibleForTesting
  public static final FileType PARAMETER_FILE = FileType.of(".params");

  /**
   * Creates a parameter file with the given parameters.
   */
  private ParameterFile() {
  }
  /**
   * Derives an exec path from a given exec path by appending <code>".params"</code>.
   */
  public static PathFragment derivePath(PathFragment original) {
    return original.replaceName(original.getBaseName() + "-2.params");
  }

}
