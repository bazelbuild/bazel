// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkdocextract;

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.devtools.starlark.common.DocstringUtils.DocstringParseError;
import java.util.List;
import net.starlark.java.syntax.Location;

/**
 * An exception that may be thrown during construction of {@link StarlarkFunctionInfo} if the
 * function's docstring is malformed.
 */
public class DocstringParseException extends Exception {
  public DocstringParseException(
      String functionName, Location definedLocation, List<DocstringParseError> parseErrors) {
    super(getMessage(functionName, definedLocation, parseErrors));
  }

  private static String getMessage(
      String functionName, Location definedLocation, List<DocstringParseError> parseErrors) {
    StringBuilder message = new StringBuilder();
    message.append(
        String.format(
            "Unable to generate documentation for function %s (defined at %s) "
                + "due to malformed docstring. Parse errors:\n",
            functionName, definedLocation));
    for (DocstringParseError parseError : parseErrors) {
      message.append(
          String.format(
              "  %s line %s: %s\n",
              definedLocation,
              parseError.getLineNumber(),
              parseError.getMessage().replace('\n', ' ')));
    }
    return message.toString();
  }
}
