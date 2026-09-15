// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Strings.nullToEmpty;

import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import javax.annotation.Nullable;
import net.starlark.java.eval.Module;

/** An exception indicating that Starlark API documentation could not be extracted. */
public final class ExtractionException extends Exception {
  public ExtractionException(String message) {
    super(message);
  }

  public ExtractionException(Module module, String message) {
    super(prefixWithModuleFilename(module, message, null));
  }

  public ExtractionException(Module module, Throwable cause) {
    super(prefixWithModuleFilename(module, null, cause), cause);
  }

  public ExtractionException(Module module, String message, Throwable cause) {
    super(prefixWithModuleFilename(module, message, cause), cause);
  }

  private static String prefixWithModuleFilename(
      Module module, @Nullable String message, @Nullable Throwable cause) {
    BazelModuleContext bazelModuleContext = BazelModuleContext.of(module);
    if (bazelModuleContext == null) {
      return message;
    }
    if (message == null) {
      message = cause.getMessage();
    }
    return String.format("in %s: %s", bazelModuleContext.filename(), nullToEmpty(message));
  }
}
