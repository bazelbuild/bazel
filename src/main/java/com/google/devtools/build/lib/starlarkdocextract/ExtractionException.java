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

/** An exception indicating that Starlark API documentation could not be extracted. */
public final class ExtractionException extends Exception {
  public ExtractionException(String message) {
    super(message);
  }

  public ExtractionException(Throwable cause) {
    super(cause);
  }

  public ExtractionException(String message, Throwable cause) {
    super(message, cause);
  }
}
