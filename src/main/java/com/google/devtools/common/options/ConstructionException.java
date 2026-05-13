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
package com.google.devtools.common.options;

// TODO(b/65049598) make ConstructionException checked.
/**
 * An unchecked exception thrown when there is a problem constructing a parser, e.g. an error while
 * validating an {@link OptionDefinition} in one of its {@link OptionsBase} subclasses.
 *
 * <p>This exception is unchecked because it generally indicates an internal error affecting all
 * invocations of the program. I.e., any such error should be immediately obvious to the developer.
 * Although unchecked, we explicitly mark some methods as throwing it as a reminder in the API.
 */
public class ConstructionException extends RuntimeException {

  public ConstructionException(String message) {
    super(message);
  }

  public ConstructionException(Throwable cause) {
    super(cause);
  }

  public ConstructionException(String message, Throwable cause) {
    super(message, cause);
  }
}
