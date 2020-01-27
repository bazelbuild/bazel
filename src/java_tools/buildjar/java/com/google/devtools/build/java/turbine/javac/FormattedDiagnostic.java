// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine.javac;

import javax.tools.Diagnostic;
import javax.tools.JavaFileObject;

/**
 * A wrapper for a {@link Diagnostic<JavaFileObject>} that includes the full formatted message
 * produced by javac, which relies on compilation internals and can't be reproduced after the
 * compilation is complete.
 */
class FormattedDiagnostic {
  private final Diagnostic<? extends JavaFileObject> diagnostic;
  private final String message;

  FormattedDiagnostic(Diagnostic<? extends JavaFileObject> diagnostic, String message) {
    this.diagnostic = diagnostic;
    this.message = message;
  }

  Diagnostic<? extends JavaFileObject> diagnostic() {
    return diagnostic;
  }

  String message() {
    return message;
  }
}
