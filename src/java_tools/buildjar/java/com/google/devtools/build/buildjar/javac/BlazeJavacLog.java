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

package com.google.devtools.build.buildjar.javac;

import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.JCDiagnostic;
import com.sun.tools.javac.util.Log;

/**
 * Log class for our custom patched javac.
 *
 * <p> This log class tweaks the standard javac log class so
 * that it drops all non-errors after the first error that
 * gets reported. By doing this, we
 * ensure that all warnings are listed before all errors in javac's
 * output. This makes life easier for everybody.
 */
public class BlazeJavacLog extends Log {

  private boolean hadError = false;

  /**
   * Registers a custom BlazeJavacLog for the given context and -Werror spec.
   *
   * @param context Context
   */
  public static void preRegister(final Context context) {
    context.put(logKey, new Context.Factory<Log>() {
      @Override
      public Log make(Context c) {
        return new BlazeJavacLog(c);
      }
    });
  }

  public static BlazeJavacLog instance(Context context) {
    return (BlazeJavacLog) context.get(logKey);
  }

  BlazeJavacLog(Context context) {
    super(context);
  }

  /**
   * Returns true if we should display the note diagnostic
   * passed in as argument, and false if we should discard
   * it.
   */
  private boolean shouldDisplayNote(JCDiagnostic diag) {
    String noteCode = diag.getCode();
    return noteCode == null ||
        (!noteCode.startsWith("compiler.note.deprecated") &&
         !noteCode.startsWith("compiler.note.unchecked"));
  }

  @Override
  protected void writeDiagnostic(JCDiagnostic diag) {
    switch (diag.getKind()) {
      case NOTE:
        if (shouldDisplayNote(diag)) {
          super.writeDiagnostic(diag);
        }
        break;
      case ERROR:
        hadError = true;
        super.writeDiagnostic(diag);
        break;
      default:
        if (!hadError) {
          // Do not print further warnings if an error has occured.
          super.writeDiagnostic(diag);
        }
    }
  }
}
