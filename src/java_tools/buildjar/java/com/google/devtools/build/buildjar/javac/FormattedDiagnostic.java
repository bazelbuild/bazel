// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static java.util.Locale.ENGLISH;

import com.google.common.collect.ImmutableList;
import com.sun.tools.javac.api.ClientCodeWrapper.Trusted;
import com.sun.tools.javac.api.DiagnosticFormatter;
import com.sun.tools.javac.code.Lint.LintCategory;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.JCDiagnostic;
import com.sun.tools.javac.util.JavacMessages;
import com.sun.tools.javac.util.Log;
import java.util.Locale;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticListener;
import javax.tools.JavaFileObject;

/**
 * A {@link Diagnostic<JavaFileObject>} that includes the full formatted message produced by javac,
 * which relies on compilation internals and can't be reproduced after the compilation is complete.
 */
public class FormattedDiagnostic implements Diagnostic<JavaFileObject> {

  public final Diagnostic<? extends JavaFileObject> diagnostic;
  public final String formatted;
  public final String lintCategory;

  public FormattedDiagnostic(
      Diagnostic<? extends JavaFileObject> diagnostic, String formatted, String lintCategory) {
    this.diagnostic = diagnostic;
    this.formatted = formatted;
    this.lintCategory = lintCategory;
  }

  /** The formatted diagnostic message produced by javac's diagnostic formatter. */
  public String getFormatted() {
    return formatted;
  }

  public String getLintCategory() {
    return lintCategory;
  }

  @Override
  public String toString() {
    return formatted;
  }

  @Override
  public Kind getKind() {
    return diagnostic.getKind();
  }

  @Override
  public JavaFileObject getSource() {
    return diagnostic.getSource();
  }

  @Override
  public long getPosition() {
    return diagnostic.getPosition();
  }

  @Override
  public long getStartPosition() {
    return diagnostic.getStartPosition();
  }

  @Override
  public long getEndPosition() {
    return diagnostic.getEndPosition();
  }

  @Override
  public long getLineNumber() {
    return diagnostic.getLineNumber();
  }

  @Override
  public long getColumnNumber() {
    return diagnostic.getColumnNumber();
  }

  @Override
  public String getCode() {
    return diagnostic.getCode();
  }

  @Override
  public String getMessage(Locale locale) {
    return diagnostic.getMessage(locale);
  }

  /** A {@link DiagnosticListener<JavaFileObject>} that saves {@link FormattedDiagnostic}s. */
  @Trusted
  static class Listener implements DiagnosticListener<JavaFileObject> {

    private final ImmutableList.Builder<FormattedDiagnostic> diagnostics = ImmutableList.builder();
    private final boolean failFast;
    private final Context context;

    Listener(boolean failFast, Context context) {
      this.failFast = failFast;
      // retrieve context values later, in case it isn't initialized yet
      this.context = context;
    }

    @Override
    public void report(Diagnostic<? extends JavaFileObject> diagnostic) {
      DiagnosticFormatter<JCDiagnostic> formatter = Log.instance(context).getDiagnosticFormatter();
      Locale locale = JavacMessages.instance(context).getCurrentLocale();
      String formatted = formatter.format((JCDiagnostic) diagnostic, locale);
      LintCategory lintCategory = ((JCDiagnostic) diagnostic).getLintCategory();
      FormattedDiagnostic formattedDiagnostic =
          new FormattedDiagnostic(
              diagnostic, formatted, lintCategory != null ? lintCategory.option : null);
      diagnostics.add(formattedDiagnostic);
      if (failFast && diagnostic.getKind().equals(Diagnostic.Kind.ERROR)) {
        throw new FailFastException(formatted);
      }
    }

    ImmutableList<FormattedDiagnostic> build() {
      return diagnostics.build();
    }
  }

  static class FailFastException extends RuntimeException {
    FailFastException(String message) {
      super(message);
    }
  }

  public boolean isJSpecifyDiagnostic() {
    return getKind().equals(Diagnostic.Kind.ERROR)
        && getCode().equals("compiler.err.proc.messager")
        && getMessage(ENGLISH).startsWith("[nullness] ");
  }
}
