// Copyright 2020 The Bazel Authors. All rights reserved.
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

package net.starlark.java.syntax;

import com.google.auto.value.AutoValue;

/**
 * FileOptions is a set of options that affect the static processing---scanning, parsing, validation
 * (identifier resolution), and compilation---of a single Starlark file. These options affect the
 * language accepted by the frontend (in effect, the dialect), and "code generation", analogous to
 * the command-line options of a typical compiler.
 *
 * <p>Different files within the same application and even executed within the same thread may be
 * subject to different file options. For example, in Bazel, load statements in WORKSPACE files may
 * need to be interleaved with other statements, whereas in .bzl files, load statements must appear
 * all at the top. A single thread may execute a WORKSPACE file and call functions defined in .bzl
 * files.
 *
 * <p>The {@link #DEFAULT} options represent the desired behavior for new uses of Starlark. It is a
 * goal to keep this set of options small and closed. Each represents a language feature, perhaps a
 * deprecated, obscure, or regrettable one. By contrast, {@link StarlarkSemantics} defines a
 * (soon-to-be) open-ended set of options that affect the dynamic behavior of Starlark threads and
 * (mostly application-defined) built-in functions, and particularly attribute selection operations
 * {@code x.f}.
 */
@AutoValue
public abstract class FileOptions {

  /** The default options for Starlark static processing. New clients should use these defaults. */
  public static final FileOptions DEFAULT = builder().build();

  /**
   * During resolution, permit load statements to access private names such as {@code _x}. <br>
   * (Required for continued support of Bazel "WORKSPACE.resolved" files.)
   */
  public abstract boolean allowLoadPrivateSymbols();

  /**
   * During resolution, permit multiple assignments to a given top-level binding, whether file-local
   * or global. However, as usual, you may not create both a file-local and a global binding of the
   * same name (e.g. {@code load(..., x="x"); x=1}), so if you use this option, you probably want
   * {@link #loadBindsGlobally} too, to avoid confusing errors. <br>
   * (Required for continued support of Bazel BUILD files and Copybara files.)
   */
  public abstract boolean allowToplevelRebinding();

  /**
   * During resolution, make load statements bind global variables of the module, not file-local
   * variables.<br>
   * (Intended for use in REPLs, and the Bazel prelude; and in Bazel BUILD files, which make
   * frequent use of {@code load(..., "x"); x=1} for reasons unclear.)
   */
  public abstract boolean loadBindsGlobally();

  /**
   * During resolution, require load statements to appear before other kinds of statements. <br>
   * (Required for continued support of Bazel BUILD and especially WORKSPACE files.)
   */
  public abstract boolean requireLoadStatementsFirst();

  /**
   * During lexing, whether to ban non-ASCII characters (i.e., characters with code point > U+7F) in
   * string literals.
   *
   * <p>This applies to string literals' raw content as well as escape sequences.
   */
  public abstract boolean stringLiteralsAreAsciiOnly();

  public static Builder builder() {
    // These are the DEFAULT values.
    return new AutoValue_FileOptions.Builder()
        .allowLoadPrivateSymbols(false)
        .allowToplevelRebinding(false)
        .loadBindsGlobally(false)
        .requireLoadStatementsFirst(true)
        .stringLiteralsAreAsciiOnly(false);
  }

  public abstract Builder toBuilder();

  /** This javadoc comment states that FileOptions.Builder is a builder for FileOptions. */
  @AutoValue.Builder
  public abstract static class Builder {
    // AutoValue why u make me say it 3 times?
    public abstract Builder allowLoadPrivateSymbols(boolean value);

    public abstract Builder allowToplevelRebinding(boolean value);

    public abstract Builder loadBindsGlobally(boolean value);

    public abstract Builder requireLoadStatementsFirst(boolean value);

    public abstract Builder stringLiteralsAreAsciiOnly(boolean value);

    public abstract FileOptions build();
  }
}
