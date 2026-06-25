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
package com.google.devtools.build.lib.rules.cpp;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.annotations.SerializedName;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Generator for JSON Compilation Database format as specified by Clang.
 *
 * @see <a href="https://clang.llvm.org/docs/JSONCompilationDatabase.html">
 *     JSON Compilation Database Format</a>
 */
public final class CompilationDatabaseGenerator {

  private CompilationDatabaseGenerator() {
    // Utility class, do not instantiate
  }

  /** Entry in the compilation database. */
  public static class Entry {
    /** The working directory of the compilation. */
    @SerializedName("directory")
    public final String directory;

    /** The relative or absolute path of the source file. */
    @SerializedName("file")
    public final String file;

    /** The command as an argument array. */
    @SerializedName("arguments")
    public final List<String> arguments;

    /** The output file path (optional). */
    @SerializedName("output")
    public final String output;

    public Entry(String directory, String file, List<String> arguments, String output) {
      this.directory = directory;
      this.file = file;
      this.arguments = arguments;
      this.output = output;
    }
  }

  /**
   * Converts a list of entries to JSON format.
   *
   * @param entries the compilation database entries
   * @return JSON string representation
   */
  public static String toJson(List<Entry> entries) {
    Gson gson = new GsonBuilder().disableHtmlEscaping().create();
    return gson.toJson(entries);
  }

  /** Converts the JSON string to UTF-8 bytes. */
  public static byte[] toJsonBytes(List<Entry> entries) {
    return toJson(entries).getBytes(StandardCharsets.UTF_8);
  }
}
