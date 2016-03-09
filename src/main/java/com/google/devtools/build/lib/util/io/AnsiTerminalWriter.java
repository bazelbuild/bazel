// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util.io;

import java.io.IOException;

/**
 * An append-only interface to access to a terminal.
 *
 * This interface allows to specify a text to be written to the user
 * using semantical highlighting (like failure) without any knowledge
 * about the nature of the terminal or the position the text is to be
 * written to. Callers to this interface should make no assumption
 * about how the text is rendered; if the user's terminal does not
 * support, e.g., colors, the highlighting might be done by adding
 * additional characters.
 *
 * All interface functions are supposed to return the object itself
 * to allow chaining of commands.
 */
public interface AnsiTerminalWriter {

  /**
   * Write some text to the user
   */
  AnsiTerminalWriter append(String text) throws IOException;

  /**
   * Start a new line in the way appropriate for the given terminal
   */
  AnsiTerminalWriter newline() throws IOException;

  /**
   * Tell the terminal that the following text will be a positive
   * status message.
   */
  AnsiTerminalWriter okStatus() throws IOException;

  /**
   * Tell the terminal that the following text will be an error-reporting
   * status message.
   */
  AnsiTerminalWriter failStatus() throws IOException;

  /**
   * Tell the terminal that the following text will be normal text, not
   * indicating a status or similar.
   */
  AnsiTerminalWriter normal() throws IOException;
}
