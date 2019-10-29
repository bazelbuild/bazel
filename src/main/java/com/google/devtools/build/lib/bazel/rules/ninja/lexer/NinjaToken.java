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
//

package com.google.devtools.build.lib.bazel.rules.ninja.lexer;

import java.nio.charset.StandardCharsets;
import javax.annotation.Nullable;

/**
 * Token types for {@link NinjaLexer}.
 */
public enum NinjaToken {
  ERROR(null),
  BUILD("build"),
  RULE("rule"),
  TEXT(null),
  IDENTIFIER(null),
  VARIABLE(null),
  DEFAULT("default"),
  POOL("pool"),
  SUBNINJA("subninja"),
  INCLUDE("include"),

  COLON(null),
  EQUALS(null),
  PIPE(null),
  PIPE2(null),

  INDENT(null),
  NEWLINE(null),

  ZERO(null),
  EOF(null);

  private final byte[] bytes;

  NinjaToken(@Nullable String text) {
    this.bytes = text != null ? text.getBytes(StandardCharsets.ISO_8859_1) : new byte[0];
  }

  public byte[] getBytes() {
    return bytes;
  }
}
