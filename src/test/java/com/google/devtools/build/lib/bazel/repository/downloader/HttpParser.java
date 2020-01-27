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

package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.common.base.Ascii;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/** Utility class for parsing HTTP messages. */
final class HttpParser {

  /** Exhausts request line and headers of HTTP request. */
  static void readHttpRequest(InputStream stream) throws IOException {
    readHttpRequest(stream, new HashMap<String, String>());
  }

  /**
   * Parses request line and headers of HTTP request.
   *
   * <p>This parser is correct and extremely lax. This implementation is Θ(n) and the stream should
   * be buffered. All decoding is ISO-8859-1. A 1mB upper bound on memory is enforced.
   *
   * @throws IOException if reading failed or premature end of stream encountered
   * @throws HttpParserError if 400 error should be sent to client and connection must be closed
   */
  static void readHttpRequest(InputStream stream, Map<String, String> output) throws IOException {
    StringBuilder builder = new StringBuilder(256);
    State state = State.METHOD;
    String key = "";
    int toto = 0;
    while (true) {
      int c = stream.read();
      if (c == -1) {
        throw new IOException();  // RFC7230 § 3.4
      }
      if (++toto == 1024 * 1024) {
        throw new HttpParserError();  // RFC7230 § 3.2.5
      }
      switch (state) {
        case METHOD:
          if (c == ' ') {
            if (builder.length() == 0) {
              throw new HttpParserError();
            }
            output.put("x-method", builder.toString());
            builder.setLength(0);
            state = State.URI;
          } else if (c == '\r' || c == '\n') {
            break;  // RFC7230 § 3.5
          } else {
            builder.append(Ascii.toUpperCase((char) c));
          }
          break;
        case URI:
          if (c == ' ') {
            if (builder.length() == 0) {
              throw new HttpParserError();
            }
            output.put("x-request-uri", builder.toString());
            builder.setLength(0);
            state = State.VERSION;
          } else {
            builder.append((char) c);
          }
          break;
        case VERSION:
          if (c == '\r' || c == '\n') {
            output.put("x-version", builder.toString());
            builder.setLength(0);
            state = c == '\r' ? State.CR1 : State.LF1;
          } else {
            builder.append(Ascii.toUpperCase((char) c));
          }
          break;
        case CR1:
          if (c == '\n') {
            state = State.LF1;
            break;
          }
          throw new HttpParserError();
        case LF1:
          if (c == '\r') {
            state = State.LF2;
            break;
          } else if (c == '\n') {
            return;
          } else if (c == ' ' || c == '\t') {
            throw new HttpParserError("Line folding unacceptable");  // RFC7230 § 3.2.4
          }
          state = State.HKEY;
          // fall through
        case HKEY:
          if (c == ':') {
            key = builder.toString();
            builder.setLength(0);
            state = State.HSEP;
          } else {
            builder.append(Ascii.toLowerCase((char) c));
          }
          break;
        case HSEP:
          if (c == ' ' || c == '\t') {
            break;
          }
          state = State.HVAL;
          // fall through
        case HVAL:
          if (c == '\r' || c == '\n') {
            output.put(key, builder.toString());
            builder.setLength(0);
            state = c == '\r' ? State.CR1 : State.LF1;
          } else {
            builder.append((char) c);
          }
          break;
        case LF2:
          if (c == '\n') {
            return;
          }
          throw new HttpParserError();
        default:
          throw new AssertionError();
      }
    }
  }

  static final class HttpParserError extends IOException {
    HttpParserError() {
      this("Malformed Request");
    }

    HttpParserError(String messageForClient) {
      super(messageForClient);
    }
  }

  private enum State { METHOD, URI, VERSION, HKEY, HSEP, HVAL, CR1, LF1, LF2 }

  private HttpParser() {}
}
