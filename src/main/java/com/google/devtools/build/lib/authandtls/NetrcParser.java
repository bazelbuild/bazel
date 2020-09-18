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
package com.google.devtools.build.lib.authandtls;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.authandtls.Netrc.Credential;
import java.io.BufferedReader;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** A parser used to parse .netrc content */
public class NetrcParser {
  private final static String MACHINE = "machine";
  private final static String MACDEF = "macdef";
  private final static String DEFAULT = "default";
  private final static String LOGIN = "login";
  private final static String PASSWORD = "password";
  private final static String ACCOUNT = "account";

  private enum TokenKind {
    NEWLINE,
    ITEM
  }

  private static class Token {
    static Token newline() {
      Token token = new Token();
      token.kind = TokenKind.NEWLINE;
      return token;
    }

    static Token item(String item) {
      Preconditions.checkNotNull(item, "item");
      Token token = new Token();
      token.kind = TokenKind.ITEM;
      token.item = item;
      return token;
    }

    private TokenKind kind;
    @Nullable
    private String item;

    @Nullable
    String getItem() {
      if (kind == TokenKind.ITEM) {
        return item;
      } else {
        return null;
      }
    }
  }

  private static class TokenStream implements Closeable {
    private final BufferedReader bufferedReader;
    private final Deque<Token> tokens = new ArrayDeque<>();

    TokenStream(InputStream inputStream) throws IOException {
      bufferedReader = new BufferedReader(new InputStreamReader(inputStream));

      processLine();
    }

    @Override
    public void close() throws IOException {
      bufferedReader.close();
    }

    private void processLine() throws IOException {
      String line = bufferedReader.readLine();
      if (line == null) { return; }
      // Comments start with #. Ignore these lines.
      if (!line.startsWith("#")) {
        List<Token> newTokens = Arrays.stream(line.split("\\s+"))
            .filter(s -> !Strings.isNullOrEmpty(s)).map(
                Token::item).collect(Collectors.toList());
        tokens.addAll(newTokens);
      }
      tokens.add(Token.newline());
    }

    void addFirst(Token token) {
      tokens.addFirst(token);
    }

    public boolean hasNext() {
      return !tokens.isEmpty();
    }

    public Token next() throws IOException {
      Token token = tokens.removeFirst();
      if (tokens.isEmpty()) {
        processLine();
      }
      return token;
    }
  }

  public static Netrc parseAndClose(InputStream inputStream) throws IOException {
    try (TokenStream tokenStream = new TokenStream(inputStream)) {
      return parse(tokenStream);
    }
  }

  private static Netrc parse(TokenStream tokenStream) throws IOException {
    Credential defaultCredential = null;
    Map<String, Credential> credentialMap = new HashMap<>();

    boolean done = false;
    while (!done && tokenStream.hasNext()) {
      Token token = tokenStream.next();
      switch (token.kind) {
        case ITEM: {
          String item = token.getItem();
          Preconditions.checkState(item != null);
          switch (item) {
            case MACHINE: {
              String machine = nextItem(tokenStream);
              Credential.Builder builder = Credential.builder(machine);
              parseCredentialForMachine(tokenStream, builder);
              credentialMap.put(machine, builder.build());
              break;
            }
            case MACDEF: {
              skipMacdef(tokenStream);
              break;
            }
            case DEFAULT: {
              Credential.Builder builder = Credential.builder("default");
              parseCredentialForMachine(tokenStream, builder);
              defaultCredential = builder.build();
              // There can be only one default token, and it must be after all machine tokens.
              done = true;
              break;
            }
            default: {
              throw new IOException(String
                  .format("Unexpected token: %s (expecting %s, %s or %s)", item, MACHINE, MACDEF,
                      DEFAULT));
            }
          }
        }
        case NEWLINE: {
          break;
        }
      }
    }

    return new Netrc(defaultCredential, ImmutableMap.copyOf(credentialMap));
  }

  private static String nextItem(TokenStream tokenStream) throws IOException {
    while (tokenStream.hasNext()) {
      Token token = tokenStream.next();
      switch (token.kind) {
        case ITEM: {
          String item = token.getItem();
          Preconditions.checkState(item != null);
          return item;
        }
        case NEWLINE: {
          break;
        }
      }
    }

    throw new IOException("Unexpected EOF");
  }

  /** Parse credentials for a given machine from token stream. */
  private static void parseCredentialForMachine(TokenStream tokenStream, Credential.Builder builder) throws IOException {
    boolean done = false;
    while (!done && tokenStream.hasNext()) {
      Token token = tokenStream.next();
      switch (token.kind) {
        case ITEM: {
          String item = token.getItem();
          Preconditions.checkState(item != null);
          switch (item) {
            case LOGIN:
              builder.setLogin(nextItem(tokenStream));
              break;
            case PASSWORD:
              builder.setPassword(nextItem(tokenStream));
              break;
            case ACCOUNT:
              builder.setAccount(nextItem(tokenStream));
              break;
            case MACHINE:
            case MACDEF:
            case DEFAULT:
              tokenStream.addFirst(token);
              done = true;
              break;
            default:
              throw new IOException(String
                  .format("Unexpected item: %s (expecting %s, %s, %s, %s, %s or %s)", item, LOGIN,
                      PASSWORD, ACCOUNT, MACHINE, MACDEF, DEFAULT));
          }
        }
        case NEWLINE: {
          break;
        }
      }
    }
  }

  private static void skipMacdef(TokenStream tokenStream) throws IOException {
    int numNewlines = 0;
    while (tokenStream.hasNext()) {
      Token token = tokenStream.next();
      if (token.kind == TokenKind.NEWLINE) {
        ++numNewlines;
      } else {
        numNewlines = 0;
      }
      if (numNewlines >= 2) {
        break;
      }
    }
  }
}
