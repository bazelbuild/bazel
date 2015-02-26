// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.rules.SkylarkModules;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FsApparatus;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * SkylarkShell is a standalone shell executing Skylark. This is intended for
 * testing purposes and not for end-users. This is very limited (environment is
 * almost empty), but it can be used to play with the language and reproduce
 * bugs. Imports and includes are not supported.
 */
class SkylarkShell {
  static final EventCollectionApparatus syntaxEvents = new EventCollectionApparatus();
  static final FsApparatus scratch = FsApparatus.newInMemory();
  static final CachingPackageLocator locator = new AbstractParserTestCase.EmptyPackageLocator();
  static final Path path = scratch.path("stdin");

  private static void exec(String inputSource, Environment env) {
    try {
      ParserInputSource input = ParserInputSource.create(inputSource, path);
      Lexer lexer = new Lexer(input, syntaxEvents.reporter());
      Parser.ParseResult result =
          Parser.parseFileForSkylark(lexer, syntaxEvents.reporter(), locator,
              SkylarkModules.getValidationEnvironment(
                  ImmutableMap.<String, SkylarkType>of()));

      Object last = null;
      for (Statement st : result.statements) {
        if (st instanceof ExpressionStatement) {
          last = ((ExpressionStatement) st).getExpression().eval(env);
        } else {
          st.exec(env);
          last = null;
        }
      }
      if (last != null) {
        System.out.println(last);
      }
    } catch (Throwable e) { // Catch everything to avoid killing the shell.
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Environment env = SkylarkModules.getNewEnvironment(new EventHandler() {
      @Override
      public void handle(Event event) {
        System.out.println(event.getMessage());
      }
    });
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

    String currentInput = "";
    String line;
    System.out.print(">> ");
    try {
      while ((line = br.readLine()) != null) {
        if (line.isEmpty()) {
          exec(currentInput, env);
          currentInput = "";
          System.out.print(">> ");
        } else {
          currentInput = currentInput + "\n" + line;
          System.out.print(".. ");
        }
      }
    } catch (IOException io) {
      io.printStackTrace();
    }
  }
}
