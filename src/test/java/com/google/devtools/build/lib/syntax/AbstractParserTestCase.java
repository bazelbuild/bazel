// Copyright 2006-2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.rules.SkylarkModules;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FsApparatus;

import java.util.List;

/**
 * Base class for test cases that use parsing services.
 */
public abstract class AbstractParserTestCase {
  public static final class EmptyPackageLocator implements CachingPackageLocator {
    @Override
    public Path getBuildFileForPackage(String packageName) {
      return null;
    }
  }

  protected EventCollectionApparatus syntaxEvents = new EventCollectionApparatus();
  private FsApparatus scratch = FsApparatus.newInMemory();
  private CachingPackageLocator locator = new EmptyPackageLocator();

  private static Lexer createLexer(String input,
      EventCollectionApparatus syntaxEvents, FsApparatus scratch) {
    Path someFile = scratch.path("/some/file.txt");
    ParserInputSource inputSource = ParserInputSource.create(input, someFile);
    return new Lexer(inputSource, syntaxEvents.reporter());
  }

  protected Lexer createLexer(String input) {
    return createLexer(input, syntaxEvents, scratch);
  }

  protected List<Statement> parseFile(String input) {
    return Parser.parseFile(createLexer(input), syntaxEvents.reporter(), locator, false)
        .statements;
  }

  protected List<Statement> parseFile(String input, boolean parsePython) {
    return Parser.parseFile(createLexer(input), syntaxEvents.reporter(), locator, parsePython)
        .statements;
  }

  protected List<Statement> parseFileForSkylark(String input) {
    return Parser.parseFileForSkylark(createLexer(input), syntaxEvents.reporter(), locator,
        SkylarkModules.getValidationEnvironment()).statements;
  }

  protected List<Statement> parseFileForSkylark(
      String input, ImmutableMap<String, SkylarkType> extraObject) {
    return Parser.parseFileForSkylark(createLexer(input), syntaxEvents.reporter(), locator,
        SkylarkModules.getValidationEnvironment(extraObject)).statements;
  }

  protected Parser.ParseResult parseFileWithComments(String input) {
    return Parser.parseFile(createLexer(input), syntaxEvents.reporter(), locator, false);
  }

  protected Statement parseStmt(String input) {
    return Parser.parseStatement(createLexer(input), syntaxEvents.reporter());
  }

  protected Expression parseExpr(String input) {
    return Parser.parseExpression(createLexer(input), syntaxEvents.reporter());
  }

  public static List<Statement> parseFileForSkylark(
      EventCollectionApparatus syntaxEvents, FsApparatus scratch, String input) {
    return Parser.parseFileForSkylark(createLexer(input, syntaxEvents, scratch),
        syntaxEvents.reporter(), null,
        SkylarkModules.getValidationEnvironment()).statements;
  }

  public static List<Statement> parseFileForSkylark(
      EventCollectionApparatus syntaxEvents, FsApparatus scratch, String input,
      ImmutableMap<String, SkylarkType> extraObject) {
    return Parser.parseFileForSkylark(createLexer(input, syntaxEvents, scratch),
        syntaxEvents.reporter(), null,
        SkylarkModules.getValidationEnvironment(extraObject)).statements;
  }
}
