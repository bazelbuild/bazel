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

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.devtools.build.lib.bazel.rules.ninja.file.DeclarationConsumer;
import com.google.devtools.build.lib.bazel.rules.ninja.file.FileFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaToken;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaFileParseResult.NinjaPromise;
import com.google.devtools.build.lib.bazel.rules.ninja.pipeline.NinjaPipeline;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;

/**
 * Ninja parser: an instance will be created per a fragment of Ninja file, to avoid synchronization
 * while parsing independent values.
 *
 * <p>Populates the {@link NinjaFileParseResult} with variables and rules.
 */
public class NinjaParser implements DeclarationConsumer {
  private final NinjaPipeline pipeline;
  private final NinjaFileParseResult parseResult;
  private final String ninjaFileName;

  private static final String UNSUPPORTED_TOKEN_MESSAGE =
      " is an unsupported type of Ninja lexeme for the parser at this state. "
          + "These are unexpected lexemes at this state for the parser, and running into a "
          + "token of this lexeme hints that the parser did not progress the lexer to a "
          + "valid state during a previous parsing step.";

  public NinjaParser(
      NinjaPipeline pipeline, NinjaFileParseResult parseResult, String ninjaFileName) {
    this.pipeline = pipeline;
    this.parseResult = parseResult;
    this.ninjaFileName = ninjaFileName;
  }

  @Override
  public void declaration(FileFragment fragment) throws GenericParsingException, IOException {
    long offset = fragment.getFragmentOffset();

    NinjaLexer lexer = new NinjaLexer(fragment);
    if (!lexer.hasNextToken()) {
      throw new IllegalStateException("Empty fragment passed as declaration.");
    }
    NinjaToken token = lexer.nextToken();
    // Skip possible leading newlines in the fragment for parsing.
    while (lexer.hasNextToken() && NinjaToken.NEWLINE.equals(token)) {
      token = lexer.nextToken();
    }
    if (!lexer.hasNextToken()) {
      // If fragment contained only newlines.
      return;
    }
    long declarationStart = offset + lexer.getLastStart();
    lexer.undo();
    NinjaParserStep parser =
        new NinjaParserStep(lexer, pipeline.getPathFragmentInterner(), pipeline.getNameInterner());

    switch (token) {
      case IDENTIFIER:
        Pair<String, NinjaVariableValue> variable = parser.parseVariable();
        parseResult.addVariable(variable.getFirst(), declarationStart, variable.getSecond());
        break;
      case RULE:
        NinjaRule rule = parser.parseNinjaRule();
        parseResult.addRule(declarationStart, rule);
        break;
      case POOL:
        parseResult.addPool(declarationStart, parser.parseNinjaPool());
        break;
      case INCLUDE:
        NinjaVariableValue includeStatement = parser.parseIncludeStatement();
        NinjaPromise<NinjaFileParseResult> includeFuture =
            pipeline.createChildFileParsingPromise(
                includeStatement, declarationStart, ninjaFileName);
        parseResult.addIncludeScope(declarationStart, includeFuture);
        break;
      case SUBNINJA:
        NinjaVariableValue subNinjaStatement = parser.parseSubNinjaStatement();
        NinjaPromise<NinjaFileParseResult> subNinjaFuture =
            pipeline.createChildFileParsingPromise(
                subNinjaStatement, declarationStart, ninjaFileName);
        parseResult.addSubNinjaScope(declarationStart, subNinjaFuture);
        break;
      case BUILD:
        FileFragment targetFragment;
        if (declarationStart == offset) {
          targetFragment = fragment;
        } else {
          // Method subFragment accepts only the offset *inside fragment*.
          // So we should subtract the offset of fragment's buffer in file
          // (fragment.getFileOffset()),
          // and start of fragment inside buffer (fragment.getStartIncl()).
          long fragmentStart =
              declarationStart - fragment.getFileOffset() - fragment.getStartIncl();

          // While the absolute offset is typed as long (because of larger ninja files), the
          // fragments are only at most Integer.MAX_VALUE long, so fragmentStart cannot be
          // larger than that. Check this here.
          if (fragmentStart > Integer.MAX_VALUE) {
            throw new GenericParsingException(
                String.format(
                    "The fragmentStart value %s is not expected to be larger than max-int, "
                        + "since each fragment is at most max-int long.",
                    fragmentStart));
          }
          targetFragment = fragment.subFragment((int) fragmentStart, fragment.length());
        }
        parseResult.addTarget(targetFragment);
        break;
      case DEFAULT:
        // Do nothing.
        break;
      case ZERO:
      case EOF:
        return;
      case COLON:
      case EQUALS:
      case ESCAPED_TEXT:
      case INDENT:
      case NEWLINE:
      case PIPE:
      case PIPE2:
      case PIPE_AT:
      case TEXT:
      case VARIABLE:
        throw new UnsupportedOperationException(token.name() + UNSUPPORTED_TOKEN_MESSAGE);
      case ERROR:
        throw new GenericParsingException(lexer.getError());
        // Omit default case on purpose. Explicitly specify *all* NinjaToken enum cases above or the
        // compilation will fail.
    }
  }
}
