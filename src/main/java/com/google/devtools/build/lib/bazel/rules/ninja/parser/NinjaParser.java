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

import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteFragmentAtOffset;
import com.google.devtools.build.lib.bazel.rules.ninja.file.DeclarationConsumer;
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

  public NinjaParser(
      NinjaPipeline pipeline, NinjaFileParseResult parseResult, String ninjaFileName) {
    this.pipeline = pipeline;
    this.parseResult = parseResult;
    this.ninjaFileName = ninjaFileName;
  }

  @Override
  public void declaration(ByteFragmentAtOffset byteFragmentAtOffset)
      throws GenericParsingException, IOException {
    ByteBufferFragment fragment = byteFragmentAtOffset.getFragment();
    int offset = byteFragmentAtOffset.getRealStartOffset();

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
    int declarationStart = offset + lexer.getLastStart();
    lexer.undo();
    NinjaParserStep parser = new NinjaParserStep(lexer);

    switch (token) {
      case IDENTIFIER:
        Pair<String, NinjaVariableValue> variable = parser.parseVariable();
        parseResult.addVariable(variable.getFirst(), declarationStart, variable.getSecond());
        break;
      case RULE:
        NinjaRule rule = parser.parseNinjaRule();
        parseResult.addRule(declarationStart, rule);
        break;
      case ERROR:
        throw new GenericParsingException(lexer.getError());
      case ZERO:
      case EOF:
        return;
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
        ByteFragmentAtOffset targetFragment;
        if (declarationStart == offset) {
          targetFragment = byteFragmentAtOffset;
        } else {
          // Method subFragment accepts only the offset *inside fragment*.
          // So we should subtract the offset of fragment's buffer in file
          // (byteFragmentAtOffset.getOffset()),
          // and start of fragment inside buffer (fragment.getStartIncl()).
          int fragmentStart =
              declarationStart - byteFragmentAtOffset.getOffset() - fragment.getStartIncl();
          targetFragment =
              new ByteFragmentAtOffset(
                  byteFragmentAtOffset.getOffset(),
                  fragment.subFragment(fragmentStart, fragment.length()));
        }
        parseResult.addTarget(targetFragment);
        break;
      case DEFAULT:
      case POOL:
        // Do nothing.
        break;
      default:
        throw new UnsupportedOperationException("Unknown type of Ninja token.");
    }
  }
}
