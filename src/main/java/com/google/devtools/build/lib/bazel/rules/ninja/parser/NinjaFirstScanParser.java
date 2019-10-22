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

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bazel.rules.ninja.file.AbstractDeclarationConsumerFactory;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.DeclarationConsumer;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.vfs.Path;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

public class NinjaFirstScanParser implements DeclarationConsumer {
  private static final ImmutableSortedMap<byte[], NinjaKeyword> ALL_KEYWORDS_BYTES =
      ImmutableSortedMap.copyOf(
          Arrays.stream(NinjaKeyword.values())
              .collect(Collectors.toMap(NinjaKeyword::getBytes, Function.identity())));
  private final static byte COMMENT_BYTE = (byte) ('#' & 0xff);

  private final List<ByteBufferFragment> targets;
  private final Map<String, String> variables;
  private final Map<String, NinjaRule> rules;
  private final Path basePath;
  private final Consumer<Path> fileProcessingScheduler;
  private final Charset charset;

  private NinjaFirstScanParser(
      Path basePath,
      Consumer<Path> fileProcessingScheduler,
      Charset charset) {
    this.basePath = basePath;
    this.fileProcessingScheduler = fileProcessingScheduler;
    this.charset = charset;
    targets = Collections.synchronizedList(Lists.newArrayList());
    variables = Maps.synchronizedNavigableMap(Maps.newTreeMap());
    rules = Maps.synchronizedNavigableMap(Maps.newTreeMap());
  }

  @Override
  public void declaration(ByteBufferFragment fragment) throws GenericParsingException {
    byte[] firstWordFragment = NinjaLineSplitterUtil.getFirstWordFragment(fragment);
    Preconditions.checkState(firstWordFragment.length > 0);

    if (COMMENT_BYTE == firstWordFragment[0]) {
      return;
    }

    NinjaKeyword keyword = getKeyword(firstWordFragment);
    if (keyword == null) {
      if (useParser(NinjaVariableParser.INSTANCE, null, fragment,
          pair -> variables.put(pair.getFirst(), pair.getSecond()))) {
        return;
      }
      throw new GenericParsingException(String.format("Unknown line start: '%s'",
          new String(firstWordFragment, charset)));
    }

    if (useParser(NinjaIncludeParser.INSTANCE, keyword, fragment,
        pf -> fileProcessingScheduler.accept(basePath.getRelative(pf)))) {
      return;
    }
    if (useParser(NinjaRuleParser.INSTANCE, keyword, fragment,
        rule -> rules.put(rule.getName(), rule))) {
      return;
    }
    if (NinjaKeyword.build.equals(keyword)) {
      targets.add(fragment);
    }
  }

  private <T> boolean useParser(
      NinjaDeclarationParser<T> parser,
      @Nullable NinjaKeyword keyword,
      ByteBufferFragment fragment,
      Consumer<T> consumer) throws GenericParsingException {
    if (keyword == null && parser.getKeywords().isEmpty()
        || parser.getKeywords().contains(keyword)) {
      List<String> lines = NinjaLineSplitterUtil.splitIntoLines(fragment, charset);
      T value = parser.parse(lines);
      consumer.accept(value);
      return true;
    }
    return false;
  }

  @Nullable
  private static NinjaKeyword getKeyword(byte[] bytes) {
    NinjaKeyword keyword = ALL_KEYWORDS_BYTES.get(bytes);
    if (keyword != null && Arrays.equals(keyword.getBytes(), bytes)) {
      return keyword;
    }
    return null;
  }

  public List<ByteBufferFragment> getTargets() {
    return targets;
  }

  public Map<String, String> getVariables() {
    return variables;
  }

  public Map<String, NinjaRule> getRules() {
    return rules;
  }

  public static class Factory extends AbstractDeclarationConsumerFactory<NinjaFirstScanParser> {
    private final Path basePath;
    private final Consumer<Path> fileProcessingScheduler;
    private final Charset charset;

    public Factory(Path basePath,
        Consumer<Path> fileProcessingScheduler, Charset charset) {
      this.basePath = basePath;
      this.fileProcessingScheduler = fileProcessingScheduler;
      this.charset = charset;
    }

    @Override
    protected NinjaFirstScanParser createParserImpl() {
      return new NinjaFirstScanParser(basePath, fileProcessingScheduler, charset);
    }
  }
}
