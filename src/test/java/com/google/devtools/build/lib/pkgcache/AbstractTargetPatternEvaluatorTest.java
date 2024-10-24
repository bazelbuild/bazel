// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Predicates;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.DelegatingEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;
import org.junit.Before;

/**
 * Abstract framework for target pattern evaluation tests. The {@link TargetPatternEvaluatorTest}
 * contains much of the functionality that might be needed for future tests, and its methods should
 * be extracted here if they are needed by other classes.
 */
public abstract class AbstractTargetPatternEvaluatorTest extends PackageLoadingTestCase {
  protected TargetPatternPreloader parser;
  protected RecordingParsingListener parsingListener;

  protected static ResolvedTargets<Target> parseTargetPatternList(
      TargetPatternPreloader parser,
      ExtendedEventHandler eventHandler,
      List<String> targetPatterns,
      boolean keepGoing)
      throws TargetParsingException, InterruptedException {
    return parseTargetPatternList(
        PathFragment.EMPTY_FRAGMENT, parser, eventHandler, targetPatterns, keepGoing);
  }

  protected static ResolvedTargets<Target> parseTargetPatternList(
      PathFragment relativeWorkingDirectory,
      TargetPatternPreloader parser,
      ExtendedEventHandler eventHandler,
      List<String> targetPatterns,
      boolean keepGoing)
      throws TargetParsingException, InterruptedException {
    List<String> positivePatterns =
        targetPatterns.stream()
            .map((s) -> s.startsWith("-") ? s.substring(1) : s)
            .collect(Collectors.toList());
    Map<String, Collection<Target>> resolvedTargetsMap =
        parser.preloadTargetPatterns(
            eventHandler,
            TargetPattern.mainRepoParser(relativeWorkingDirectory),
            positivePatterns,
            keepGoing);
    ResolvedTargets.Builder<Target> result = ResolvedTargets.builder();
    for (String pattern : targetPatterns) {
      if (pattern.startsWith("-")) {
        String positivePattern = pattern.substring(1);
        Collection<Target> resolvedTargets = resolvedTargetsMap.get(positivePattern);
        result.filter(Predicates.not(Predicates.in(resolvedTargets)));
      } else {
        Collection<Target> resolvedTargets = resolvedTargetsMap.get(pattern);
        result.addAll(resolvedTargets);
      }
    }
    return result.build();
  }

  /**
   * Method converts collection of targets to the new, mutable, lexicographically-ordered set of
   * corresponding labels.
   */
  protected static Set<Label> targetsToLabels(Iterable<Target> targets) {
    Set<Label> labels = new TreeSet<>();
    for (Target target : targets) {
      labels.add(target.getLabel());
    }
    return labels;
  }

  @Before
  public final void initializeParser() throws Exception {
    setUpSkyframe(RuleVisibility.PRIVATE);
    parser = skyframeExecutor.newTargetPatternPreloader();
    parsingListener = new RecordingParsingListener(reporter);
  }

  protected static Set<Label> labels(String... labelStrings) throws LabelSyntaxException {
    Set<Label> labels = new HashSet<>();
    for (String labelString : labelStrings) {
      labels.add(Label.parseCanonical(labelString));
    }
    return labels;
  }

  protected Pair<Set<Label>, Boolean> parseListKeepGoing(String... patterns)
      throws TargetParsingException, InterruptedException {
    ResolvedTargets<Target> result =
        parseTargetPatternList(parser, parsingListener, Arrays.asList(patterns), true);
    return Pair.of(targetsToLabels(result.getTargets()), result.hasError());
  }

  /** Event handler that records all parsing errors. */
  protected static final class RecordingParsingListener extends DelegatingEventHandler {
    protected final List<Pair<String, String>> events = new ArrayList<>();

    private RecordingParsingListener(ExtendedEventHandler delegate) {
      super(delegate);
    }

    @Override
    public void post(Postable post) {
      super.post(post);
      if (post instanceof ParsingFailedEvent e) {
        events.add(Pair.of(e.getPattern(), e.getMessage()));
      }
    }

    protected void assertEmpty() {
      assertThat(events).isEmpty();
    }
  }
}
