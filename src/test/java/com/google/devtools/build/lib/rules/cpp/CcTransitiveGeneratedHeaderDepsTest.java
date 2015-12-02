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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.util.StringUtilities.joinLines;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCaseForJunit4;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Tests how generated header dependencies make up a little middlemen +
 * headers DAG which hangs off of cc_library nodes.
 */
@RunWith(JUnit4.class)
public class CcTransitiveGeneratedHeaderDepsTest extends BuildViewTestCaseForJunit4 {

  @Before
  public final void writeFiles() throws Exception {
    scratch.file("foo/BUILD", "cc_library(name = 'foo',",
                                      "          srcs = ['foo.cc'],",
                                      "          deps = ['//bar', '//boo'])");

    scratch.file("bar/BUILD", "cc_library(name = 'bar',",
                                      "           srcs = ['bar.cc',",
                                      "                   'bargen.h',",
                                      "                   'bargen2.h'],",
                                      "           deps = ['//baz', '//boo'])",
                                      "genrule(name = 'generated',",
                                      "        srcs = ['gen.py'],",
                                      "        cmd = 'gen.py',",
                                      "        outs = ['bargen.h'])",
                                      "genrule(name = 'generated2',",
                                      "        srcs = ['gen.py'],",
                                      "        cmd = 'gen.py',",
                                      "        outs = ['bargen2.h'])");

    scratch.file("baz/BUILD", "cc_library(name = 'baz',",
                                      "           srcs = ['baz.cc',",
                                      "                   'bazgen.h'])",
                                      "genrule(name = 'generated',",
                                      "        srcs = ['gen.py'],",
                                      "        cmd = 'gen.py',",
                                      "        outs = ['bazgen.h'])");

    scratch.file("boo/BUILD", "cc_library(name = 'boo',",
                                      "           srcs = ['boo.cc',",
                                      "                   'boogen.h'])",
                                      "genrule(name = 'boogen',",
                                      "        srcs = ['gen.py'],",
                                      "        cmd = 'gen.py',",
                                      "        outs = ['boogen.h'])");
  }

  private ConfiguredTarget createTargets() throws Exception {
    getConfiguredTarget("//bar:bar");
    getConfiguredTarget("//baz:baz");
    return getConfiguredTarget("//foo:foo");
  }

  private ConfiguredTarget setupWithOptions(String... optStrings) throws Exception {
    useConfiguration(optStrings);
    return createTargets();
  }

  @Test
  public void testQuoteIncludeDirs() throws Exception {
    ConfiguredTarget fooLib = setupWithOptions();
    ImmutableList<PathFragment> quoteIncludeDirs =
        fooLib.getProvider(CppCompilationContext.class).getQuoteIncludeDirs();
    assertThat(filterExternalBazelTools(quoteIncludeDirs)).containsExactly(
        PathFragment.EMPTY_FRAGMENT, targetConfig.getGenfilesFragment()).inOrder();
  }

  protected static Iterable<PathFragment> filterExternalBazelTools(
      ImmutableList<PathFragment> quoteIncludeDirs) {
    return Iterables.filter(
        quoteIncludeDirs,
        new Predicate<PathFragment>() {
          @Override
          public boolean apply(PathFragment pathFragment) {
            return !pathFragment.endsWith(new PathFragment("external/bazel_tools"));
          }
        });
  }

  @Test
  public void testGeneratesTreeOfMiddlemenAndGeneratedHeaders() throws Exception {
    ConfiguredTarget fooLib = setupWithOptions("--noextract_generated_inclusions");
    Set<Artifact> middlemen = fooLib.getProvider(CppCompilationContext.class)
        .getCompilationPrerequisites();

    // While normal middlemen are not created if they are depend on just one
    // input, C++ compilation dependencies are expressed using scheduling
    // middleman where one input optimization no longer applies.
    assertEquals(joinLines(
        "middleman-0",
        "  middleman-1",
        "    bargen.h",
        "    bargen2.h",
        "    middleman-2",
        "      bazgen.h",
        "    middleman-3",
        "      boogen.h",
        "  middleman-3",
        "    boogen.h",
        ""), new MiddlemenRenderer(middlemen).toString());
  }

  @Test
  public void testExtractInclusionsInActionGraph() throws Exception {
    ConfiguredTarget fooLib = setupWithOptions("--extract_generated_inclusions");

    Set<Artifact> middlemen = fooLib.getProvider(CppCompilationContext.class)
        .getCompilationPrerequisites();

    // While normal middlemen are not created if they are depend on just one
    // input, C++ compilation dependencies are expressed using scheduling
    // middleman where one input optimization no longer applies.
    assertEquals(joinLines(
        "middleman-0",
        "  middleman-1",
        "    bargen.h.includes",
        "    bargen2.h.includes",
        "    bargen.h",
        "    bargen2.h",
        "    middleman-2",
        "      bazgen.h.includes",
        "      bazgen.h",
        "    middleman-3",
        "      boogen.h.includes",
        "      boogen.h",
        "  middleman-3",
        "    boogen.h.includes",
        "    boogen.h",
        ""), new MiddlemenRenderer(middlemen).toString());

    List<Artifact> nonMiddlemen = new ArrayList<>();
    getRealArtifacts(middlemen, nonMiddlemen);
    assertThat(nonMiddlemen).isNotEmpty();
    Iterable<Artifact> includes = Iterables.filter(nonMiddlemen, new Predicate<Artifact>(){
      @Override
      public boolean apply(Artifact artifact) {
        return artifact.getExecPathString().endsWith(".h.includes");
      }
    });
    assertThat(includes).isNotEmpty();
    for (Artifact file : nonMiddlemen) {
      assertNotNull(file.getExecPathString(), getGeneratingAction(file));
      assertFalse(file.getExecPathString(), file.isSourceArtifact());
    }

    Iterable<Artifact> pregreppedArtifacts =
        Iterables.transform(fooLib
            .getProvider(CppCompilationContext.class)
            .getPregreppedHeaders().toCollection(), Pair.<Artifact, Artifact>secondFunction());
    Iterable<String> pregreppedFiles = Iterables.transform(pregreppedArtifacts,
        new Function<Artifact, String>() {
          @Override
          public String apply(Artifact input) {
            return input.getPath().getBaseName();
          }
        });
    assertThat(pregreppedFiles).containsExactly("bargen.h.includes", "bargen2.h.includes",
        "bazgen.h.includes", "boogen.h.includes");
  }

  private void getRealArtifacts(Iterable<Artifact> middlemenOrHeaders,
      List<Artifact> artifacts) {
    for (Artifact file : middlemenOrHeaders) {
      if (isMiddleman(file)) {
        getRealArtifacts(getGeneratingAction(file).getInputs(), artifacts);
      } else {
        artifacts.add(file);
      }
    }
  }

  private static boolean isMiddleman(Artifact artifact) {
    return artifact.getExecPath().toString().contains("_middlemen");
  }

  /**
   * Renders a little tree representation of a set of middlemen and headers
   * into a string.
   */
  private class MiddlemenRenderer {

    /** The rendering buffer */
    StringBuilder buffer = new StringBuilder();

    /** How much to indent while we're emitting to buffer */
    int indent = 0;

    /** Middlemen have weird path names, so we map them to integer ids */
    int nextId = 0;
    Map<Artifact, Integer> middlemanToIdOld = new HashMap<>();

    MiddlemenRenderer(Set<Artifact> middlemen) {
      print(middlemen);
    }

    void print(Collection<Artifact> middlemenOrHeaders) {
      for (Artifact middleman : middlemenOrHeaders) {
        printIndent();
        if (isMiddleman(middleman)) {
          printMiddleman(middleman);
          indent += 2;
          List<Artifact> inputs = Ordering.from(Artifact.EXEC_PATH_COMPARATOR)
              .sortedCopy(getGeneratingAction(middleman).getInputs());
          print(inputs);
          indent -= 2;
        } else {
          printPrerequisite(middleman);
        }
      }
    }

    private void printPrerequisite(Artifact header) {
       buffer.append(header.getExecPath().getBaseName() + "\n");
    }

    void printIndent() {
      for (int i = 0; i < indent; i++) {
        buffer.append(' ');
      }
    }

    void printMiddleman(Artifact middleman) {
      if (!middlemanToIdOld.containsKey(middleman)) {
        middlemanToIdOld.put(middleman, nextId++);
      }
      int id = middlemanToIdOld.get(middleman);
      buffer.append("middleman-" + id + "\n");
    }

    @Override
    public String toString() {
      return buffer.toString();
    }
  }

}
