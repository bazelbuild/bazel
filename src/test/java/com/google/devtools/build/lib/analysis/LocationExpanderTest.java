// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.LocationExpander.LocationFunction;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link LocationExpander}. */
@RunWith(JUnit4.class)
public class LocationExpanderTest {
  private static final class Capture implements RuleErrorConsumer {
    private final List<String> warnsOrErrors = new ArrayList<>();

    @Override
    public void ruleWarning(String message) {
      warnsOrErrors.add("WARN: " + message);
    }

    @Override
    public void ruleError(String message) {
      warnsOrErrors.add("ERROR: " + message);
    }

    @Override
    public void attributeWarning(String attrName, String message) {
      warnsOrErrors.add("WARN-" + attrName + ": " + message);
    }

    @Override
    public void attributeError(String attrName, String message) {
      warnsOrErrors.add("ERROR-" + attrName + ": " + message);
    }

    @Override
    public boolean hasErrors() {
      return !warnsOrErrors.isEmpty();
    }
  }

  private LocationExpander makeExpander(RuleErrorConsumer ruleErrorConsumer) throws Exception {
    LocationFunction f1 =
        new LocationFunctionBuilder("//a", false)
            .setPathType(LocationFunction.PathType.LOCATION)
            .add("//a", "/exec/src/a")
            .build();

    LocationFunction f2 =
        new LocationFunctionBuilder("//b", true)
            .setPathType(LocationFunction.PathType.LOCATION)
            .add("//b", "/exec/src/b")
            .build();

    return new LocationExpander(
        ruleErrorConsumer,
        ImmutableMap.<String, LocationFunction>of(
            "location", f1,
            "locations", f2),
        RepositoryMapping.EMPTY,
        "workspace");
  }

  private String expand(String input) throws Exception {
    return makeExpander(new Capture()).expand(input);
  }

  @Test
  public void noExpansion() throws Exception {
    assertThat(expand("abc")).isEqualTo("abc");
  }

  @Test
  public void oneOrMore() throws Exception {
    assertThat(expand("$(location a)")).isEqualTo("src/a");
    assertThat(expand("$(locations b)")).isEqualTo("src/b");
    assertThat(expand("---$(location a)---")).isEqualTo("---src/a---");
  }

  @Test
  public void twoInOne() throws Exception {
    assertThat(expand("$(location a) $(locations b)")).isEqualTo("src/a src/b");
  }

  @Test
  public void notAFunction() throws Exception {
    assertThat(expand("$(locationz a)")).isEqualTo("$(locationz a)");
  }

  @Test
  public void missingClosingParen() throws Exception {
    Capture capture = new Capture();
    String value = makeExpander(capture).expand("foo $(location a");
    // In case of an error, no location expansion is performed.
    assertThat(value).isEqualTo("foo $(location a");
    assertThat(capture.warnsOrErrors).containsExactly("ERROR: unterminated $(location) expression");
  }

  // In case of errors, the exact return value is unspecified. However, we don't want to
  // accidentally change the behavior even in this unspecified case - that's why I added a test
  // here.
  @Test
  public void noExpansionOnError() throws Exception {
    Capture capture = new Capture();
    String value = makeExpander(capture).expand("foo $(location a) $(location a");
    assertThat(value).isEqualTo("foo $(location a) $(location a");
    assertThat(capture.warnsOrErrors).containsExactly("ERROR: unterminated $(location) expression");
  }

  @Test
  public void expansionWithRepositoryMapping() throws Exception {
    LocationFunction f1 =
        new LocationFunctionBuilder("//a", false)
            .setPathType(LocationFunction.PathType.LOCATION)
            .add("@bar//a", "/exec/src/a")
            .build();

    ImmutableMap<String, RepositoryName> repositoryMapping =
        ImmutableMap.of("foo", RepositoryName.create("bar"));

    LocationExpander locationExpander =
        new LocationExpander(
            new Capture(),
            ImmutableMap.<String, LocationFunction>of("location", f1),
            RepositoryMapping.create(repositoryMapping, RepositoryName.MAIN),
            "workspace");

    String value = locationExpander.expand("$(location @foo//a)");
    assertThat(value).isEqualTo("src/a");
  }
}
