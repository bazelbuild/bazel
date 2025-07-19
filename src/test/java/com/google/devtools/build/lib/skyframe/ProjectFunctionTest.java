// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.PathFragmentPrefixTrie;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.EvaluationResult;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ProjectFunctionTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    writeProjectSclDefinition("test/project_proto.scl");
  }

  @Test
  public void projectFunction_emptyFile_isValid() throws Exception {
    scratch.file("test/PROJECT.scl", "project = {}");
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isFalse();

    ProjectValue value = result.get(key);
    assertThat(value.getDefaultProjectDirectories()).isEmpty();
  }

  @Test
  public void projectFunction_returnsActiveDirectories() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "active_directories": {'default': ['foo'], 'a': ['bar', '-bar/baz']},
        }
        """);
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isFalse();

    ProjectValue value = result.get(key);
    ImmutableMap<String, PathFragmentPrefixTrie> trie =
        PathFragmentPrefixTrie.transformValues(value.getProjectDirectories());
    assertThat(trie.get("default").includes(PathFragment.create("foo"))).isTrue();
    assertThat(trie.get("default").includes(PathFragment.create("bar"))).isFalse();
    assertThat(trie.get("a").includes(PathFragment.create("bar"))).isTrue();
    assertThat(trie.get("a").includes(PathFragment.create("bar/baz"))).isFalse();
    assertThat(trie.get("a").includes(PathFragment.create("bar/qux"))).isTrue();
    assertThat(trie.get("b")).isNull();
  }

  @Test
  public void projectFunction_returnsDefaultActiveDirectories() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "active_directories": { 'default': ['a', 'b/c'] },
        }
        """);
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isFalse();

    ProjectValue value = result.get(key);
    PathFragmentPrefixTrie trie = PathFragmentPrefixTrie.of(value.getDefaultProjectDirectories());
    assertThat(trie.includes(PathFragment.create("a"))).isTrue();
    assertThat(trie.includes(PathFragment.create("b/c"))).isTrue();
    assertThat(trie.includes(PathFragment.create("d"))).isFalse();
  }

  @Test
  public void projectFunction_returnsDefaultActiveDirectories_topLevelProjectSchema()
      throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "active_directories": { "default": ["a", "b/c"] }
        }
        """);
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isFalse();

    ProjectValue value = result.get(key);
    PathFragmentPrefixTrie trie = PathFragmentPrefixTrie.of(value.getDefaultProjectDirectories());
    assertThat(trie.includes(PathFragment.create("a"))).isTrue();
    assertThat(trie.includes(PathFragment.create("b/c"))).isTrue();
    assertThat(trie.includes(PathFragment.create("d"))).isFalse();
  }

  @Test
  public void projectFunction_nonEmptyActiveDirectoriesMustHaveADefault() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "active_directories": { 'foo': ['a', 'b/c'] },
        }
        """);
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains("non-empty active_directories must contain the 'default' key");
  }

  @Test
  public void projectFunction_incorrectType() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "active_directories": 42,
        }
        """);
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .matches("expected a map of string to list of strings, got .+Int32");
  }

  @Test
  public void projectFunction_incorrectType_inList() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "active_directories": { 'default': [42] },
        }
        """);
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .matches("expected a list of strings, got element of .+Int32");
  }

  @Test
  public void projectFunction_incorrectProjectType() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        project = 1
        """);

    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .matches("project variable: expected a map of string to objects, got .+Int32");
  }

  @Test
  public void projectFunction_incorrectProjectKeyType() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {1: [] }
        """);

    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .matches("project variable: expected string key, got element of .+Int32");
  }

  @Test
  public void projectFunction_buildableUnitsFormat() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          enforcement_policy = "warn",
          project_directories = [ "//test/..."],
          always_allowed_configs = ["--config=foo"],
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "default",
                  target_patterns = [
                      "//test/...",
                  ],
                  description = "default",
                  flags = ["--define=foo=bar"],
                  is_default = True,
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "non_default",
                  target_patterns = [
                      "//test/...",
                  ],
                  description = "non default",
                  flags = ["--define=bar=baz"],
                  is_default = False,
              ),
          ],
        )
        """);

    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isFalse();
    ProjectValue value = result.get(key);
    assertThat(value.getEnforcementPolicy()).isEqualTo(ProjectValue.EnforcementPolicy.WARN);
    assertThat(value.getAlwaysAllowedConfigs()).isEqualTo(ImmutableList.of("--config=foo"));
    assertThat(value.getActualProjectFile()).isEqualTo(Label.parseCanonical("//test:PROJECT.scl"));
    assertThat(value.getBuildableUnits().get("default").isDefault()).isTrue();
    assertThat(value.getBuildableUnits().get("non_default").isDefault()).isFalse();
    assertThat(value.getProjectDirectories()).hasSize(1);
    assertThat(value.getProjectDirectories().get("default")).containsExactly("//test/...");

    assertThat(value.getBuildableUnits()).containsKey("default");
    assertThat(value.getBuildableUnits().get("default"))
        .isEqualTo(
            ProjectValue.BuildableUnit.create(
                ImmutableList.of("//test/..."),
                "default",
                ImmutableList.of("--define=foo=bar"),
                true));

    assertThat(value.getBuildableUnits()).containsKey("non_default");

    assertThat(value.getBuildableUnits().get("non_default"))
        .isEqualTo(
            ProjectValue.BuildableUnit.create(
                ImmutableList.of("//test/..."),
                "non default",
                ImmutableList.of("--define=bar=baz"),
                false));
  }

  @Test
  public void duplicateBuildableUnitNames() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(name = "foo"),
              buildable_unit_pb2.BuildableUnit.create(name = "foo"),
          ],
        )
        """);

    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));
    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);

    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .isEqualTo(
            "buildable_unit name='foo' is repeated. Buildable units must have unique names.");
  }

  @Test
  public void buildableUnitSchemaDefaults() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(name = "foo"),
          ],
        )
        """);

    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));
    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);

    assertThat(result.hasError()).isFalse();

    // Project-wide defaults:
    assertThat(result.get(key).getEnforcementPolicy())
        .isEqualTo(ProjectValue.EnforcementPolicy.WARN);
    assertThat(result.get(key).getAlwaysAllowedConfigs()).isNull();
    assertThat(result.get(key).getProjectDirectories()).hasSize(1);
    assertThat(result.get(key).getProjectDirectories().get("default")).isEmpty();

    // Buildable unit defaults:
    assertThat(result.get(key).getBuildableUnits().get("foo").targetPatternMatcher().isEmpty())
        .isTrue();
    assertThat(result.get(key).getBuildableUnits().get("foo").flags()).isEmpty();
    assertThat(result.get(key).getBuildableUnits().get("foo").description()).isEqualTo("foo");
    assertThat(result.get(key).getBuildableUnits().get("foo").isDefault()).isFalse();
  }

  /** Asserts that a PROJECT.scl with the given contexts fails with the given message. */
  private void assertParseError(String projectFileContents, String expectedError) throws Exception {
    scratch.file("test/PROJECT.scl", projectFileContents);
    scratch.file("test/BUILD");

    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));
    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);

    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException()).hasMessageThat().matches(expectedError);
  }

  @Test
  public void projectNameTypeError_notCurrentlyParsedSoNotYetAnError() throws Exception {
    // TODO: b/415068036 - update when/if we start reading project(name = "foo") to catch type
    // errors.
    scratch.file(
        "test/PROJECT.scl",
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = 123,
          buildable_units = [],
        )
        """);

    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));
    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);

    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void buildableUnitFieldTypeError() throws Exception {
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = "bad value",
        )
        """,
        "buildable_units must be a list of buildable unit definitions, got .*String");
  }

  @Test
  public void builableUnitEntryTypeError() throws Exception {
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              "bad value",
          ],
        )
        """,
        "buildable_units entries must be structured objects, got .*String");
  }

  @Test
  public void buildableUnitNameTypeError() throws Exception {
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = 2,
                  target_patterns = [
                      "//test/...",
                  ],
                  description = "default",
                  flags = ["--define=foo=bar"],
                  is_default = True,
              ),
          ],
        )
        """,
        "buildable_unit names must be strings, got .*Int32");
  }

  @Test
  public void buildableUnitTargetPatternsFieldTypeError() throws Exception {
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "default",
                  target_patterns = 1,
                  description = "default",
                  flags = ["--define=foo=bar"],
                  is_default = True,
              ),
          ],
        )
        """,
        "target_patterns must be a list of strings, got .*Int32");
  }

  @Test
  public void buildableUnitTargetPatternsEntryTypeError() throws Exception {
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "default",
                  target_patterns = [2, 3],
                  description = "default",
                  flags = ["--define=foo=bar"],
                  is_default = True,
              ),
          ],
        )
        """,
        "target_patterns entries must be strings, got .*Int32");
  }

  @Test
  public void buildableUnitDescriptionTypeError() throws Exception {
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "default",
                  target_patterns = [
                      "//test/...",
                  ],
                  description = 123,
                  flags = ["--define=foo=bar"],
                  is_default = True,
              ),
          ],
        )
        """,
        "buildable_unit descriptions must be strings, got .*Int32");
  }

  @Test
  public void buildableUnitFlagsFieldTypeError() throws Exception {
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "default",
                  target_patterns = [
                      "//test/...",
                  ],
                  description = "default",
                  flags = "bad value",
                  is_default = True,
              ),
          ],
        )
        """,
        "flags must be a list of strings, got .*String");
  }

  @Test
  public void buildableUnitFlagsEntryTypeError() throws Exception {
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "default",
                  target_patterns = [
                      "//test/...",
                  ],
                  description = "default",
                  flags = [123],
                  is_default = True,
              ),
          ],
        )
        """,
        "flags entries must be strings, got .*Int32");
  }

  @Test
  public void buildableUnitIsDefaultTypeError() throws Exception {
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "default",
                  target_patterns = [
                      "//test/...",
                  ],
                  description = "default",
                  flags = ["--define=foo=bar"],
                  is_default = "not valid",
              ),
          ],
        )
        """,
        "is_default must be a boolean, got .*String");
  }

  @Test
  public void unknownProjectFieldError() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          invalid_field = "not valid",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "default",
              ),
          ],
        )
        """,
        // The direct exception doesn't explain the cause but actual builds fail with more context:
        // "Error: project_project_Project() got unexpected keyword argument: invalid_field"
        "initialization of module 'test/PROJECT.scl' failed");
  }

  @Test
  public void unknownBuildableUntFieldError() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    assertParseError(
        """
        load(
            "//test:project_proto.scl",
            "buildable_unit_pb2",
            "project_pb2",
        )
        project = project_pb2.Project.create(
          name = "test",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "default",
                  invalid_field = "not valid",
              ),
          ],
        )
        """,
        // The direct exception doesn't explain the cause but actual builds fail with more context:
        // "Error: project_project_Project() got unexpected keyword argument: invalid_field"
        "initialization of module 'test/PROJECT.scl' failed");
  }

  @Test
  public void projectFunction_catchSyntaxError() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        something_is_wrong =
        """);
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    AssertionError e =
        assertThrows(
            AssertionError.class,
            () -> SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter));
    assertThat(e).hasMessageThat().contains("syntax error at 'newline': expected expression");
  }

}
