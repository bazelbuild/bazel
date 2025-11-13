// Copyright 2025 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class MaterializerRulesRealDepsTest extends AnalysisTestCase {

  @Before
  public void enableDormantDeps() throws Exception {
    useConfiguration(
        "--experimental_dormant_deps", "--incompatible_package_group_has_public_syntax");
  }

  @Before
  public void writeMaterializerRulesAllowlist() throws Exception {
    writeMaterializerRulesAllowlist(true, true);
  }

  public void writeMaterializerRulesAllowlist(
      boolean materializerRuleAllowed, boolean allowRealDeps) throws Exception {

    scratch.overwriteFile(
        TestConstants.TOOLS_REPOSITORY_SCRATCH
            + "tools/allowlists/materializer_rule_allowlist/BUILD",
        """
        package_group(
            name = 'materializer_rule_allowlist',
            packages = [%s],
        )

        package_group(
            name = 'materializer_rule_real_deps_allowlist',
            packages = [%s],
        )
        """
            .formatted(
                materializerRuleAllowed ? "\"public\"" : "", allowRealDeps ? "\"public\"" : ""));
  }

  private void writeBasicMaterializerRule() throws Exception {
    scratch.file(
        "defs.bzl",
"""
# Component ######################################

ComponentInfo = provider(fields = ["output", "info"])

def _component_impl(ctx):
    f = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(f, ctx.label.name)
    return ComponentInfo(output = f, info = ctx.attr.info)

component = rule(
    implementation = _component_impl,
    provides = [ComponentInfo],
    attrs = {
        "info": attr.string(),
    }
)

# Component selector #############################

def _component_selector_impl(ctx):
    selected = []
    for c in ctx.attr.all_components:
        if "yes" in c[ComponentInfo].info:
            selected.append(c)
    return MaterializedDepsInfo(deps = selected)

component_selector = materializer_rule(
    implementation = _component_selector_impl,
    allow_real_deps = True,
    attrs = {
        "all_components": attr.label_list(),
    },
)

# Binary #########################################

def _binary_impl(ctx):
    files = [dep[ComponentInfo].output for dep in ctx.attr.deps]
    return DefaultInfo(files = depset(direct = files))

binary = rule(
    implementation = _binary_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo]),
    },
)
""");
  }

  /** Tests materializing real deps through materializer rules. */
  @Test
  public void basicMaterializerRuleRealDeps_works() throws Exception {
    writeBasicMaterializerRule();

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

binary(
    name = "bin",
    deps = [
        ":aaa",
        ":component_selector",
        ":zzz",
    ],
)

component_selector(
    name = "component_selector",
    all_components = [
        ":a",
        ":b",
        ":c",
        ":d",
    ],
)

component(name = "aaa", info = "yes")
component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "no")
component(name = "d", info = "no")
component(name = "zzz", info = "yes")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    assertThat(ActionsTestUtil.baseArtifactNames(filesToBuild))
        .containsExactly("aaa.txt", "a.txt", "b.txt", "zzz.txt");
  }

  /** Tests that multiple materializer rules in an attribute works. */
  @Test
  public void multipleMaterializerRules_works() throws Exception {
    writeBasicMaterializerRule();

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

binary(
    name = "bin",
    deps = [
        ":aaa",
        ":component_selector",
        ":component_selector_2",
        ":zzz",
    ],
)

component_selector(
    name = "component_selector",
    all_components = [
        ":a",
        ":b",
        ":c",
        ":d",
    ],
)

component_selector(
    name = "component_selector_2",
    all_components = [
        ":e",
        ":f",
        ":g",
        ":h",
    ],
)

component(name = "aaa", info = "yes")
component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "no")
component(name = "d", info = "no")
component(name = "e", info = "yes")
component(name = "f", info = "yes")
component(name = "g", info = "no")
component(name = "h", info = "no")
component(name = "zzz", info = "yes")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    assertThat(ActionsTestUtil.baseArtifactNames(filesToBuild))
        .containsExactly("aaa.txt", "a.txt", "b.txt", "e.txt", "f.txt", "zzz.txt");
  }

  @Test
  public void multipleMaterializersReturnSameTarget_works() throws Exception {
    writeBasicMaterializerRule();

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

binary(
    name = "bin",
    deps = [
        ":aaa",
        ":component_selector",
        ":component_selector_2",
        ":zzz",
    ],
)

component_selector(
    name = "component_selector",
    all_components = [
        ":a",
        ":b", # <- overlap
        ":c", # <- overlap
        ":d",
        ":e",
    ],
)

component_selector(
    name = "component_selector_2",
    all_components = [
        ":b", # <- overlap
        ":c", # <- overlap
        ":f",
        ":g",
        ":h",
    ],
)

component(name = "aaa", info = "yes")
component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "yes")
component(name = "d", info = "no")
component(name = "e", info = "no")
component(name = "f", info = "yes")
component(name = "g", info = "no")
component(name = "h", info = "no")
component(name = "zzz", info = "yes")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    assertThat(ActionsTestUtil.baseArtifactNames(filesToBuild))
        .containsExactly("aaa.txt", "a.txt", "b.txt", "c.txt", "f.txt", "zzz.txt");
  }

  @Test
  public void materializerWithRealDepsNotInAllowlist_throwsError() throws Exception {

    writeMaterializerRulesAllowlist(true, false);

    writeBasicMaterializerRule();

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

binary(
    name = "bin",
    deps = [":component_selector"],
)

component_selector(
    name = "component_selector",
    all_components = [":a"],
)

component(name = "a", info = "yes")
""");

    this.reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//:bin"));
    assertContainsEvent(
        "in component_selector rule //:component_selector: Non-allowlisted use of real deps in "
            + "materializer target");
  }

  @Test
  public void aspectsThroughMaterializerRules_works() throws Exception {
    scratch.file(
        "defs.bzl",
"""
# Component ######################################

ComponentInfo = provider(fields = ["info"])

def _component_impl(ctx):
    return ComponentInfo(info = ctx.attr.info)

component = rule(
    implementation = _component_impl,
    provides = [ComponentInfo],
    attrs = {
        "info": attr.string(),
    }
)

# Component selector #############################

def _component_selector_impl(ctx):
    selected = []
    for c in ctx.attr.all_components:
        if "yes" in c[ComponentInfo].info:
            selected.append(c)
    return MaterializedDepsInfo(deps = selected)

component_selector = materializer_rule(
    implementation = _component_selector_impl,
    allow_real_deps = True,
    attrs = {
        "all_components": attr.label_list(),
    },
)

# Aspect #########################################

AspectInfo = provider(fields = ["info_artifact"])

def _mt_aspect_impl(target, ctx):
    if ctx.rule.kind != "component":
        return []
    artifact = ctx.actions.declare_file(target.label.name + ".info")
    ctx.actions.write(artifact, str(target.label))
    return AspectInfo(info_artifact = artifact)

mt_aspect = aspect(
    implementation = _mt_aspect_impl,
    attrs = {
        # Context creation code is shared between aspects and rules, so having this aspect
        # dependency in the test ensure that materializer dependency validation is only performed
        # for rules and not for aspects.
        "_tool": attr.label(
            default = Label("//:aspect_tool"),
            cfg = "exec",
        ),
    }
)

# Binary #########################################

def _binary_impl(ctx):
    for dep in ctx.attr.deps:
        print(dep[AspectInfo])
    return DefaultInfo(files = depset(direct = [dep[AspectInfo].info_artifact for dep in ctx.attr.deps]))

binary = rule(
    implementation = _binary_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo], aspects = [mt_aspect]),
    },
)
""");

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

binary(
    name = "bin",
    deps = [
        ":aaa",
        ":component_selector",
        ":zzz",
    ],
)

component_selector(
    name = "component_selector",
    all_components = [
        ":a",
        ":b",
        ":c",
        ":d",
    ],
)

component(name = "aaa", info = "yes")
component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "no")
component(name = "d", info = "no")
component(name = "zzz", info = "yes")

genrule(
    name = "aspect_tool",
    outs = ["tool"],
    executable = True,
    cmd = "echo 'touch $$1' > $@",
)
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    // The .info files come from the aspect, and only the files from the selected deps
    // should be returned.
    assertThat(ActionsTestUtil.baseArtifactNames(filesToBuild))
        .containsExactly("aaa.info", "a.info", "b.info", "zzz.info");
  }

  @Test
  public void aspectsOriginatingFromMaterializerRules_works() throws Exception {
    scratch.file(
        "defs.bzl",
"""
# Component ######################################

ComponentInfo = provider(fields = ["output", "deps_outputs", "include"])

def _component_impl(ctx):
    f = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(f, ctx.label.name)
    return ComponentInfo(
        output = f,
        deps_outputs = depset(
            direct = [f],
            transitive = [dep[ComponentInfo].deps_outputs for dep in ctx.attr.deps],
        ),
        include = ctx.attr.include,
    )

component = rule(
    implementation = _component_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo]),
        "include": attr.bool(),
    },
    provides = [ComponentInfo],
)

# Aspect #########################################

AspectInfo = provider(fields = ["include"])

def _mt_aspect_impl(target, ctx):
    include = False
    for dep in ctx.rule.attr.deps:
        if dep[AspectInfo].include or dep[ComponentInfo].include:
            include = True
            break
    return AspectInfo(include = include)

mt_aspect = aspect(
    implementation = _mt_aspect_impl,
    attr_aspects = ["deps"],
)

# Component selector #############################

def _component_selector_impl(ctx):

    components = []
    for component in ctx.attr.components:
        if AspectInfo in component:
            if component[AspectInfo].include:
                components.append(component)

    return MaterializedDepsInfo(deps = components)

component_selector = materializer_rule(
    implementation = _component_selector_impl,
    allow_real_deps = True,
    attrs = {
        "components": attr.label_list(aspects = [mt_aspect]),
    },
)

# Binary #########################################

def _binary_impl(ctx):
    files = [impl[ComponentInfo].deps_outputs for impl in ctx.attr.deps]
    return DefaultInfo(files = depset(transitive = files))

binary = rule(
    implementation = _binary_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo]),
    },
)
""");

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "binary", "component_selector", "component")

binary(
    name = "bin",
    deps = [":component_selector"],
)

component_selector(
    name = "component_selector",
    components = [":a", ":d", ":g"],
)

component(name = "a", deps = [":b", ":c"])
component(name = "b")
component(name = "c")

component(name = "d", deps = [":e", ":f"])
component(name = "e")
component(name = "f", include = True)

component(name = "g", deps = [":h", ":i"])
component(name = "h")
component(name = "i")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();

    // Only the artifacts from branches that have "include = True" somewhere in the DAG as collected
    // by the aspect are included.
    assertThat(ActionsTestUtil.baseArtifactNames(filesToBuild))
        .containsExactly("d.txt", "e.txt", "f.txt");
  }

  /**
   * Materializers-to-materializers aren't implemented to with dormant deps because they would need
   * to be recursively resolved within a single ConfiguredTarget function (and cycles would need to
   * be detected), but with "real deps" they're properly resolved at each level.
   */
  @Test
  public void materializerToMaterializer_works() throws Exception {

    scratch.file(
        "defs.bzl",
"""
#################################################
# Component

ComponentInfo = provider(fields = ["output", "info"])

def _component_impl(ctx):
    f = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(f, ctx.label.name)
    return ComponentInfo(output = f, info = ctx.attr.info)

component = rule(
    implementation = _component_impl,
    provides = [ComponentInfo],
    attrs = {
        "info": attr.string(),
    }
)

#################################################
# Component selector

def _component_selector_impl(ctx):
    selected = []
    for c in ctx.attr.all_components:
        if "yes" in c[ComponentInfo].info:
            selected.append(c)
    return MaterializedDepsInfo(deps = selected)

component_selector = materializer_rule(
    implementation = _component_selector_impl,
    allow_real_deps = True,
    attrs = {
        "all_components": attr.label_list(),
    },
)

#################################################
# Binary

def _binary_impl(ctx):
    files = [dep[ComponentInfo].output for dep in ctx.attr.deps]
    return DefaultInfo(files = depset(direct = files))

binary = rule(
    implementation = _binary_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo]),
    },
)
""");

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

binary(
    name = "bin",
    deps = [":component_selector"],
)

# Because the materializer handler code codes doesn't account for materializers materializing
# other materializers, component_selector2 would materialize "b" and "c" and "a" would be
# overridden here. So we throw an error.
component_selector(
    name = "component_selector",
    all_components = [":component_selector2", ":a"],
)

component_selector(
    name = "component_selector2",
    all_components = [":b", ":c", ":d"],
)

component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "yes")
component(name = "d", info = "yes")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    assertThat(ActionsTestUtil.baseArtifactNames(filesToBuild))
        .containsExactly("a.txt", "b.txt", "c.txt", "d.txt");
  }

  /**
   * Tests that an alias can point to a materializer rule (i.e. a materializer rule can go through
   * an alias).
   */
  @Test
  public void aliasToMaterializerRule_works() throws Exception {

    scratch.file(
        "defs.bzl",
"""
# Component ######################################

ComponentInfo = provider(fields = ["info"])

def _component_impl(ctx):
    return ComponentInfo(info = ctx.attr.info)

component = rule(
    implementation = _component_impl,
    provides = [ComponentInfo],
    attrs = {
        "info": attr.string(),
    }
)

# Component selector #############################

def _component_selector_impl(ctx):
    selected = []
    for c in ctx.attr.all_components:
        if "yes" in c[ComponentInfo].info:
            selected.append(c)
    return MaterializedDepsInfo(deps = selected)

component_selector = materializer_rule(
    implementation = _component_selector_impl,
    allow_real_deps = True,
    attrs = {
        "all_components": attr.label_list(),
    },
)

# Aspect #########################################

AspectInfo = provider(fields = ["info_artifact"])

def _mt_aspect_impl(target, ctx):
    if ctx.rule.kind != "component":
        return []
    print("aspect visiting target: " + str(target.label))
    artifact = ctx.actions.declare_file(target.label.name + ".info")
    ctx.actions.write(artifact, str(target.label))
    return AspectInfo(info_artifact = artifact)

mt_aspect = aspect(
    implementation = _mt_aspect_impl,
)

# Binary #########################################

def _binary_impl(ctx):
    return DefaultInfo(files = depset(direct = [dep[AspectInfo].info_artifact for dep in ctx.attr.deps]))

binary = rule(
    implementation = _binary_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo], aspects = [mt_aspect]),
    },
)
""");

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

# Real deps through single alias ####################################

binary(
    name = "bin",
    deps = [
        ":AAA",
        ":component_selector_alias",
        ":ZZZ",
    ],
)

alias(
    name = "component_selector_alias",
    actual = ":component_selector",
)

# Real deps through alias chain ####################################

binary(
    name = "bin_alias_chain",
    deps = [
        ":AAA",
        ":component_selector_alias_alias",
        ":ZZZ",
    ],
)

alias(
    name = "component_selector_alias_alias",
    actual = ":component_selector_alias",
)

# Materializer rules #############################

component_selector(
    name = "component_selector",
    all_components = [
        ":a",
        ":b",
        ":c",
        ":d",
    ],
)

# capitalized because the "a" component will also match the "aaa" component in the assertions below
component(name = "AAA", info = "yes")
component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "no")
component(name = "d", info = "no")
component(name = "ZZZ", info = "yes")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    assertThat(ActionsTestUtil.baseArtifactNames(filesToBuild))
        .containsExactly("AAA.info", "a.info", "b.info", "ZZZ.info");
    eventCollector.clear();

    cleanSkyframe();
    update("//:bin_alias_chain");
    ConfiguredTarget targetAliasChain = getConfiguredTarget("//:bin_alias_chain");
    NestedSet<Artifact> filesToBuildAliasChain =
        targetAliasChain.getProvider(FileProvider.class).getFilesToBuild();
    assertThat(ActionsTestUtil.baseArtifactNames(filesToBuildAliasChain))
        .containsExactly("AAA.info", "a.info", "b.info", "ZZZ.info");

    // Especially when going through an alias chain, an aspect should still visit the nodes once.
    assertContainsEventWithFrequency("aspect visiting target: @@//:AAA", 1);
    assertContainsEventWithFrequency("aspect visiting target: @@//:ZZZ", 1);
    assertContainsEventWithFrequency("aspect visiting target: @@//:b", 1);
    assertContainsEventWithFrequency("aspect visiting target: @@//:a", 1);
    assertDoesNotContainEvent("aspect visiting target: @@//:c");
    assertDoesNotContainEvent("aspect visiting target: @@//:d");
  }

  /** Tests that a materializer can point to an alias and the final target is materialized. */
  @Test
  public void materializerToAlias_works() throws Exception {

    writeBasicMaterializerRule();

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

binary(
    name = "bin",
    deps = [
        ":aaa",
        ":component_selector",
        ":zzz",
    ],
)

component_selector(
    name = "component_selector",
    all_components = [
        ":a_alias",
        ":b_alias",
        ":c",
        ":d",
        ":d_alias",
    ],
)

alias(name = "a_alias", actual = "a")
alias(name = "b_alias", actual = "b")
alias(name = "d_alias", actual = "d")

component(name = "aaa", info = "yes")
component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "no")
component(name = "d", info = "no")
component(name = "zzz", info = "yes")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    assertThat(ActionsTestUtil.baseArtifactNames(filesToBuild))
        .containsExactly("aaa.txt", "a.txt", "b.txt", "zzz.txt");
  }

  /** Tests alias -> materializer -> alias -> materializer -> alias works with real deps. */
  @Test
  public void aliasToMaterializerToAliasToMaterializerToAlias_works() throws Exception {
    scratch.file(
        "defs.bzl",
"""
# Component ######################################

ComponentInfo = provider(fields = ["output", "info"])

def _component_impl(ctx):
    f = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(f, ctx.label.name)
    return ComponentInfo(output = f, info = ctx.attr.info)

component = rule(
    implementation = _component_impl,
    provides = [ComponentInfo],
    attrs = {
        "info": attr.string(),
    }
)

# Component selector #############################

def _component_selector_impl(ctx):
    return MaterializedDepsInfo(deps = ctx.attr.components)

component_selector = materializer_rule(
    implementation = _component_selector_impl,
    allow_real_deps = True,
    attrs = {
        "components": attr.label_list(),
    },
)

# Binary #########################################

def _binary_impl(ctx):
    files = [dep[ComponentInfo].output for dep in ctx.attr.deps]
    return DefaultInfo(files = depset(direct = files))

binary = rule(
    implementation = _binary_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo]),
    },
)
""");

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

binary(
    name = "bin",
    deps = [
        ":aaa",
        ":alias_to_materializer_to_alias_to_materializer_to_alias",
        ":zzz",
    ],
)

alias(
    name = "alias_to_materializer_to_alias_to_materializer_to_alias",
    actual = ":materializer_to_alias_to_materializer_to_alias",
)

component_selector(
    name = "materializer_to_alias_to_materializer_to_alias",
    components = [":alias_to_materializer_to_alias"],
)

alias(
    name = "alias_to_materializer_to_alias",
    actual = ":materializer_to_alias",
)

component_selector(
    name = "materializer_to_alias",
    components = [":a_alias"],
)

alias(
    name = "a_alias",
    actual = ":a",
)

component(name = "aaa", info = "yes")
component(name = "a", info = "yes")
component(name = "zzz", info = "yes")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    List<String> artifactNames = ActionsTestUtil.baseArtifactNames(filesToBuild);
    assertThat(artifactNames).containsExactly("aaa.txt", "a.txt", "zzz.txt");
  }

  private void writeMaterializerSplitTransitionBzlFile() throws Exception {
    scratch.file(
        "defs.bzl",
"""
# Component selector setting ###########################################

ComponentSelectorProvider = provider(fields = ["selector"])

def _component_selector_setting_impl(ctx):
    return ComponentSelectorProvider(selector = ctx.build_setting_value)

component_selector_setting = rule(
    implementation = _component_selector_setting_impl,
    build_setting = config.string()
)

# Component transition

def _component_selector_setting_transition_impl(settings, attr):
    return [
        {"//:component_selector_setting" : "foo"},
        {"//:component_selector_setting" : "bar"},
    ]

component_transition = transition(
    implementation = _component_selector_setting_transition_impl,
    inputs = [],
    outputs = ["//:component_selector_setting"],
)

# Component ############################################################

ComponentInfo = provider(fields = ["output", "info"])

def _component_impl(ctx):
    f = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(f, ctx.label.name)
    return ComponentInfo(output = f, info = ctx.attr.info)

component = rule(
    implementation = _component_impl,
    provides = [ComponentInfo],
    attrs = {
        "info": attr.string(),
    }
)

# Component selector ###################################################

def _component_selector_impl(ctx):
    selected = []
    for c in ctx.attr.all_components:
        if "yes" in c[ComponentInfo].info:
            selected.append(c)
    return MaterializedDepsInfo(deps = selected)

component_selector = materializer_rule(
    implementation = _component_selector_impl,
    allow_real_deps = True,
    attrs = {
        "all_components": attr.label_list(),
    },
)

# Binary ###############################################################

def _binary_impl(ctx):
    files = [dep[ComponentInfo].output for dep in ctx.attr.deps]
    return DefaultInfo(files = depset(direct = files))

binary = rule(
    implementation = _binary_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo], cfg = component_transition),
    },
)
""");
  }

  /** Tests a materializer rule under a split transition. */
  @Test
  public void materializerRulesUnderSplitTransition_works() throws Exception {

    writeMaterializerSplitTransitionBzlFile();

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary", "component_selector_setting")

component_selector_setting(
    name = "component_selector_setting",
    build_setting_default = "foo",
)

binary(
    name = "bin",
    # has a split transition!
    deps = [":component_selector"],
)

component_selector(
    name = "component_selector",
    all_components = [
        ":a",
        ":b",
        ":c",
        ":d",
    ],
)

component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "no")
component(name = "d", info = "no")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    List<String> artifactNames = ActionsTestUtil.baseArtifactNames(filesToBuild);
    assertThat(artifactNames).containsAtLeast("a.txt", "b.txt");
    assertThat(artifactNames).containsNoneOf("c.txt", "d.txt");
  }

  /**
   * Tests a materializer rule under a split transition with a select() input to the materializer.
   */
  @Test
  public void materializerRulesUnderSplitTransitionAndSelect_works() throws Exception {

    writeMaterializerSplitTransitionBzlFile();

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary", "component_selector_setting")

component_selector_setting(
    name = "component_selector_setting",
    build_setting_default = "foo",
)

config_setting(
    name = "config_setting_foo",
    flag_values = {"//:component_selector_setting": "foo"},
)

config_setting(
    name = "config_setting_bar",
    flag_values = {"//:component_selector_setting": "bar"},
)

binary(
    name = "bin",
    # has a split transition!
    deps = [":component_selector"],
)

component_selector(
    name = "component_selector",
    all_components = select({
        ":config_setting_foo": [
            ":a",
            ":b",
            ":f",
            ":g",
        ],
        ":config_setting_bar": [
            ":c",
            ":d",
            ":e",
            ":f",
            ":g",
        ],
    }),
)

component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "yes")
component(name = "d", info = "yes")
component(name = "e", info = "yes")
component(name = "f", info = "no")
component(name = "g", info = "no")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    NestedSet<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    List<String> artifactNames = ActionsTestUtil.baseArtifactNames(filesToBuild);
    assertThat(artifactNames).containsAtLeast("a.txt", "b.txt", "c.txt", "d.txt", "e.txt");
    assertThat(artifactNames).containsNoneOf("f.txt", "g.txt");
  }

  @Test
  public void materializerRulesPropagatesValidationActions_works() throws Exception {
    scratch.file(
        "defs.bzl",
"""
# Component ######################################

ComponentInfo = provider(fields = ["output", "info"])

def _component_impl(ctx):
    f = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(f, ctx.label.name)

    validation_output = ctx.actions.declare_file(ctx.attr.name + ".validation")
    ctx.actions.run_shell(
        inputs = [f],
        outputs = [validation_output],
        command = "touch $1",
        arguments = [validation_output.path],
    )

    return [
        ComponentInfo(output = f, info = ctx.attr.info),
        OutputGroupInfo(_validation = depset([validation_output])),
    ]

component = rule(
    implementation = _component_impl,
    provides = [ComponentInfo],
    attrs = {
        "info": attr.string(),
    }
)

# Component selector #############################

def _component_selector_impl(ctx):
    selected = []
    for c in ctx.attr.all_components:
        if "yes" in c[ComponentInfo].info:
            selected.append(c)
    return MaterializedDepsInfo(deps = selected)

component_selector = materializer_rule(
    implementation = _component_selector_impl,
    allow_real_deps = True,
    attrs = {
        "all_components": attr.label_list(),
    },
)

# Binary #########################################

def _binary_impl(ctx):
    files = [dep[ComponentInfo].output for dep in ctx.attr.deps]
    return DefaultInfo(files = depset(direct = files))

binary = rule(
    implementation = _binary_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo]),
    },
)
""");

    scratch.file(
        "BUILD",
"""
load(":defs.bzl", "component", "component_selector", "binary")

binary(
    name = "bin",
    deps = [
        ":aaa",
        ":component_selector",
        ":zzz",
    ],
)

component_selector(
    name = "component_selector",
    all_components = [
        ":a",
        ":b",
        ":c",
        ":d",
    ],
)

component(name = "aaa", info = "yes")
component(name = "a", info = "yes")
component(name = "b", info = "yes")
component(name = "c", info = "no")
component(name = "d", info = "no")
component(name = "zzz", info = "yes")
""");

    update("//:bin");
    ConfiguredTarget target = getConfiguredTarget("//:bin");
    OutputGroupInfo outputGroupInfo = target.get(OutputGroupInfo.STARLARK_CONSTRUCTOR);
    NestedSet<Artifact> validationOutputs =
        outputGroupInfo.getOutputGroup(OutputGroupInfo.VALIDATION);
    List<String> artifactNames = ActionsTestUtil.baseArtifactNames(validationOutputs);
    assertThat(artifactNames)
        .containsExactly("aaa.validation", "a.validation", "b.validation", "zzz.validation");
  }

  private void writeVisibilityDefsBzlFile() throws Exception {
    scratch.file(
        "defs.bzl",
"""
# Component ######################################

ComponentInfo = provider(fields = ["info"])

def _component_impl(ctx):
    return ComponentInfo(info = ctx.attr.info)

component = rule(
    implementation = _component_impl,
    provides = [ComponentInfo],
    attrs = {
        "info": attr.string(),
    }
)

# Component selector #############################

def _component_selector_impl(ctx):
    selected = []
    for c in ctx.attr.all_components:
        if "yes" in c[ComponentInfo].info:
            selected.append(c)
    return MaterializedDepsInfo(deps = selected)

component_selector = materializer_rule(
    implementation = _component_selector_impl,
    allow_real_deps = True,
    attrs = {
        "all_components": attr.label_list(),
    },
)

# Binary #########################################

def _binary_impl(ctx):
    return DefaultInfo()

binary = rule(
    implementation = _binary_impl,
    attrs = {
        "deps": attr.label_list(providers = [ComponentInfo]),
    },
)
""");
  }

  @Test
  public void materializerRuleVisibilityViolation_throwsError() throws Exception {
    writeVisibilityDefsBzlFile();
    scratch.file("BUILD", "");

    scratch.file(
        "binary1/BUILD",
"""
load("//:defs.bzl", "binary")

binary(
    name = "bin1",
    deps = [
        "//components:component_selector",
    ],
)
""");

    scratch.file(
        "binary2/BUILD",
"""
load("//:defs.bzl", "binary")

binary(
    name = "bin2",
    deps = [
        "//components:component_selector",
    ],
)
""");

    scratch.file(
        "components/BUILD",
"""
load("//:defs.bzl", "component", "component_selector", "binary")

component_selector(
    name = "component_selector",
    all_components = [
        ":a",
        ":b",
        ":c",
        ":d",
    ],
    visibility = ["//binary1:__pkg__"],
)

component(name = "a", info = "yes", visibility = ["//:__subpackages__"])
component(name = "b", info = "yes", visibility = ["//:__subpackages__"])
component(name = "c", info = "no", visibility = ["//:__subpackages__"])
component(name = "d", info = "no", visibility = ["//:__subpackages__"])
""");

    // The materializer target is visible to bin1.
    update("//binary1:bin1");

    // The materializer target is not visible to bin2.
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//binary2:bin2"));
    assertContainsEvent(
"""
ERROR /workspace/binary2/BUILD:3:7: in binary rule //binary2:bin2: Visibility error:
target '//components:component_selector' is not visible from
target '//binary2:bin2'
""");
  }

  @Test
  public void materializerRuleMaterializedTargetVisibilityViolation_throwsError() throws Exception {
    scratch.file("BUILD", "");

    writeVisibilityDefsBzlFile();

    scratch.file(
        "binary/BUILD",
"""
load("//:defs.bzl", "binary")

binary(
    name = "bin",
    deps = [
        "//components:component_selector",
    ],
)
""");

    scratch.file(
        "components/BUILD",
"""
load("//:defs.bzl", "component", "component_selector", "binary")

component_selector(
    name = "component_selector",
    all_components = [
        ":a",
        ":b",
        ":c",
        ":d",
    ],
    visibility = ["//binary:__pkg__"],
)

component(name = "a", info = "yes", visibility = ["//visibility:private"])
component(name = "b", info = "yes", visibility = ["//visibility:private"])
component(name = "c", info = "no", visibility = ["//visibility:private"])
component(name = "d", info = "no", visibility = ["//visibility:private"])
""");

    // The materializer target is visible to bin, but the materialized targets are not.
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//binary:bin"));
    assertContainsEvent(
"""
ERROR /workspace/binary/BUILD:3:7: in binary rule //binary:bin: Visibility error:
target '//components:a' is not visible from
target '//binary:bin'
""");
    assertContainsEvent(
"""
ERROR /workspace/binary/BUILD:3:7: in binary rule //binary:bin: Visibility error:
target '//components:b' is not visible from
target '//binary:bin'
""");
  }
}
