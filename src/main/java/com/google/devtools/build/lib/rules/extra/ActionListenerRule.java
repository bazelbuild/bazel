// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.extra;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Types.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;

/**
 * Rule definition for action_listener rule.
 */
public final class ActionListenerRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /*<!-- #BLAZE_RULE(action_listener).ATTRIBUTE(mnemonics) -->
        A list of action mnemonics this <code>action_listener</code> should listen
        for, e.g. <code>[ "Javac" ]</code>.
        <p>
          Mnemonics are not a public interface.
          There's no guarantee that the mnemonics and their actions don't change.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("mnemonics", STRING_LIST).mandatory())
        /*<!-- #BLAZE_RULE(action_listener).ATTRIBUTE(extra_actions) -->
        A list of <code><a href="${link extra_action}">extra_action</a></code> targets
        this <code>action_listener</code> should add to the build graph.
        E.g. <code>[ "//my/tools:analyzer" ]</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("extra_actions", LABEL_LIST).mandatory()
            .allowedRuleClasses("extra_action")
            .allowedFileTypes())
        .removeAttribute("deps")
        .removeAttribute("data")
        .removeAttribute(":action_listener")
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("action_listener")
        .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
        .factoryClass(ActionListener.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = action_listener, FAMILY = Extra Actions)[GENERIC_RULE] -->

<p>
  <b>WARNING:</b> Extra actions are deprecated. Use
  <a href="https://bazel.build/rules/aspects">aspects</a>
  instead.
</p>

<p>
  An <code>action_listener</code> rule doesn't produce any output itself.
  Instead, it allows tool developers to insert
  <a href="${link extra_action}"><code>extra_action</code></a>s into the build system,
  by providing a mapping from action to <a href="${link extra_action}"><code>extra_action</code>
  </a>.
</p>

<p>
  This rule's arguments map action mnemonics to
  <a href="${link extra_action}"><code>extra_action</code></a> rules.
</p>

<p>
  By specifying the option <a href="${link user-manual#flag--experimental_action_listener}">
  <code>--experimental_action_listener=&lt;label&gt;</code></a>,
  the build will use the specified <code>action_listener</code> to insert
  <a href="${link extra_action}"><code>extra_action</code></a>s into the build graph.
</p>

<h4 id="action_listener_example">Example</h4>
<pre>
action_listener(
    name = "index_all_languages",
    mnemonics = [
        "Javac",
        "CppCompile",
        "Python",
    ],
    extra_actions = [":indexer"],
)

action_listener(
    name = "index_java",
    mnemonics = ["Javac"],
    extra_actions = [":indexer"],
)

extra_action(
    name = "indexer",
    tools = ["//my/tools:indexer"],
    cmd = "$(location //my/tools:indexer)" +
          "--extra_action_file=$(EXTRA_ACTION_FILE)",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
