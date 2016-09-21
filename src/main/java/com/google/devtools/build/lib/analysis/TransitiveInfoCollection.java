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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.SkylarkIndexable;
import javax.annotation.Nullable;

/**
 * Multiple {@link TransitiveInfoProvider}s bundled together.
 *
 * <p>Represents the information made available by a {@link ConfiguredTarget} to other ones that
 * depend on it. For more information about the analysis phase, see {@link
 * com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory}.
 *
 * <p>Implementations of build rules should <b>not</b> hold on to references to the {@link
 * TransitiveInfoCollection}s representing their direct prerequisites in order to reduce their
 * memory footprint (otherwise, the referenced object could refer one of its direct dependencies in
 * turn, thereby making the size of the objects reachable from a single instance unbounded).
 *
 * @see com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory
 * @see TransitiveInfoProvider
 */
@SkylarkModule(
  name = "Target",
  category = SkylarkModuleCategory.BUILTIN,
  doc =
      "A BUILD target. It is essentially a <code>struct</code> with the following fields:"
    + "<ul>"
    + "<li><h3 id=\"modules.Target.label\">label</h3><code><a class=\"anchor\" "
    + "href=\"Label.html\">Label</a> Target.label</code><br>The identifier of the target.</li>"
    + "<li><h3 id=\"modules.Target.files\">files</h3><code><a class=\"anchor\" "
    + "href=\"set.html\">set</a> Target.files </code><br>The (transitive) set of <a "
    + "class=\"anchor\" href=\"File.html\">File</a>s produced by this target.</li>"
    + "<li><h3 id=\"modules.Target.extraproviders\">Extra providers</h3>For rule targets all "
    + "additional providers provided by this target are accessible as <code>struct</code> fields. "
    + "These extra providers are defined in the <code>struct</code> returned by the rule "
    + "implementation function.</li>"
    + "</ul>")
public interface TransitiveInfoCollection extends SkylarkIndexable {

  /**
   * Returns the transitive information provider requested, or null if the provider is not found.
   * The provider has to be a TransitiveInfoProvider Java class.
   */
  @Nullable <P extends TransitiveInfoProvider> P getProvider(Class<P> provider);

  /**
   * Returns the label associated with this prerequisite.
   */
  Label getLabel();

  /**
   * <p>Returns the {@link BuildConfiguration} for which this transitive info collection is defined.
   * Configuration is defined for all configured targets with exception of {@link
   * InputFileConfiguredTarget} and {@link PackageGroupConfiguredTarget} for which it is always
   * <b>null</b>.</p>
   */
  @Nullable BuildConfiguration getConfiguration();

  /**
   * Returns the transitive information requested or null, if the information is not found.
   * The transitive information has to have been added using the Skylark framework.
   */
  @Nullable Object get(String providerKey);
}
