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
package com.google.devtools.build.lib.rules.python;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConvenienceSymlinks.SymlinkDefinition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Set;
import java.util.function.Function;

/** Rule definitions for Python rules. */
public class PyRuleClasses {

  public static final FileType PYTHON_SOURCE = FileType.of(".py", ".py3");

  /**
   * A value set of the target and sentinel values that doesn't mention the sentinel in error
   * messages.
   */
  public static final AllowedValueSet TARGET_PYTHON_ATTR_VALUE_SET =
      new AllowedValueSet(PythonVersion.TARGET_AND_SENTINEL_STRINGS) {
        @Override
        public String getErrorReason(Object value) {
          return String.format("has to be one of 'PY2' or 'PY3' instead of '%s'", value);
        }
      };

  /**
   * Returns a rule transition factory for Python binary rules and other rules that may change the
   * Python version.
   *
   * <p>The factory reads the version specified by the target's {@code python_version} attribute if
   * given, falling back on the {@code default_python_version} attribute otherwise. Both attributes
   * must exist on the rule class. If a value was read successfully, the factory returns a
   * transition that sets the version to that value. Otherwise if neither attribute was set, the
   * factory returns {@code defaultTransition} instead.
   *
   * <p>If either attribute has an unparsable value on the target, then the factory returns {@code
   * defaultTransition} and it is up to the rule's analysis phase ({@link
   * PyCommon#validateTargetPythonVersionAttr}) to report an attribute error to the user. This case
   * should be prevented by attribute validation if the rule class is defined correctly.
   */
  public static TransitionFactory<Rule> makeVersionTransition(
      PythonVersionTransition defaultTransition) {
    return (rule) -> {
      AttributeMap attrs = RawAttributeMapper.of(rule);
      // Fail fast if we're used on an ill-defined rule class.
      Preconditions.checkArgument(
          attrs.has(PyCommon.PYTHON_VERSION_ATTRIBUTE, Type.STRING)
              && attrs.has(PyCommon.DEFAULT_PYTHON_VERSION_ATTRIBUTE, Type.STRING),
          "python version transitions require that the RuleClass define both "
              + "'default_python_version' and 'python_version'");
      // Attribute validation should enforce that the attribute string value is either a target
      // value ("PY2" or "PY3") or the sentinel value ("_INTERNAL_SENTINEL"). But just in case,
      // we'll, treat an invalid value as the default value rather than propagate an unchecked
      // exception in this context. That way the user can at least get a clean error message
      // instead of a crash.
      PythonVersionTransition transition;
      try {
        PythonVersion versionFromAttributes = PyCommon.readPythonVersionFromAttributes(attrs);
        if (versionFromAttributes == null) {
          transition = defaultTransition;
        } else {
          transition = PythonVersionTransition.toConstant(versionFromAttributes);
        }
      } catch (IllegalArgumentException ex) {
        transition = defaultTransition;
      }
      return transition;
    };
  }

  /**
   * A Python version transition that sets the version as specified by the target's attributes, with
   * a default determined by {@link PythonOptions#getDefaultPythonVersion}.
   */
  public static final TransitionFactory<Rule> VERSION_TRANSITION =
      makeVersionTransition(PythonVersionTransition.toDefault());

  /** The py2 and py3 symlinks. */
  public enum PySymlink implements SymlinkDefinition {
    PY2(PythonVersion.PY2),
    PY3(PythonVersion.PY3);

    private final String versionString;
    private final PythonVersionTransition transition;

    private PySymlink(PythonVersion version) {
      this.versionString = Ascii.toLowerCase(version.toString());
      this.transition = PythonVersionTransition.toConstant(version);
    }

    @Override
    public String getLinkName(String symlinkPrefix, String productName, String workspaceBaseName) {
      return symlinkPrefix + versionString;
    }

    @Override
    public Set<Path> getLinkPaths(
        BuildRequestOptions buildRequestOptions,
        Set<BuildConfiguration> targetConfigs,
        Function<BuildOptions, BuildConfiguration> configGetter,
        RepositoryName repositoryName,
        Path outputPath,
        Path execRoot) {
      if (!buildRequestOptions.experimentalCreatePySymlinks) {
        return ImmutableSet.of();
      }
      return targetConfigs.stream()
          .map(config -> configGetter.apply(transition.patch(config.getOptions())))
          .map(config -> config.getOutputDirectory(repositoryName).getRoot().asPath())
          .distinct()
          .collect(toImmutableSet());
    }
  }
}
