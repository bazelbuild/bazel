// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableListMultimap;
import com.google.devtools.build.lib.bazel.bzlmod.Module.WhichRepoMappings;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.BuildType.LabelConversionContext;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashMap;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Structure;

/** A Starlark object representing a Bazel module in the external dependency graph. */
@StarlarkBuiltin(
    name = "bazel_module",
    doc = "Represents a Bazel module in the external dependency graph.")
public class StarlarkBazelModule implements StarlarkValue {
  private final String name;
  private final String version;
  private final Tags tags;

  @StarlarkBuiltin(name = "bazel_module_tags", doc = "TODO")
  static class Tags implements Structure {
    private final ModuleExtension extension;
    private final ImmutableListMultimap<String, TypeCheckedTag> typeCheckedTags;

    private Tags(
        ModuleExtension extension, ImmutableListMultimap<String, TypeCheckedTag> typeCheckedTags) {
      this.extension = extension;
      this.typeCheckedTags = typeCheckedTags;
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Nullable
    @Override
    public Object getValue(String name) throws EvalException {
      if (extension.getTagClasses().containsKey(name)) {
        return typeCheckedTags.get(name);
      }
      return null;
    }

    @Override
    public ImmutableCollection<String> getFieldNames() {
      return extension.getTagClasses().keySet();
    }

    @Nullable
    @Override
    public String getErrorMessageForUnknownField(String field) {
      return "unknown tag class " + field;
    }
  }

  private StarlarkBazelModule(String name, String version, Tags tags) {
    this.name = name;
    this.version = version;
    this.tags = tags;
  }

  /**
   * Creates a new {@link StarlarkBazelModule} object representing the given {@link Module}, with
   * its scope limited to the given {@link ModuleExtension}. It'll be populated with the tags
   * present in the given {@link ModuleExtensionUsage}.
   */
  public static StarlarkBazelModule create(
      Module module, ModuleExtension extension, ModuleExtensionUsage usage)
      throws ExternalDepsException {
    Label moduleRootLabel =
        Label.createUnvalidated(
            PackageIdentifier.create(
                RepositoryName.createFromValidStrippedName(module.getKey().getCanonicalRepoName()),
                PathFragment.EMPTY_FRAGMENT),
            "unused_dummy_target_name");
    LabelConversionContext labelConversionContext =
        new LabelConversionContext(
            moduleRootLabel,
            module.getRepoMapping(WhichRepoMappings.BAZEL_DEPS_ONLY),
            /* convertedLabelsInPackage= */ new HashMap<>());
    ImmutableListMultimap.Builder<String, TypeCheckedTag> typeCheckedTags =
        ImmutableListMultimap.builder();
    for (Tag tag : usage.getTags()) {
      TagClass tagClass = extension.getTagClasses().get(tag.getTagName());
      if (tagClass == null) {
        throw ExternalDepsException.withMessage(
            Code.BAD_MODULE,
            "The module extension defined at %s does not have a tag class named %s, but its use is"
                + " attempted at %s",
            extension.getLocation(),
            tag.getTagName(),
            tag.getLocation());
      }

      // Now we need to type-check the attribute values and convert them into "build language types"
      // (for example, String to Label).
      typeCheckedTags.put(
          tag.getTagName(), TypeCheckedTag.create(tagClass, tag, labelConversionContext));
    }
    return new StarlarkBazelModule(
        module.getName(),
        module.getVersion().getOriginal(),
        new Tags(extension, typeCheckedTags.build()));
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @StarlarkMethod(name = "name", structField = true, doc = "The name of the module.")
  public String getName() {
    return name;
  }

  @StarlarkMethod(name = "version", structField = true, doc = "The version of the module.")
  public String getVersion() {
    return version;
  }

  @StarlarkMethod(
      name = "tags",
      structField = true,
      doc = "The tags in the module related to the module extension currently being processed.")
  public Tags getTags() {
    return tags;
  }
}
