// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.starlark;

import static com.google.devtools.build.lib.packages.Types.STRING_DICT;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkAttrModule.Descriptor;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtension;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionEvalStarlarkThreadContext;
import com.google.devtools.build.lib.bazel.bzlmod.TagClass;
import com.google.devtools.build.lib.bazel.repository.RepoRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeValueSource;
import com.google.devtools.build.lib.packages.BzlInitThreadContext;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryModuleApi;
import com.google.protobuf.ByteString;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.Location;

/**
 * The Starlark module containing the definition of {@code repository_rule} function to define a
 * Starlark remote repository.
 */
public class StarlarkRepositoryModule implements RepositoryModuleApi {

  @Override
  public StarlarkCallable repositoryRule(
      StarlarkCallable implementation,
      Object attrs,
      Boolean local,
      Sequence<?> environ, // <String> expected
      Boolean configure,
      Boolean remotable,
      Object doc, // <String> or Starlark.NONE
      StarlarkThread thread)
      throws EvalException {
    RepoRule.Builder builder =
        RepoRule.builder()
            .impl(implementation)
            .local(local)
            .configure(configure)
            .remotable(remotable)
            .environ(ImmutableSet.copyOf(Sequence.cast(environ, String.class, "repository_rule")))
            .doc(Starlark.toJavaOptional(doc, String.class).map(Starlark::trimDocString));
    if (thread.getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_REPO_REMOTE_EXEC)) {
      builder.addAttribute(
          Attribute.attr("exec_properties", STRING_DICT).defaultValue(ImmutableMap.of()).build());
    }
    if (attrs != Starlark.NONE) {
      for (Map.Entry<String, Descriptor> attr :
          Dict.cast(attrs, String.class, Descriptor.class, "attrs").entrySet()) {
        Descriptor attrDescriptor = attr.getValue();
        AttributeValueSource source = attrDescriptor.getValueSource();
        String attrName = source.convertToNativeName(attr.getKey());
        if (builder.hasAttribute(attrName)) {
          throw Starlark.errorf(
              "There is already a built-in attribute '%s' which cannot be overridden", attrName);
        }
        builder.addAttribute(attrDescriptor.build(attrName));
      }
    }
    BzlInitThreadContext bzlInitContext =
        BzlInitThreadContext.fromOrFail(thread, "repository_rule");
    builder.idBuilder().bzlFileLabel(bzlInitContext.getBzlFile());
    builder.transitiveBzlDigest(ByteString.copyFrom(bzlInitContext.getTransitiveDigest()));
    Label.RepoMappingRecorder repoMappingRecorder =
        thread.getThreadLocal(Label.RepoMappingRecorder.class);
    if (repoMappingRecorder != null) {
      builder.recordedRepoMappingEntries(repoMappingRecorder.recordedEntries());
    }
    return new StarlarkRepoRule(builder);
  }

  /**
   * The value returned by calling the {@code repository_rule} function in Starlark. It itself is a
   * callable value; calling it defines a repo.
   */
  @StarlarkBuiltin(
      name = "repository_rule",
      category = DocCategory.BUILTIN,
      doc =
"""
A callable value that may be invoked within the implementation function of a module extension to \
instantiate and return a repository rule. Created by \
<a href="../globals/bzl.html#repository_rule"><code>repository_rule()</code></a>.
""")
  public static final class StarlarkRepoRule
      implements StarlarkCallable, StarlarkExportable, RepoRule.Supplier {
    private final RepoRule.Builder builder;
    // Populated on first use after export to avoid recreating the repo rule on each usage.
    @Nullable private volatile RepoRule repoRule;

    private StarlarkRepoRule(RepoRule.Builder builder) {
      this.builder = builder;
    }

    @Override
    public String getName() {
      return "repository_rule";
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Override
    public void export(
        EventHandler handler,
        Label extensionLabel,
        String exportedName,
        Location exportedLocation) {
      builder.idBuilder().ruleName(exportedName);
    }

    @Override
    public boolean isExported() {
      return builder.idBuilder().isRuleNameSet();
    }

    @Override
    public void repr(Printer printer) {
      if (!isExported()) {
        printer.append("<anonymous starlark repository rule>");
      } else {
        printer.append("<starlark repository rule " + builder.idBuilder().build() + ">");
      }
    }

    @Override
    public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
        throws EvalException, InterruptedException {
      if (!args.isEmpty()) {
        throw new EvalException("unexpected positional arguments");
      }
      ModuleExtensionEvalStarlarkThreadContext extensionEvalContext =
          ModuleExtensionEvalStarlarkThreadContext.fromOrNull(thread);
      if (extensionEvalContext == null) {
        throw new EvalException(
            "repo rules can only be called from within module extension impl functions");
      }
      if (!isExported()) {
        throw new EvalException("attempting to instantiate a non-exported repository rule");
      }
      extensionEvalContext.lazilyCreateRepo(thread, kwargs, getRepoRule());
      return Starlark.NONE;
    }

    @Override
    public RepoRule getRepoRule() {
      if (repoRule != null) {
        return repoRule;
      }
      synchronized (this) {
        if (repoRule != null) {
          return repoRule;
        }
        repoRule = builder.build();
        return repoRule;
      }
    }
  }

  @Override
  public void failWithIncompatibleUseCcConfigureFromRulesCc(StarlarkThread thread)
      throws EvalException {
    if (thread
        .getSemantics()
        .getBool(BuildLanguageOptions.INCOMPATIBLE_USE_CC_CONFIGURE_FROM_RULES_CC)) {
      throw Starlark.errorf(
          "Incompatible flag "
              + "--incompatible_use_cc_configure_from_rules_cc has been flipped. Please use "
              + "cc_configure and related logic from https://github.com/bazelbuild/rules_cc. "
              + "See https://github.com/bazelbuild/bazel/issues/10134 for details and migration "
              + "instructions.");
    }
  }

  @Override
  public Object moduleExtension(
      StarlarkCallable implementation,
      Dict<?, ?> tagClasses, // Dict<String, TagClass>
      Object doc, // <String> or Starlark.NONE
      Sequence<?> environ, // <String>
      boolean osDependent,
      boolean archDependent,
      StarlarkThread thread)
      throws EvalException {
    return ModuleExtension.builder()
        .setImplementation(implementation)
        .setTagClasses(
            ImmutableMap.copyOf(Dict.cast(tagClasses, String.class, TagClass.class, "tag_classes")))
        .setDoc(Starlark.toJavaOptional(doc, String.class).map(Starlark::trimDocString))
        .setDefiningBzlFileLabel(
            BzlInitThreadContext.fromOrFail(thread, "module_extension()").getBzlFile())
        .setEnvVariables(ImmutableList.copyOf(Sequence.cast(environ, String.class, "environ")))
        .setLocation(thread.getCallerLocation())
        .setOsDependent(osDependent)
        .setArchDependent(archDependent)
        .build();
  }

  @Override
  public TagClass tagClass(
      Dict<?, ?> attrs, // Dict<String, StarlarkAttrModule.Descriptor>
      Object doc // <String> or Starlark.NONE
      ) throws EvalException {
    ImmutableList.Builder<Attribute> attrBuilder = ImmutableList.builder();
    for (Map.Entry<String, Descriptor> attr :
        Dict.cast(attrs, String.class, Descriptor.class, "attrs").entrySet()) {
      Descriptor attrDescriptor = attr.getValue();
      AttributeValueSource source = attrDescriptor.getValueSource();
      String attrName = source.convertToNativeName(attr.getKey());
      attrBuilder.add(attrDescriptor.build(attrName));
      // TODO(wyv): validate attributes. No selects, no latebound defaults, or any crazy stuff like
      //   that.
    }
    return TagClass.create(
        attrBuilder.build(),
        Starlark.toJavaOptional(doc, String.class).map(Starlark::trimDocString));
  }
}
