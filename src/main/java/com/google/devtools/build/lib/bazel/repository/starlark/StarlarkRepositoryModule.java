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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.DocumentMethods;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.starlark.StarlarkAttrModule.Descriptor;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleCreator;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtension;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionEvalStarlarkThreadContext;
import com.google.devtools.build.lib.bazel.bzlmod.TagClass;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeValueSource;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleFactory;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import com.google.devtools.build.lib.packages.WorkspaceFactoryHelper;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryModuleApi;
import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkThread.CallStackEntry;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.Location;

/**
 * The Starlark module containing the definition of {@code repository_rule} function to define a
 * Starlark remote repository.
 */
@DocumentMethods
public class StarlarkRepositoryModule implements RepositoryModuleApi {

  @Override
  public StarlarkCallable repositoryRule(
      StarlarkCallable implementation,
      Object attrs,
      Boolean local,
      Sequence<?> environ, // <String> expected
      Boolean configure,
      Boolean remotable,
      String doc,
      StarlarkThread thread)
      throws EvalException {
    BazelStarlarkContext context = BazelStarlarkContext.from(thread);
    context.checkLoadingOrWorkspacePhase("repository_rule");
    // We'll set the name later, pass the empty string for now.
    RuleClass.Builder builder = new RuleClass.Builder("", RuleClassType.WORKSPACE, true);

    ImmutableList<StarlarkThread.CallStackEntry> callstack = thread.getCallStack();
    builder.setCallStack(
        callstack.subList(0, callstack.size() - 1)); // pop 'repository_rule' itself

    builder.addAttribute(attr("$local", BOOLEAN).defaultValue(local).build());
    builder.addAttribute(attr("$configure", BOOLEAN).defaultValue(configure).build());
    if (thread.getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_REPO_REMOTE_EXEC)) {
      builder.addAttribute(attr("$remotable", BOOLEAN).defaultValue(remotable).build());
      BaseRuleClasses.execPropertiesAttribute(builder);
    }
    builder.addAttribute(attr("$environ", STRING_LIST).defaultValue(environ).build());
    BaseRuleClasses.commonCoreAndStarlarkAttributes(builder);
    builder.add(attr("expect_failure", STRING));
    if (attrs != Starlark.NONE) {
      for (Map.Entry<String, Descriptor> attr :
          Dict.cast(attrs, String.class, Descriptor.class, "attrs").entrySet()) {
        Descriptor attrDescriptor = attr.getValue();
        AttributeValueSource source = attrDescriptor.getValueSource();
        String attrName = source.convertToNativeName(attr.getKey());
        if (builder.contains(attrName)) {
          throw Starlark.errorf(
              "There is already a built-in attribute '%s' which cannot be overridden", attrName);
        }
        builder.addAttribute(attrDescriptor.build(attrName));
      }
    }
    builder.setConfiguredTargetFunction(implementation);
    BazelModuleContext bzlModule =
        BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(thread));
    builder.setRuleDefinitionEnvironmentLabelAndDigest(
        bzlModule.label(), bzlModule.bzlTransitiveDigest());
    builder.setWorkspaceOnly();
    return new RepositoryRuleFunction(builder, implementation);
  }

  // RepositoryRuleFunction is the result of repository_rule(...).
  // It is a callable value; calling it yields a Rule instance.
  @StarlarkBuiltin(
      name = "repository_rule",
      category = DocCategory.BUILTIN,
      doc =
          "A callable value that may be invoked during evaluation of the WORKSPACE file or within"
              + " the implementation function of a module extension to instantiate and return a"
              + " repository rule.")
  private static final class RepositoryRuleFunction
      implements StarlarkCallable, StarlarkExportable, BzlmodRepoRuleCreator {
    private final RuleClass.Builder builder;
    private final StarlarkCallable implementation;
    private Label extensionLabel;
    private String exportedName;

    private RepositoryRuleFunction(RuleClass.Builder builder, StarlarkCallable implementation) {
      this.builder = builder;
      this.implementation = implementation;
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
    public void export(EventHandler handler, Label extensionLabel, String exportedName) {
      this.extensionLabel = extensionLabel;
      this.exportedName = exportedName;
    }

    @Override
    public boolean isExported() {
      return extensionLabel != null;
    }

    @Override
    public void repr(Printer printer) {
      if (exportedName == null) {
        printer.append("<anonymous starlark repository rule>");
      } else {
        printer.append("<starlark repository rule " + extensionLabel + "%" + exportedName + ">");
      }
    }

    @Override
    public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
        throws EvalException, InterruptedException {
      if (!args.isEmpty()) {
        throw new EvalException("unexpected positional arguments");
      }
      // Decide whether we're operating in the new mode (during module extension evaluation) or in
      // legacy mode (during workspace evaluation).
      ModuleExtensionEvalStarlarkThreadContext extensionEvalContext =
          ModuleExtensionEvalStarlarkThreadContext.from(thread);
      if (extensionEvalContext == null) {
        return createRuleLegacy(thread, kwargs);
      }
      if (!isExported()) {
        throw new EvalException("attempting to instantiate a non-exported repository rule");
      }
      Object nameValue = kwargs.getOrDefault("name", Starlark.NONE);
      if (!(nameValue instanceof String)) {
        throw Starlark.errorf(
            "expected string for attribute 'name', got '%s'", Starlark.type(nameValue));
      }
      String name = (String) nameValue;
      String prefixedName = extensionEvalContext.getRepoPrefix() + name;
      extensionEvalContext.createRepo(
          name,
          this,
          Maps.transformEntries(kwargs, (k, v) -> k.equals("name") ? prefixedName : v),
          thread.getSemantics(),
          thread.getCallerLocation());
      return Starlark.NONE;
    }

    private Object createRuleLegacy(StarlarkThread thread, Dict<String, Object> kwargs)
        throws EvalException, InterruptedException {
      BazelStarlarkContext.from(thread).checkWorkspacePhase("repository rule " + exportedName);
      String ruleClassName;
      // If the function ever got exported (the common case), we take the name
      // it was exported to. Only in the not intended case of calling an unexported
      // repository function through an exported macro, we fall back, for lack of
      // alternatives, to the name in the local context.
      // TODO(b/111199163): we probably should disallow the use of non-exported
      // repository rules anyway.
      if (isExported()) {
        ruleClassName = exportedName;
      } else {
        // repository_rules should be subject to the same "exported" requirement
        // as package rules, but sadly we forgot to add the necessary check and
        // now many projects create and instantiate repository_rules without an
        // intervening export; see b/111199163. An incompatible flag is required.
        if (false) {
          throw new EvalException("attempt to instantiate a non-exported repository rule");
        }

        // The historical workaround was a fragile hack to introspect on the call
        // expression syntax, f() or x.f(), to find the name f, but we no longer
        // have access to the call expression, so now we just create an ugly
        // name from the function. See github.com/bazelbuild/bazel/issues/10441
        ruleClassName = "unexported_" + implementation.getName();
      }
      try {
        RuleClass ruleClass = builder.build(ruleClassName, ruleClassName);
        PackageContext context = PackageFactory.getContext(thread);
        Package.Builder packageBuilder = context.getBuilder();

        // TODO(adonovan): is this cast safe? Check.
        String name = (String) kwargs.get("name");
        WorkspaceFactoryHelper.addMainRepoEntry(packageBuilder, name, thread.getSemantics());
        WorkspaceFactoryHelper.addRepoMappings(packageBuilder, kwargs, name);
        Rule rule =
            WorkspaceFactoryHelper.createAndAddRepositoryRule(
                context.getBuilder(),
                ruleClass,
                /*bindRuleClass=*/ null,
                WorkspaceFactoryHelper.getFinalKwargs(kwargs),
                thread.getSemantics(),
                thread.getCallStack());
        return rule;
      } catch (InvalidRuleException | NameConflictException | LabelSyntaxException e) {
        throw Starlark.errorf("%s", e.getMessage());
      }
    }

    @Override
    public Rule createAndAddRule(
        Package.Builder packageBuilder,
        StarlarkSemantics semantics,
        Map<String, Object> kwargs,
        EventHandler handler)
        throws InterruptedException, InvalidRuleException, NameConflictException {
      RuleClass ruleClass = builder.build(exportedName, exportedName);
      BuildLangTypedAttributeValuesMap attributeValues =
          new BuildLangTypedAttributeValuesMap(kwargs);
      ImmutableList.Builder<CallStackEntry> callStack = ImmutableList.builder();
      // TODO(pcloudy): Optimize the callstack
      callStack.add(new CallStackEntry("RepositoryRuleFunction.createRule", Location.BUILTIN));
      return RuleFactory.createAndAddRule(
          packageBuilder, ruleClass, attributeValues, handler, semantics, callStack.build());
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

  @StarlarkMethod(
      name = "module_extension",
      doc =
          "Creates a new module extension. Store it in a global value, so that it can be exported"
              + " and used in a MODULE.bazel file.",
      parameters = {
        @Param(
            name = "implementation",
            named = true,
            doc =
                "The function that implements this module extension. Must take a single parameter,"
                    + " <code><a href=\"module_ctx.html\">module_ctx</a></code>. The function is"
                    + " called once at the beginning of a build to determine the set of available"
                    + " repos."),
        @Param(
            name = "tag_classes",
            defaultValue = "{}",
            doc =
                "A dictionary to declare all the tag classes used by the extension. It maps from"
                    + " the name of the tag class to a <code><a"
                    + " href=\"tag_class.html\">tag_class</a></code> object.",
            named = true,
            positional = false),
        @Param(
            name = "doc",
            defaultValue = "''",
            doc =
                "A description of the module extension that can be extracted by documentation"
                    + " generating tools.",
            named = true,
            positional = false)
      },
      useStarlarkThread = true)
  public Object moduleExtension(
      StarlarkCallable implementation,
      Dict<?, ?> tagClasses, // Dict<String, TagClass>
      String doc,
      StarlarkThread thread)
      throws EvalException {
    ModuleExtension.InStarlark inStarlark = new ModuleExtension.InStarlark();
    inStarlark
        .getBuilder()
        .setImplementation(implementation)
        .setTagClasses(
            ImmutableMap.copyOf(Dict.cast(tagClasses, String.class, TagClass.class, "tag_classes")))
        .setDoc(doc)
        .setDefinitionEnvironmentLabel(
            BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(thread)).label())
        .setLocation(thread.getCallerLocation());
    return inStarlark;
  }

  @StarlarkMethod(
      name = "tag_class",
      doc =
          "Creates a new tag_class object, which defines an attribute schema for a class of tags,"
              + " which are data objects usable by a module extension.",
      parameters = {
        @Param(
            name = "attrs",
            defaultValue = "{}",
            named = true,
            doc =
                "A dictionary to declare all the attributes of this tag class. It maps from an"
                    + " attribute name to an attribute object (see <a href=\"attr.html\">attr</a>"
                    + " module)."),
        @Param(
            name = "doc",
            defaultValue = "''",
            doc =
                "A description of the tag class that can be extracted by documentation"
                    + " generating tools.",
            named = true,
            positional = false)
      },
      useStarlarkThread = true)
  public TagClass tagClass(
      Dict<?, ?> attrs, // Dict<String, StarlarkAttrModule.Descriptor>
      String doc,
      StarlarkThread thread)
      throws EvalException {
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
    return TagClass.create(attrBuilder.build(), doc, thread.getCallerLocation());
  }
}
