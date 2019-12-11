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

package com.google.devtools.build.lib.bazel.repository.skylark;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.syntax.SkylarkType.castMap;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.skylark.SkylarkAttr.Descriptor;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AttributeValueSource;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.packages.SkylarkExportable;
import com.google.devtools.build.lib.packages.WorkspaceFactoryHelper;
import com.google.devtools.build.lib.skylarkbuildapi.repository.RepositoryModuleApi;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.DebugFrame;
import com.google.devtools.build.lib.syntax.DotExpression;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.SkylarkUtils;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import java.util.Map;

/**
 * The Skylark module containing the definition of {@code repository_rule} function to define a
 * skylark remote repository.
 */
public class SkylarkRepositoryModule implements RepositoryModuleApi {

  @Override
  public BaseFunction repositoryRule(
      StarlarkFunction implementation,
      Object attrs,
      Boolean local,
      Sequence<?> environ, // <String> expected
      Boolean configure,
      Boolean remotable,
      String doc,
      FuncallExpression ast,
      StarlarkThread funcallThread)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(funcallThread, "repository_rule", ast.getLocation());
    StarlarkSemantics semantics = funcallThread.getSemantics();
    // We'll set the name later, pass the empty string for now.
    RuleClass.Builder builder = new RuleClass.Builder("", RuleClassType.WORKSPACE, true);

    builder.addOrOverrideAttribute(attr("$local", BOOLEAN).defaultValue(local).build());
    builder.addOrOverrideAttribute(attr("$configure", BOOLEAN).defaultValue(configure).build());
    if (semantics.experimentalRepoRemoteExec()) {
      builder.addOrOverrideAttribute(attr("$remotable", BOOLEAN).defaultValue(remotable).build());
      BaseRuleClasses.execPropertiesAttribute(builder);
    }
    builder.addOrOverrideAttribute(
        attr("$environ", STRING_LIST).defaultValue(environ).build());
    BaseRuleClasses.nameAttribute(builder);
    BaseRuleClasses.commonCoreAndSkylarkAttributes(builder);
    builder.add(attr("expect_failure", STRING));
    if (attrs != Starlark.NONE) {
      for (Map.Entry<String, Descriptor> attr :
          castMap(attrs, String.class, Descriptor.class, "attrs").entrySet()) {
        Descriptor attrDescriptor = attr.getValue();
        AttributeValueSource source = attrDescriptor.getValueSource();
        String attrName = source.convertToNativeName(attr.getKey(), ast.getLocation());
        builder.addOrOverrideAttribute(attrDescriptor.build(attrName));
      }
    }
    builder.setConfiguredTargetFunction(implementation);
    builder.setRuleDefinitionEnvironmentLabelAndHashCode(
        (Label) funcallThread.getGlobals().getLabel(),
        funcallThread.getTransitiveContentHashCode());
    builder.setWorkspaceOnly();
    return new RepositoryRuleFunction(builder, ast.getLocation());
  }

  private static final class RepositoryRuleFunction extends BaseFunction
      implements SkylarkExportable {
    private final RuleClass.Builder builder;
    private Label extensionLabel;
    private String exportedName;
    private final Location ruleClassDefinitionLocation;

    public RepositoryRuleFunction(RuleClass.Builder builder, Location ruleClassDefinitionLocation) {
      super(FunctionSignature.KWARGS);
      this.builder = builder;
      this.ruleClassDefinitionLocation = ruleClassDefinitionLocation;
    }

    @Override
    public String getName() {
      return "repository_rule";
    }

    @Override
    public void export(Label extensionLabel, String exportedName) {
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
    public Object call(
        Object[] args,
        FuncallExpression ast,
        com.google.devtools.build.lib.syntax.StarlarkThread thread)
        throws EvalException, InterruptedException {
      String ruleClassName = null;
      Expression function = ast.getFunction();
      // If the function ever got exported (the common case), we take the name
      // it was exprted to. Only in the not intended case of calling an unexported
      // repository function through an exported macro, we fall back, for lack of
      // alternatives, to the name in the local context.
      // TODO(b/111199163): we probably should disallow the use of non-exported
      // repository rules anyway.
      if (isExported()) {
        ruleClassName = exportedName;
      } else if (function instanceof Identifier) {
        ruleClassName = ((Identifier) function).getName();
      } else if (function instanceof DotExpression) {
        ruleClassName = ((DotExpression) function).getField().getName();
      } else {
        // TODO: Remove the wrong assumption that a  "function name" always exists and is relevant
        throw new IllegalStateException("Function is not an identifier or method call");
      }
      try {
        RuleClass ruleClass = builder.build(ruleClassName, ruleClassName);
        PackageContext context = PackageFactory.getContext(thread, ast.getLocation());
        Package.Builder packageBuilder = context.getBuilder();

        @SuppressWarnings("unchecked")
        Map<String, Object> attributeValues = (Map<String, Object>) args[0];
        String externalRepoName = (String) attributeValues.get("name");

        StringBuilder callStack =
            new StringBuilder("Call stack for the definition of repository '")
                .append(externalRepoName)
                .append("' which is a ")
                .append(ruleClassName)
                .append(" (rule definition at ")
                .append(ruleClassDefinitionLocation.toString())
                .append("):");
        for (DebugFrame frame : thread.listFrames(ast.getLocation())) {
          callStack.append("\n - ").append(frame.location().toString());
        }

        WorkspaceFactoryHelper.addMainRepoEntry(
            packageBuilder, externalRepoName, thread.getSemantics());

        WorkspaceFactoryHelper.addRepoMappings(
            packageBuilder, attributeValues, externalRepoName, ast.getLocation());

        Rule rule =
            WorkspaceFactoryHelper.createAndAddRepositoryRule(
                context.getBuilder(),
                ruleClass,
                null,
                WorkspaceFactoryHelper.getFinalKwargs(attributeValues),
                ast.getLocation(),
                callStack.toString());
        return rule;
      } catch (InvalidRuleException | NameConflictException | LabelSyntaxException e) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
    }
  }

  @Override
  public void failWithIncompatibleUseCcConfigureFromRulesCc(
      Location location, StarlarkThread thread) throws EvalException {
    if (thread.getSemantics().incompatibleUseCcConfigureFromRulesCc()) {
      throw new EvalException(
          location,
          "Incompatible flag "
              + "--incompatible_use_cc_configure_from_rules_cc has been flipped. Please use "
              + "cc_configure and related logic from https://github.com/bazelbuild/rules_cc. "
              + "See https://github.com/bazelbuild/bazel/issues/10134 for details and migration "
              + "instructions.");
    }
  }
}
