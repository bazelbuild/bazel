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

package com.google.devtools.build.lib.analysis.skylark;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.ActionsProvider;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.LabelExpander;
import com.google.devtools.build.lib.analysis.LabelExpander.NotUniqueExpansionException;
import com.google.devtools.build.lib.analysis.LocationExpander;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.FragmentCollection;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.AbstractConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.stringtemplate.ExpansionException;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression.FuncallException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkIndexable;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.syntax.Type.LabelClass;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import javax.annotation.Nullable;

/** A Skylark API for the ruleContext.
 *
 * "This object becomes featureless once the rule implementation function that it was created for
 * has completed. To achieve this, the {@link #nullify()} should be called once the evaluation of
 * the function is completed. The method both frees memory by deleting all significant fields of the
 * object and makes it impossible to accidentally use this object where it's not supposed to be used
 * (such attempts will result in {@link EvalException}s).
 */
@SkylarkModule(
  name = "ctx",
  category = SkylarkModuleCategory.BUILTIN,
  doc =
      "The context of the rule containing helper functions and "
          + "information about attributes, depending targets and outputs. "
          + "You get a ctx object as an argument to the <code>implementation</code> function when "
          + "you create a rule."
)
public final class SkylarkRuleContext implements SkylarkValue {

  private static final String DOC_NEW_FILE_TAIL = "Does not actually create a file on the file "
      + "system, just declares that some action will do so. You must create an action that "
      + "generates the file. If the file should be visible to other rules, declare a rule output "
      + "instead when possible. Doing so enables Blaze to associate a label with the file that "
      + "rules can refer to (allowing finer dependency control) instead of referencing the whole "
      + "rule.";
  public static final String EXECUTABLE_DOC =
      "A <code>struct</code> containing executable files defined in label type "
          + "attributes marked as <code>executable=True</code>. The struct fields correspond "
          + "to the attribute names. Each value in the struct is either a <code>file</code> or "
          + "<code>None</code>. If an optional attribute is not specified in the rule "
          + "then the corresponding struct value is <code>None</code>. If a label type is not "
          + "marked as <code>executable=True</code>, no corresponding struct field is generated. "
          + "<a href=\"https://github.com/bazelbuild/examples/blob/master/rules/actions_run/"
          + "execute.bzl\">See example of use</a>.";
  public static final String FILES_DOC =
      "A <code>struct</code> containing files defined in label or label list "
          + "type attributes. The struct fields correspond to the attribute names. The struct "
          + "values are <code>list</code> of <code>file</code>s.  "
          + "It is a shortcut for:"
          + "<pre class=language-python>[f for t in ctx.attr.&lt;ATTR&gt; for f in t.files]</pre> "
          + "In other words, use <code>files</code> to access the "
          + "<a href=\"../rules.$DOC_EXT#requesting-output-files\">default outputs</a> of a "
          + "dependency. "
          + "<a href=\"https://github.com/bazelbuild/examples/blob/master/rules/depsets/foo.bzl\">"
          + "See example of use</a>.";
  public static final String FILE_DOC =
      "A <code>struct</code> containing files defined in label type "
          + "attributes marked as <code>allow_single_file</code>. The struct fields correspond "
          + "to the attribute names. The struct value is always a <code>file</code> or "
          + "<code>None</code>. If an optional attribute is not specified in the rule "
          + "then the corresponding struct value is <code>None</code>. If a label type is not "
          + "marked as <code>allow_single_file</code>, no corresponding struct field is generated. "
          + "It is a shortcut for:"
          + "<pre class=language-python>list(ctx.attr.&lt;ATTR&gt;.files)[0]</pre>"
          + "In other words, use <code>file</code> to access the (singular) "
          + "<a href=\"../rules.$DOC_EXT#requesting-output-files\">default output</a> of a "
          + "dependency. "
          + "<a href=\"https://github.com/bazelbuild/examples/blob/master/rules/expand_template/"
          + "hello.bzl\">See example of use</a>.";
  public static final String ATTR_DOC =
      "A struct to access the values of the attributes. The values are provided by "
          + "the user (if not, a default value is used). The attributes of the struct and the "
          + "types of their values correspond to the keys and values of the <code>attrs</code> "
          + "dict provided to the <code>rule</code> function. "
          + "<a href=\"https://github.com/bazelbuild/examples/blob/master/rules/attributes/"
          + "printer.bzl\">See example of use</a>.";
  public static final String SPLIT_ATTR_DOC =
      "A struct to access the values of attributes with split configurations. If the attribute is "
          + "a label list, the value of split_attr is a dict of the keys of the split (as strings) "
          + "to lists of the ConfiguredTargets in that branch of the split. If the attribute is a "
          + "label, then the value of split_attr is a dict of the keys of the split (as strings) "
          + "to single ConfiguredTargets. Attributes with split configurations still appear in the "
          + "attr struct, but their values will be single lists with all the branches of the split "
          + "merged together.";
  public static final String OUTPUTS_DOC =
      "A pseudo-struct containing all the predeclared output files, represented by "
          + "<a href='File.html'><code>File</code></a> objects. See the "
          + "<a href='../rules.$DOC_EXT#files'>Rules page</a> for more information and examples."
          + "<p>This field does not exist on aspect contexts, since aspects do not have "
          + "predeclared outputs."
          + "<p>The fields of this object are defined as follows. It is an error if two outputs "
          + "produce the same field name or have the same label."
          + "<ul>"
          + "<li>If the rule declares an <a href='globals.html#rule.outputs'><code>outputs</code>"
          + "</a> dict, then for every entry in the dict, there is a field whose name is the key "
          + "and whose value is the corresponding <code>File</code>."
          + "<li>For every attribute of type <a href='attr.html#output'><code>attr.output</code>"
          + "</a> that the rule declares, there is a field whose name is the attribute's name. "
          + "If the target specified a label for that attribute, then the field value is the "
          + "corresponding <code>File</code>; otherwise the field value is <code>None</code>."
          + "<li>For every attribute of type <a href='attr.html#output_list'><code>attr.output_list"
          + "</code></a> that the rule declares, there is a field whose name is the attribute's "
          + "name. The field value is a list of <code>File</code> objects corresponding to the "
          + "labels given for that attribute in the target, or an empty list if the attribute was "
          + "not specified in the target."
          + "<li><b>(Deprecated)</b> If the rule is marked <a href='globals.html#rule.executable'>"
          + "<code>executable</code></a> or <a href='globals.html#rule.test'><code>test</code></a>,"
          + "there is a field named <code>\"executable\"</code>, which is the default executable. "
          + "It is recommended that instead of using this, you pass another file (either "
          + "predeclared or not) to the <code>executable</code> arg of "
          + "<a href='globals.html#DefaultInfo'><code>DefaultInfo</code></a>."
          + "</ul>";
  public static final Function<Attribute, Object> ATTRIBUTE_VALUE_EXTRACTOR_FOR_ASPECT =
      new Function<Attribute, Object>() {
        @Nullable
        @Override
        public Object apply(Attribute attribute) {
          return attribute.getDefaultValue(null);
        }
      };
  public static final String EXECUTABLE_OUTPUT_NAME = "executable";

  // This field is a copy of the info from ruleContext, stored separately so it can be accessed
  // after this object has been nullified.
  private final String ruleLabelCanonicalName;

  private final boolean isForAspect;

  private final SkylarkActionFactory actionFactory;

  // The fields below intended to be final except that they can be cleared by calling `nullify()`
  // when the object becomes featureless.
  private RuleContext ruleContext;
  private FragmentCollection fragments;
  private FragmentCollection hostFragments;
  private AspectDescriptor aspectDescriptor;
  private final SkylarkSemantics skylarkSemantics;

  private SkylarkDict<String, String> makeVariables;
  private SkylarkAttributesCollection attributesCollection;
  private SkylarkAttributesCollection ruleAttributesCollection;
  private Info splitAttributes;

  // TODO(bazel-team): we only need this because of the css_binary rule.
  private ImmutableMap<Artifact, Label> artifactsLabelMap;
  private Outputs outputsObject;

  /**
   * Creates a new SkylarkRuleContext using ruleContext.
   * @param aspectDescriptor aspect for which the context is created, or <code>null</code>
   *        if it is for a rule.
   * @throws InterruptedException
   */
  public SkylarkRuleContext(RuleContext ruleContext,
      @Nullable AspectDescriptor aspectDescriptor,
      SkylarkSemantics skylarkSemantics)
      throws EvalException, InterruptedException {
    this.actionFactory = new SkylarkActionFactory(this, skylarkSemantics, ruleContext);
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.ruleLabelCanonicalName = ruleContext.getLabel().getCanonicalForm();
    this.fragments = new FragmentCollection(ruleContext, NoTransition.INSTANCE);
    this.hostFragments = new FragmentCollection(ruleContext, HostTransition.INSTANCE);
    this.aspectDescriptor = aspectDescriptor;
    this.skylarkSemantics = skylarkSemantics;

    if (aspectDescriptor == null) {
      this.isForAspect = false;
      Collection<Attribute> attributes = ruleContext.getRule().getAttributes();
      Outputs outputs = new Outputs(this);

      ImplicitOutputsFunction implicitOutputsFunction =
          ruleContext.getRule().getImplicitOutputsFunction();

      if (implicitOutputsFunction instanceof SkylarkImplicitOutputsFunction) {
        SkylarkImplicitOutputsFunction func =
            (SkylarkImplicitOutputsFunction) implicitOutputsFunction;
        for (Map.Entry<String, String> entry :
            func.calculateOutputs(
                    ruleContext.getAnalysisEnvironment().getEventHandler(),
                    RawAttributeMapper.of(ruleContext.getRule()))
                .entrySet()) {
          outputs.addOutput(
              entry.getKey(),
              ruleContext.getImplicitOutputArtifact(entry.getValue()));
        }
      }

      Builder<Artifact, Label> artifactLabelMapBuilder = ImmutableMap.builder();
      for (Attribute a : attributes) {
        String attrName = a.getName();
        Type<?> type = a.getType();
        if (type.getLabelClass() != LabelClass.OUTPUT) {
          continue;
        }
        ImmutableList.Builder<Artifact> artifactsBuilder = ImmutableList.builder();
        for (OutputFile outputFile : ruleContext.getRule().getOutputFileMap().get(attrName)) {
          Artifact artifact = ruleContext.createOutputArtifact(outputFile);
          artifactsBuilder.add(artifact);
          artifactLabelMapBuilder.put(artifact, outputFile.getLabel());
        }
        ImmutableList<Artifact> artifacts = artifactsBuilder.build();

        if (type == BuildType.OUTPUT) {
          if (artifacts.size() == 1) {
            outputs.addOutput(attrName, Iterables.getOnlyElement(artifacts));
          } else {
            outputs.addOutput(attrName, Runtime.NONE);
          }
        } else if (type == BuildType.OUTPUT_LIST) {
          outputs.addOutput(attrName, SkylarkList.createImmutable(artifacts));
        } else {
          throw new IllegalArgumentException(
              "Type of " + attrName + "(" + type + ") is not output type ");
        }
      }

      this.artifactsLabelMap = artifactLabelMapBuilder.build();
      this.outputsObject = outputs;

      SkylarkAttributesCollection.Builder builder = SkylarkAttributesCollection.builder(this);
      for (Attribute attribute : ruleContext.getRule().getAttributes()) {
        Object value = ruleContext.attributes().get(attribute.getName(), attribute.getType());
        builder.addAttribute(attribute, value);
      }

      this.attributesCollection = builder.build();
      this.splitAttributes = buildSplitAttributeInfo(attributes, ruleContext);
      this.ruleAttributesCollection = null;
    } else { // ASPECT
      this.isForAspect = true;
      this.artifactsLabelMap = ImmutableMap.of();
      this.outputsObject = null;

      ImmutableCollection<Attribute> attributes =
          ruleContext.getMainAspect().getDefinition().getAttributes().values();
      SkylarkAttributesCollection.Builder aspectBuilder = SkylarkAttributesCollection.builder(this);
      for (Attribute attribute : attributes) {
        aspectBuilder.addAttribute(attribute, attribute.getDefaultValue(null));
      }
      this.attributesCollection = aspectBuilder.build();

      this.splitAttributes = null;
      SkylarkAttributesCollection.Builder ruleBuilder = SkylarkAttributesCollection.builder(this);

      for (Attribute attribute : ruleContext.getRule().getAttributes()) {
        Object value = ruleContext.attributes().get(attribute.getName(), attribute.getType());
        ruleBuilder.addAttribute(attribute, value);
      }
      for (Aspect aspect : ruleContext.getAspects()) {
        if (aspect.equals(ruleContext.getMainAspect())) {
          // Aspect's own attributes are in <code>attributesCollection</code>.
          continue;
        }
        for (Attribute attribute : aspect.getDefinition().getAttributes().values()) {
          ruleBuilder.addAttribute(attribute, attribute.getDefaultValue(null));
        }
      }

      this.ruleAttributesCollection = ruleBuilder.build();
    }

    makeVariables = ruleContext.getConfigurationMakeVariableContext().collectMakeVariables();
  }

  /**
   * Represents `ctx.outputs`.
   *
   * <p>A {@link ClassObject} (struct-like data structure) with "executable" field created
   * lazily on-demand.
   *
   * <p>Note: There is only one {@code Outputs} object per rule context, so default
   * (object identity) equals and hashCode suffice.
   */
  private static class Outputs implements ClassObject, SkylarkValue {
    private final Map<String, Object> outputs;
    private final SkylarkRuleContext context;
    private boolean executableCreated = false;

    public Outputs(SkylarkRuleContext context) {
      this.outputs = new LinkedHashMap<>();
      this.context = context;
    }

    private void addOutput(String key, Object value)
        throws EvalException {
      Preconditions.checkState(!context.isImmutable(),
          "Cannot add outputs to immutable Outputs object");
      if (outputs.containsKey(key)
          || (context.isExecutable() && EXECUTABLE_OUTPUT_NAME.equals(key))) {
        throw new EvalException(null, "Multiple outputs with the same key: " + key);
      }
      outputs.put(key, value);
    }


    @Override
    public boolean isImmutable() {
      return context.isImmutable();
    }

    @Override
    public ImmutableCollection<String> getFieldNames() throws EvalException {
      checkMutable();
      ImmutableList.Builder<String> result = ImmutableList.builder();
      if (context.isExecutable() && executableCreated) {
        result.add(EXECUTABLE_OUTPUT_NAME);
      }
      result.addAll(outputs.keySet());
      return result.build();
    }

    @Nullable
    @Override
    public Object getValue(String name) throws EvalException {
      checkMutable();
      if (context.isExecutable() && EXECUTABLE_OUTPUT_NAME.equals(name)) {
        executableCreated = true;
        // createOutputArtifact() will cache the created artifact.
        return context.getRuleContext().createOutputArtifact();
      }

      return outputs.get(name);
    }

    @Nullable
    @Override
    public String getErrorMessageForUnknownField(String name) {
      return String.format(
          "No attribute '%s' in outputs. Make sure you declared a rule output with this name.",
          name);
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      if (isImmutable()) {
        printer.append("ctx.outputs(for ");
        printer.append(context.ruleLabelCanonicalName);
        printer.append(")");
        return;
      }
      boolean first = true;
      printer.append("ctx.outputs(");
      // Sort by field name to ensure deterministic output.
      try {
        for (String field : Ordering.natural().sortedCopy(getFieldNames())) {
          if (!first) {
            printer.append(", ");
          }
          first = false;
          printer.append(field);
          printer.append(" = ");
          printer.repr(getValue(field));
        }
        printer.append(")");
      } catch (EvalException e) {
        throw new AssertionError("mutable ctx.outputs should not throw", e);
      }
    }

    private void checkMutable() throws EvalException {
      if (isImmutable()) {
        throw new EvalException(
            null,
            String.format(
                "cannot access outputs of rule '%s' outside of its own "
                    + "rule implementation function",
                context.ruleLabelCanonicalName));
      }
    }

  }

  public boolean isExecutable() {
    return ruleContext.getRule().getRuleClassObject().isExecutableSkylark();
  }

  public boolean isDefaultExecutableCreated() {
    return this.outputsObject.executableCreated;
  }


  /**
   * Nullifies fields of the object when it's not supposed to be used anymore to free unused memory
   * and to make sure this object is not accessed when it's not supposed to (after the corresponding
   * rule implementation function has exited).
   */
  public void nullify() {
    actionFactory.nullify();
    ruleContext = null;
    fragments = null;
    hostFragments = null;
    aspectDescriptor = null;
    makeVariables = null;
    attributesCollection = null;
    ruleAttributesCollection = null;
    splitAttributes = null;
    artifactsLabelMap = null;
    outputsObject = null;
  }

  public void checkMutable(String attrName) throws EvalException {
    if (isImmutable()) {
      throw new EvalException(null, String.format(
          "cannot access field or method '%s' of rule context for '%s' outside of its own rule " 
              + "implementation function", attrName, ruleLabelCanonicalName));
    }
  }

  @Nullable
  public AspectDescriptor getAspectDescriptor() {
    return aspectDescriptor;
  }

  public String getRuleLabelCanonicalName() {
    return ruleLabelCanonicalName;
  }

  private static Info buildSplitAttributeInfo(
      Collection<Attribute> attributes, RuleContext ruleContext) {

    ImmutableMap.Builder<String, Object> splitAttrInfos = ImmutableMap.builder();
    for (Attribute attr : attributes) {

      if (attr.hasSplitConfigurationTransition()) {

        Map<Optional<String>, ? extends List<? extends TransitiveInfoCollection>> splitPrereqs =
            ruleContext.getSplitPrerequisites(attr.getName());

        Map<Object, Object> splitPrereqsMap = new LinkedHashMap<>();
        for (Entry<Optional<String>, ? extends List<? extends TransitiveInfoCollection>> splitPrereq
            : splitPrereqs.entrySet()) {

          Object value;
          if (attr.getType() == BuildType.LABEL) {
            Preconditions.checkState(splitPrereq.getValue().size() == 1);
            value = splitPrereq.getValue().get(0);
          } else {
            // BuildType.LABEL_LIST
            value = SkylarkList.createImmutable(splitPrereq.getValue());
          }

          if (splitPrereq.getKey().isPresent()) {
            splitPrereqsMap.put(splitPrereq.getKey().get(), value);
          } else {
            // If the split transition is not in effect, then the key will be missing since there's
            // nothing to key on because the dependencies aren't split and getSplitPrerequisites()
            // behaves like getPrerequisites(). This also means there should be only one entry in
            // the map. Use None in Skylark to represent this.
            Preconditions.checkState(splitPrereqs.size() == 1);
            splitPrereqsMap.put(Runtime.NONE, value);
          }
        }

        splitAttrInfos.put(attr.getPublicName(), SkylarkDict.copyOf(null, splitPrereqsMap));
      }
    }

    return NativeProvider.STRUCT.create(
        splitAttrInfos.build(),
        "No attribute '%s' in split_attr. Make sure that this attribute is defined with a "
            + "split configuration.");
  }

  @Override
  public boolean isImmutable() {
    return ruleContext == null;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    if (isForAspect) {
      printer.append("<aspect context for " + ruleLabelCanonicalName + ">");
    } else {
      printer.append("<rule context for " + ruleLabelCanonicalName + ">");
    }
  }

  /**
   * Returns the original ruleContext.
   */
  public RuleContext getRuleContext() {
    return ruleContext;
  }

  @SkylarkCallable(
    name = "default_provider",
    structField = true,
    doc = "Deprecated. Use <a href=\"globals.html#DefaultInfo\">DefaultInfo</a> instead."
  )
  public static Provider getDefaultProvider() {
    return DefaultInfo.PROVIDER;
  }

  @SkylarkCallable(
      name = "actions",
      structField = true,
      doc = "Functions to declare files and create actions."
  )
  public SkylarkActionFactory actions() {
    return actionFactory;
  }

  @SkylarkCallable(name = "created_actions",
      doc = "For rules with <a href=\"globals.html#rule._skylark_testable\">_skylark_testable"
          + "</a> set to <code>True</code>, this returns an "
          + "<a href=\"globals.html#Actions\">Actions</a> provider representing all actions "
          + "created so far for the current rule. For all other rules, returns <code>None</code>. "
          + "Note that the provider is not updated when subsequent actions are created, so you "
          + "will have to call this function again if you wish to inspect them. "
          + "<br/><br/>"
          + "This is intended to help write tests for rule-implementation helper functions, which "
          + "may take in a <code>ctx</code> object and create actions on it.")
  public SkylarkValue createdActions() throws EvalException {
    checkMutable("created_actions");
    if (ruleContext.getRule().getRuleClassObject().isSkylarkTestable()) {
      return ActionsProvider.create(
          ruleContext.getAnalysisEnvironment().getRegisteredActions());
    } else {
      return Runtime.NONE;
    }
  }

  @SkylarkCallable(name = "attr", structField = true, doc = ATTR_DOC)
  public Info getAttr() throws EvalException {
    checkMutable("attr");
    return attributesCollection.getAttr();
  }

  @SkylarkCallable(name = "split_attr", structField = true, doc = SPLIT_ATTR_DOC)
  public Info getSplitAttr() throws EvalException {
    checkMutable("split_attr");
    if (splitAttributes == null) {
      throw new EvalException(
          Location.BUILTIN, "'split_attr' is available only in rule implementations");
    }
    return splitAttributes;
  }

  /** See {@link RuleContext#getExecutablePrerequisite(String, Mode)}. */
  @SkylarkCallable(name = "executable", structField = true, doc = EXECUTABLE_DOC)
  public Info getExecutable() throws EvalException {
    checkMutable("executable");
    return attributesCollection.getExecutable();
  }

  /** See {@link RuleContext#getPrerequisiteArtifact(String, Mode)}. */
  @SkylarkCallable(name = "file", structField = true, doc = FILE_DOC)
  public Info getFile() throws EvalException {
    checkMutable("file");
    return attributesCollection.getFile();
  }

  /** See {@link RuleContext#getPrerequisiteArtifacts(String, Mode)}. */
  @SkylarkCallable(name = "files", structField = true, doc = FILES_DOC)
  public Info getFiles() throws EvalException {
    checkMutable("files");
    return attributesCollection.getFiles();
  }

  @SkylarkCallable(name = "workspace_name", structField = true,
      doc = "Returns the workspace name as defined in the WORKSPACE file.")
  public String getWorkspaceName() throws EvalException {
    checkMutable("workspace_name");
    return ruleContext.getWorkspaceName();
  }

  @SkylarkCallable(name = "label", structField = true, doc = "The label of this rule.")
  public Label getLabel() throws EvalException {
    checkMutable("label");
    return ruleContext.getLabel();
  }

  @SkylarkCallable(name = "fragments", structField = true,
      doc = "Allows access to configuration fragments in target configuration.")
  public FragmentCollection getFragments() throws EvalException {
    checkMutable("fragments");
    return fragments;
  }

  @SkylarkCallable(name = "host_fragments", structField = true,
      doc = "Allows access to configuration fragments in host configuration.")
  public FragmentCollection getHostFragments() throws EvalException {
    checkMutable("host_fragments");
    return hostFragments;
  }

  @SkylarkCallable(name = "configuration", structField = true,
      doc = "Returns the default configuration. See the <a href=\"configuration.html\">"
          + "configuration</a> type for more details.")
  public BuildConfiguration getConfiguration() throws EvalException {
    checkMutable("configuration");
    return ruleContext.getConfiguration();
  }

  @SkylarkCallable(name = "host_configuration", structField = true,
      doc = "Returns the host configuration. See the <a href=\"configuration.html\">"
          + "configuration</a> type for more details.")
  public BuildConfiguration getHostConfiguration() throws EvalException {
    checkMutable("host_configuration");
    return ruleContext.getHostConfiguration();
  }

  @SkylarkCallable(name = "coverage_instrumented",
    doc = "Returns whether code coverage instrumentation should be generated when performing "
        + "compilation actions for this rule or, if <code>target</code> is provided, the rule "
        + "specified by that Target. (If a non-rule or a Skylark rule Target is provided, this "
        + "returns False.) Checks if the sources of the current rule (if no Target is provided) or"
        + "the sources of Target should be instrumented based on the --instrumentation_filter and"
        + "--instrument_test_targets config settings. "
        + "This differs from <code>coverage_enabled</code> in the <a href=\"configuration.html\">"
        + "configuration</a>, which notes whether coverage data collection is enabled for the "
        + "entire run, but not whether a specific target should be instrumented.",
    parameters = {
      @Param(
          name = "target",
          type = TransitiveInfoCollection.class,
          defaultValue = "None",
          noneable = true,
          named = true,
          doc = "A Target specifying a rule. If not provided, defaults to the current rule.")
    })
  public boolean instrumentCoverage(Object targetUnchecked) throws EvalException {
    checkMutable("coverage_instrumented");
    BuildConfiguration config = ruleContext.getConfiguration();
    if (!config.isCodeCoverageEnabled()) {
      return false;
    }
    if (targetUnchecked == Runtime.NONE) {
      return InstrumentedFilesCollector.shouldIncludeLocalSources(ruleContext);
    }
    TransitiveInfoCollection target = (TransitiveInfoCollection) targetUnchecked;
    return (target.getProvider(InstrumentedFilesProvider.class) != null)
        && InstrumentedFilesCollector.shouldIncludeLocalSources(config, target);
  }

  @SkylarkCallable(name = "features", structField = true,
      doc = "Returns the set of features that are enabled for this rule."
  )
  public ImmutableList<String> getFeatures() throws EvalException {
    checkMutable("features");
    return ImmutableList.copyOf(ruleContext.getFeatures());
  }

  @SkylarkCallable(
    name = "bin_dir",
    structField = true,
    doc = "The root corresponding to bin directory."
  )
  public ArtifactRoot getBinDirectory() throws EvalException {
    checkMutable("bin_dir");
    return getConfiguration().getBinDirectory(ruleContext.getRule().getRepository());
  }

  @SkylarkCallable(
    name = "genfiles_dir",
    structField = true,
    doc = "The root corresponding to genfiles directory."
  )
  public ArtifactRoot getGenfilesDirectory() throws EvalException {
    checkMutable("genfiles_dir");
    return getConfiguration().getGenfilesDirectory(ruleContext.getRule().getRepository());
  }

  @SkylarkCallable(structField = true, doc = OUTPUTS_DOC)
  public ClassObject outputs() throws EvalException {
    checkMutable("outputs");
    if (outputsObject == null) {
      throw new EvalException(Location.BUILTIN, "'outputs' is not defined");
    }
    return outputsObject;
  }

  @SkylarkCallable(
    structField = true,
    doc =
        "Returns rule attributes descriptor for the rule that aspect is applied to."
            + " Only available in aspect implementation functions."
  )
  public SkylarkAttributesCollection rule() throws EvalException {
    checkMutable("rule");
    if (!isForAspect) {
      throw new EvalException(
          Location.BUILTIN, "'rule' is only available in aspect implementations");
    }
    return ruleAttributesCollection;
  }

  @SkylarkCallable(structField = true,
      name = "aspect_ids",
      doc = "Returns a list ids for all aspects applied to the target."
      + " Only available in aspect implementation functions.")
  public ImmutableList<String> aspectIds() throws EvalException {
    checkMutable("aspect_ids");
    if (!isForAspect) {
      throw new EvalException(
          Location.BUILTIN, "'aspect_ids' is only available in aspect implementations");
    }

    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (AspectDescriptor descriptor : ruleContext.getAspectDescriptors()) {
      result.add(descriptor.getDescription());
    }
    return result.build();
  }

  @SkylarkCallable(
    structField = true,
    doc = "Dictionary (String to String) of configuration variables."
  )
  public SkylarkDict<String, String> var() throws EvalException {
    checkMutable("var");
    return makeVariables;
  }

  @SkylarkCallable(structField = true, doc = "Toolchains required for this rule.")
  public SkylarkIndexable toolchains() throws EvalException {
    checkMutable("toolchains");
    return ruleContext.getToolchainContext().getResolvedToolchainProviders();
  }

  @Override
  public String toString() {
    return ruleLabelCanonicalName;
  }

  @SkylarkCallable(doc = "Splits a shell command to a list of tokens.", documented = false)
  public SkylarkList<String> tokenize(String optionString) throws FuncallException, EvalException {
    checkMutable("tokenize");
    List<String> options = new ArrayList<>();
    try {
      ShellUtils.tokenize(options, optionString);
    } catch (TokenizationException e) {
      throw new FuncallException(e.getMessage() + " while tokenizing '" + optionString + "'");
    }
    return SkylarkList.createImmutable(options);
  }

  @SkylarkCallable(
    doc =
        "Expands all references to labels embedded within a string for all files using a mapping "
          + "from definition labels (i.e. the label in the output type attribute) to files. "
          + "Deprecated.",
    documented = false
  )
  public String expand(
      @Nullable String expression, SkylarkList<Object> artifacts, Label labelResolver)
      throws EvalException, FuncallException {
    checkMutable("expand");
    try {
      Map<Label, Iterable<Artifact>> labelMap = new HashMap<>();
      for (Artifact artifact : artifacts.getContents(Artifact.class, "artifacts")) {
        labelMap.put(artifactsLabelMap.get(artifact), ImmutableList.of(artifact));
      }
      return LabelExpander.expand(expression, labelMap, labelResolver);
    } catch (NotUniqueExpansionException e) {
      throw new FuncallException(e.getMessage() + " while expanding '" + expression + "'");
    }
  }

  boolean isForAspect() {
    return isForAspect;
  }

  @SkylarkCallable(
    name = "new_file",
    doc =
        "DEPRECATED. Use <a href=\"actions.html#declare_file\">ctx.actions.declare_file</a>. <br>"
            + "Creates a file object with the given filename, in the current package. "
            + DOC_NEW_FILE_TAIL,
    parameters = {
      @Param(
        name = "filename",
        type = String.class,
        doc = "The path of the new file, relative to the current package."
      )
    }
  )
  public Artifact newFile(String filename) throws EvalException {
    checkDeprecated("ctx.actions.declare_file", "ctx.new_file", null, skylarkSemantics);
    checkMutable("new_file");
    return actionFactory.declareFile(filename, Runtime.NONE);
  }

  // Kept for compatibility with old code.
  @SkylarkCallable(documented = false)
  public Artifact newFile(ArtifactRoot root, String filename) throws EvalException {
    checkMutable("new_file");
    return ruleContext.getPackageRelativeArtifact(filename, root);
  }

  @SkylarkCallable(
      name = "new_file",
      doc =
          "Creates a new file object in the same directory as the original file. "
              + DOC_NEW_FILE_TAIL,
      parameters = {
        @Param(
          name = "sibling_file",
          type = Artifact.class,
          doc = "A file that lives in the same directory as the newly created file."
        ),
        @Param(
          name = "basename",
          type = String.class,
          doc = "The base name of the newly created file."
        )
      }
    )
  public Artifact newFile(Artifact baseArtifact, String newBaseName) throws EvalException {
    checkDeprecated("ctx.actions.declare_file", "ctx.new_file", null, skylarkSemantics);
    checkMutable("new_file");
    return actionFactory.declareFile(newBaseName, baseArtifact);
  }

  // Kept for compatibility with old code.
  @SkylarkCallable(documented = false)
  public Artifact newFile(ArtifactRoot root, Artifact baseArtifact, String suffix)
      throws EvalException {
    checkMutable("new_file");
    PathFragment original = baseArtifact.getRootRelativePath();
    PathFragment fragment = original.replaceName(original.getBaseName() + suffix);
    return ruleContext.getDerivedArtifact(fragment, root);
  }

  @SkylarkCallable(
    name = "experimental_new_directory",
    documented = false,
    parameters = {
      @Param(name = "name", type = String.class),
      @Param(
        name = "sibling",
        type = Artifact.class,
        defaultValue = "None",
        noneable = true,
        named = true
      )
    }
  )
  public Artifact newDirectory(String name, Object siblingArtifactUnchecked) throws EvalException {
    checkDeprecated(
        "ctx.actions.declare_directory", "ctx.experimental_new_directory", null, skylarkSemantics);
    checkMutable("experimental_new_directory");
    return actionFactory.declareDirectory(name, siblingArtifactUnchecked);
  }

  @SkylarkCallable(documented = false)
  public boolean checkPlaceholders(String template, SkylarkList<Object> allowedPlaceholders)
      throws EvalException {
    checkMutable("check_placeholders");
    List<String> actualPlaceHolders = new LinkedList<>();
    Set<String> allowedPlaceholderSet =
        ImmutableSet.copyOf(allowedPlaceholders.getContents(String.class, "allowed_placeholders"));
    ImplicitOutputsFunction.createPlaceholderSubstitutionFormatString(template, actualPlaceHolders);
    for (String placeholder : actualPlaceHolders) {
      if (!allowedPlaceholderSet.contains(placeholder)) {
        return false;
      }
    }
    return true;
  }

  @SkylarkCallable(doc =
        "<b>Deprecated.</b> Use <code>ctx.var</code> to access the variables instead.<br>"
      + "Returns a string after expanding all references to \"Make variables\". The variables "
      + "must have the following format: <code>$(VAR_NAME)</code>. Also, <code>$$VAR_NAME"
      + "</code> expands to <code>$VAR_NAME</code>. Parameters:"
      + "<ul><li>The name of the attribute (<code>string</code>). It's only used for error "
      + "reporting.</li>\n"
      + "<li>The expression to expand (<code>string</code>). It can contain references to "
      + "\"Make variables\".</li>\n"
      + "<li>A mapping of additional substitutions (<code>dict</code> of <code>string</code> : "
      + "<code>string</code>).</li></ul>\n"
      + "Examples:"
      + "<pre class=language-python>\n"
      + "ctx.expand_make_variables(\"cmd\", \"$(MY_VAR)\", {\"MY_VAR\": \"Hi\"})  # == \"Hi\"\n"
      + "ctx.expand_make_variables(\"cmd\", \"$$PWD\", {})  # == \"$PWD\"\n"
      + "</pre>"
      + "Additional variables may come from other places, such as configurations. Note that "
      + "this function is experimental.")
  public String expandMakeVariables(String attributeName, String command,
      final Map<String, String> additionalSubstitutions) throws EvalException {
    checkMutable("expand_make_variables");
    ConfigurationMakeVariableContext makeVariableContext =
        new ConfigurationMakeVariableContext(
            // TODO(lberki): This should be removed. But only after either verifying that no one
            // uses it or providing an alternative.
            ruleContext.getMakeVariables(ImmutableList.of(":cc_toolchain")),
            ruleContext.getRule().getPackage(),
            ruleContext.getConfiguration()) {
          @Override
          public String lookupVariable(String variableName) throws ExpansionException {
            if (additionalSubstitutions.containsKey(variableName)) {
              return additionalSubstitutions.get(variableName);
            } else {
              return super.lookupVariable(variableName);
            }
          }
        };
    return ruleContext.getExpander(makeVariableContext).expand(attributeName, command);
  }


  FilesToRunProvider getExecutableRunfiles(Artifact executable) {
    return attributesCollection.getExecutableRunfilesMap().get(executable);
  }

  @SkylarkCallable(
    name = "info_file",
    structField = true,
    documented = false,
    doc =
        "Returns the file that is used to hold the non-volatile workspace status for the "
            + "current build request."
  )
  public Artifact getStableWorkspaceStatus() throws InterruptedException, EvalException {
    checkMutable("info_file");
    return ruleContext.getAnalysisEnvironment().getStableWorkspaceStatusArtifact();
  }

  @SkylarkCallable(
    name = "version_file",
    structField = true,
    documented = false,
    doc =
        "Returns the file that is used to hold the volatile workspace status for the "
            + "current build request."
  )
  public Artifact getVolatileWorkspaceStatus() throws InterruptedException, EvalException {
    checkMutable("version_file");
    return ruleContext.getAnalysisEnvironment().getVolatileWorkspaceStatusArtifact();
  }

  @SkylarkCallable(
    name = "build_file_path",
    structField = true,
    documented = true,
    doc = "Returns path to the BUILD file for this rule, relative to the source root."
  )
  public String getBuildFileRelativePath() throws EvalException {
    checkMutable("build_file_path");
    Package pkg = ruleContext.getRule().getPackage();
    return pkg.getSourceRoot().relativize(pkg.getBuildFile().getPath()).getPathString();
  }

  /**
   * A Skylark built-in function to create and register a SpawnAction using a dictionary of
   * parameters: action( inputs = [input1, input2, ...], outputs = [output1, output2, ...],
   * executable = executable, arguments = [argument1, argument2, ...], mnemonic = 'Mnemonic',
   * command = 'command', )
   */
  @SkylarkCallable(
    name = "action",
    doc =
        "DEPRECATED. Use <a href=\"actions.html#run\">ctx.actions.run()</a> or"
            + " <a href=\"actions.html#run_shell\">ctx.actions.run_shell()</a>. <br>"
            + "Creates an action that runs an executable or a shell command."
            + " You must specify either <code>command</code> or <code>executable</code>.\n"
            + "Actions and genrules are very similar, but have different use cases. Actions are "
            + "used inside rules, and genrules are used inside macros. Genrules also have make "
            + "variable expansion.",
    parameters = {
      @Param(
        name = "outputs",
        type = SkylarkList.class,
        generic1 = Artifact.class,
        named = true,
        positional = false,
        doc = "List of the output files of the action."
      ),
      @Param(
        name = "inputs",
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
          @ParamType(type = SkylarkNestedSet.class),
        },
        generic1 = Artifact.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = "List of the input files of the action."
      ),
      @Param(
        name = "executable",
        type = Object.class,
        allowedTypes = {
          @ParamType(type = Artifact.class),
          @ParamType(type = String.class),
          @ParamType(type = Runtime.NoneType.class),
        },
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "The executable file to be called by the action."
      ),
      @Param(
        name = "arguments",
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
        },
        defaultValue = "[]",
        named = true,
        positional = false,
        doc =
            "Command line arguments of the action."
                + "Must be a list of strings or actions.args() objects."
      ),
      @Param(
        name = "mnemonic",
        type = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "A one-word description of the action, e.g. CppCompile or GoLink."
      ),
      @Param(
        name = "command",
        type = Object.class,
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = SkylarkList.class, generic1 = String.class),
          @ParamType(type = Runtime.NoneType.class),
        },
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "Shell command to execute. It is usually preferable to "
                + "use <code>executable</code> instead. "
                + "Arguments are available with <code>$1</code>, <code>$2</code>, etc."
      ),
      @Param(
        name = "progress_message",
        type = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "Progress message to show to the user during the build, "
                + "e.g. \"Compiling foo.cc to create foo.o\"."
      ),
      @Param(
        name = "use_default_shell_env",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = "Whether the action should use the built in shell environment or not."
      ),
      @Param(
        name = "env",
        type = SkylarkDict.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "Sets the dictionary of environment variables."
      ),
      @Param(
        name = "execution_requirements",
        type = SkylarkDict.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "Information for scheduling the action. See "
                + "<a href=\"$BE_ROOT/common-definitions.html#common.tags\">tags</a> "
                + "for useful keys."
      ),
      @Param(
        // TODO(bazel-team): The name here isn't accurate anymore. This is technically experimental,
        // so folks shouldn't be too attached, but consider renaming to be more accurate/opaque.
        name = "input_manifests",
        type = SkylarkList.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "(Experimental) sets the input runfiles metadata; "
                + "they are typically generated by resolve_command."
      )
    },
    allowReturnNones = true,
    useLocation = true,
    useEnvironment = true
  )
  public Runtime.NoneType action(
      SkylarkList outputs,
      Object inputs,
      Object executableUnchecked,
      Object arguments,
      Object mnemonicUnchecked,
      Object commandUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      Location loc,
      Environment env)
      throws EvalException {
    checkDeprecated(
        "ctx.actions.run or ctx.actions.run_shell", "ctx.action", loc, env.getSemantics());
    checkMutable("action");
    if ((commandUnchecked == Runtime.NONE) == (executableUnchecked == Runtime.NONE)) {
      throw new EvalException(loc, "You must specify either 'command' or 'executable' argument");
    }
    boolean hasCommand = commandUnchecked != Runtime.NONE;
    if (!hasCommand) {
      actions()
          .run(
              outputs,
              inputs,
              executableUnchecked,
              arguments,
              mnemonicUnchecked,
              progressMessage,
              useDefaultShellEnv,
              envUnchecked,
              executionRequirementsUnchecked,
              inputManifestsUnchecked);

    } else {
      actions()
          .runShell(
              outputs,
              inputs,
              arguments,
              mnemonicUnchecked,
              commandUnchecked,
              progressMessage,
              useDefaultShellEnv,
              envUnchecked,
              executionRequirementsUnchecked,
              inputManifestsUnchecked);
    }
    return Runtime.NONE;
  }

  @SkylarkCallable(
    name = "expand_location",
    doc =
        "Expands all <code>$(location ...)</code> templates in the given string by replacing "
            + "<code>$(location //x)</code> with the path of the output file of target //x. "
            + "Expansion only works for labels that point to direct dependencies of this rule or "
            + "that are explicitly listed in the optional argument <code>targets</code>. "
            + "<br/><br/>"
            + "<code>$(location ...)</code> will cause an error if the referenced target has "
            + "multiple outputs. In this case, please use <code>$(locations ...)</code> since it "
            + "produces a space-separated list of output paths. It can be safely used for a "
            + "single output file, too.",
    parameters = {
      @Param(name = "input", type = String.class, doc = "String to be expanded."),
      @Param(
        name = "targets",
        type = SkylarkList.class,
        generic1 = AbstractConfiguredTarget.class,
        defaultValue = "[]",
        named = true,
        doc = "List of targets for additional lookup information."
      ),
    },
    allowReturnNones = true,
    useLocation = true,
    useEnvironment = true
  )
  public String expandLocation(String input, SkylarkList targets, Location loc, Environment env)
      throws EvalException {
    checkMutable("expand_location");
    try {
      return LocationExpander.withExecPaths(
              getRuleContext(),
              makeLabelMap(targets.getContents(TransitiveInfoCollection.class, "targets")))
          .expand(input);
    } catch (IllegalStateException ise) {
      throw new EvalException(loc, ise);
    }
  }

  @SkylarkCallable(
    name = "file_action",
    doc =
        "DEPRECATED. Use <a href =\"actions.html#write\">ctx.actions.write</a> instead. <br>"
            + "Creates a file write action.",
    parameters = {
      @Param(name = "output", type = Artifact.class, named = true, doc = "The output file."),
      @Param(
        name = "content",
        type = String.class,
        named = true,
        doc = "The contents of the file."
      ),
      @Param(
        name = "executable",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        doc = "Whether the output file should be executable (default is False)."
      )
    },
    allowReturnNones = true,
    useLocation = true,
    useEnvironment = true
  )
  public Runtime.NoneType fileAction(
      Artifact output, String content, Boolean executable, Location loc, Environment env)
      throws EvalException {
    checkDeprecated("ctx.actions.write", "ctx.file_action", loc, env.getSemantics());
    checkMutable("file_action");
    actions().write(output, content, executable);
    return Runtime.NONE;
  }

  @SkylarkCallable(
    name = "empty_action",
    doc =
        "DEPRECATED. Use <a href=\"actions.html#do_nothing\">ctx.actions.do_nothing</a> instead."
            + " <br>"
            + "Creates an empty action that neither executes a command nor produces any "
            + "output, but that is useful for inserting 'extra actions'.",
    parameters = {
      @Param(
        name = "mnemonic",
        type = String.class,
        named = true,
        positional = false,
        doc = "A one-word description of the action, e.g. CppCompile or GoLink."
      ),
      @Param(
        name = "inputs",
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
          @ParamType(type = SkylarkNestedSet.class),
        },
        generic1 = Artifact.class,
        named = true,
        positional = false,
        defaultValue = "[]",
        doc = "List of the input files of the action."
      ),
    },
    allowReturnNones = true,
    useLocation = true,
    useEnvironment = true
  )
  public Runtime.NoneType emptyAction(String mnemonic, Object inputs, Location loc, Environment env)
      throws EvalException {
    checkDeprecated("ctx.actions.do_nothing", "ctx.empty_action", loc, env.getSemantics());
    checkMutable("empty_action");
    actions().doNothing(mnemonic, inputs);
    return Runtime.NONE;
  }

  @SkylarkCallable(
    name = "template_action",
    doc =
        "DEPRECATED. "
            + "Use <a href=\"actions.html#expand_template\">ctx.actions.expand_template()</a> "
            + "instead. <br>Creates a template expansion action.",
    parameters = {
      @Param(
        name = "template",
        type = Artifact.class,
        named = true,
        positional = false,
        doc = "The template file, which is a UTF-8 encoded text file."
      ),
      @Param(
        name = "output",
        type = Artifact.class,
        named = true,
        positional = false,
        doc = "The output file, which is a UTF-8 encoded text file."
      ),
      @Param(
        name = "substitutions",
        type = SkylarkDict.class,
        named = true,
        positional = false,
        doc = "Substitutions to make when expanding the template."
      ),
      @Param(
        name = "executable",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = "Whether the output file should be executable (default is False)."
      )
    },
    allowReturnNones = true,
    useLocation = true,
    useEnvironment = true
  )
  public Runtime.NoneType templateAction(
      Artifact template,
      Artifact output,
      SkylarkDict<?, ?> substitutionsUnchecked,
      Boolean executable,
      Location loc,
      Environment env)
      throws EvalException {
    checkDeprecated("ctx.actions.expand_template", "ctx.template_action", loc, env.getSemantics());
    checkMutable("template_action");
    actions().expandTemplate(template, output, substitutionsUnchecked, executable);
    return Runtime.NONE;
  }

  @SkylarkCallable(
    name = "runfiles",
    doc = "Creates a runfiles object.",
    parameters = {
      @Param(
        name = "files",
        type = SkylarkList.class,
        generic1 = Artifact.class,
        named = true,
        defaultValue = "[]",
        doc = "The list of files to be added to the runfiles."
      ),
      // TODO(bazel-team): If we have a memory efficient support for lazy list containing
      // NestedSets we can remove this and just use files = [file] + list(set)
      // Also, allow empty set for init
      @Param(
        name = "transitive_files",
        type = SkylarkNestedSet.class,
        generic1 = Artifact.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        doc =
            "The (transitive) set of files to be added to the runfiles. The depset should "
                + "use the `default` order (which, as the name implies, is the default)."
      ),
      @Param(
        name = "collect_data",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        doc =
            "Whether to collect the data "
                + "runfiles from the dependencies in srcs, data and deps attributes."
      ),
      @Param(
        name = "collect_default",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        doc =
            "Whether to collect the default "
                + "runfiles from the dependencies in srcs, data and deps attributes."
      ),
      @Param(
        name = "symlinks",
        type = SkylarkDict.class,
        defaultValue = "{}",
        named = true,
        doc = "The map of symlinks to be added to the runfiles, prefixed by workspace name."
      ),
      @Param(
        name = "root_symlinks",
        type = SkylarkDict.class,
        defaultValue = "{}",
        named = true,
        doc = "The map of symlinks to be added to the runfiles."
      )
    },
    useLocation = true
  )
  public Runfiles runfiles(
      SkylarkList files,
      Object transitiveFiles,
      Boolean collectData,
      Boolean collectDefault,
      SkylarkDict<?, ?> symlinks,
      SkylarkDict<?, ?> rootSymlinks,
      Location loc)
      throws EvalException, ConversionException {
    checkMutable("runfiles");
    Runfiles.Builder builder =
        new Runfiles.Builder(
            getRuleContext().getWorkspaceName(), getConfiguration().legacyExternalRunfiles());
    boolean checkConflicts = false;
    if (EvalUtils.toBoolean(collectData)) {
      builder.addRunfiles(getRuleContext(), RunfilesProvider.DATA_RUNFILES);
    }
    if (EvalUtils.toBoolean(collectDefault)) {
      builder.addRunfiles(getRuleContext(), RunfilesProvider.DEFAULT_RUNFILES);
    }
    if (!files.isEmpty()) {
      builder.addArtifacts(files.getContents(Artifact.class, "files"));
    }
    if (transitiveFiles != Runtime.NONE) {
      builder.addTransitiveArtifacts(((SkylarkNestedSet) transitiveFiles).getSet(Artifact.class));
    }
    if (!symlinks.isEmpty()) {
      // If Skylark code directly manipulates symlinks, activate more stringent validity checking.
      checkConflicts = true;
      for (Map.Entry<String, Artifact> entry :
          symlinks.getContents(String.class, Artifact.class, "symlinks").entrySet()) {
        builder.addSymlink(PathFragment.create(entry.getKey()), entry.getValue());
      }
    }
    if (!rootSymlinks.isEmpty()) {
      checkConflicts = true;
      for (Map.Entry<String, Artifact> entry :
          rootSymlinks.getContents(String.class, Artifact.class, "root_symlinks").entrySet()) {
        builder.addRootSymlink(PathFragment.create(entry.getKey()), entry.getValue());
      }
    }
    Runfiles runfiles = builder.build();
    if (checkConflicts) {
      runfiles.setConflictPolicy(Runfiles.ConflictPolicy.ERROR);
    }
    return runfiles;
  }

  @SkylarkCallable(
    name = "resolve_command",
    // TODO(bazel-team): The naming here isn't entirely accurate (input_manifests is no longer
    // manifests), but this is experimental/should be opaque to the end user.
    doc =
        "<i>(Experimental)</i> "
            + "Returns a tuple <code>(inputs, command, input_manifests)</code> of the list of "
            + "resolved inputs, the argv list for the resolved command, and the runfiles metadata"
            + "required to run the command, all of them suitable for passing as the same-named "
            + "arguments of the <code>ctx.action</code> method.",
    parameters = {
      @Param(
        name = "command",
        type = String.class, // string
        defaultValue = "''",
        named = true,
        positional = false,
        doc = "Command to resolve."
      ),
      @Param(
        name = "attribute",
        type = String.class, // string
        defaultValue = "None",
        noneable = true,
        named = true,
        positional = false,
        doc = "Name of the associated attribute for which to issue an error, or None."
      ),
      @Param(
        name = "expand_locations",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc =
            "Shall we expand $(location) variables? "
                + "See <a href=\"#expand_location\">ctx.expand_location()</a> for more details."
      ),
      @Param(
        name = "make_variables",
        type = SkylarkDict.class, // dict(string, string)
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "Make variables to expand, or None."
      ),
      @Param(
        name = "tools",
        defaultValue = "[]",
        type = SkylarkList.class,
        generic1 = TransitiveInfoCollection.class,
        named = true,
        positional = false,
        doc = "List of tools (list of targets)."
      ),
      @Param(
        name = "label_dict",
        type = SkylarkDict.class,
        defaultValue = "{}",
        named = true,
        positional = false,
        doc =
            "Dictionary of resolved labels and the corresponding list of Files "
                + "(a dict of Label : list of Files)."
      ),
      @Param(
        name = "execution_requirements",
        type = SkylarkDict.class,
        defaultValue = "{}",
        named = true,
        positional = false,
        doc =
            "Information for scheduling the action to resolve this command. See "
                + "<a href=\"$BE_ROOT/common-definitions.html#common.tags\">tags</a> "
                + "for useful keys."
      ),
    },
    useLocation = true,
    useEnvironment = true
  )
  public Tuple<Object> resolveCommand(
      String command,
      Object attributeUnchecked,
      Boolean expandLocations,
      Object makeVariablesUnchecked,
      SkylarkList tools,
      SkylarkDict<?, ?> labelDictUnchecked,
      SkylarkDict<?, ?> executionRequirementsUnchecked,
      Location loc,
      Environment env)
      throws ConversionException, EvalException {
    checkMutable("resolve_command");
    Label ruleLabel = getLabel();
    Map<Label, Iterable<Artifact>> labelDict = checkLabelDict(labelDictUnchecked, loc, env);
    // The best way to fix this probably is to convert CommandHelper to Skylark.
    CommandHelper helper =
        new CommandHelper(
            getRuleContext(),
            tools.getContents(TransitiveInfoCollection.class, "tools"),
            ImmutableMap.copyOf(labelDict));
    String attribute = Type.STRING.convertOptional(attributeUnchecked, "attribute", ruleLabel);
    if (expandLocations) {
      command =
          helper.resolveCommandAndExpandLabels(command, attribute, /*allowDataInLabel=*/ false);
    }
    if (!EvalUtils.isNullOrNone(makeVariablesUnchecked)) {
      Map<String, String> makeVariables =
          Type.STRING_DICT.convert(makeVariablesUnchecked, "make_variables", ruleLabel);
      command = expandMakeVariables(attribute, command, makeVariables);
    }
    List<Artifact> inputs = new ArrayList<>();
    inputs.addAll(helper.getResolvedTools());

    ImmutableMap<String, String> executionRequirements =
        ImmutableMap.copyOf(
            SkylarkDict.castSkylarkDictOrNoneToDict(
                executionRequirementsUnchecked,
                String.class,
                String.class,
                "execution_requirements"));
    List<String> argv =
        helper.buildCommandLine(command, inputs, SCRIPT_SUFFIX, executionRequirements);
    return Tuple.<Object>of(
        MutableList.copyOf(env, inputs),
        MutableList.copyOf(env, argv),
        helper.getToolsRunfilesSuppliers());
  }

  /**
   * Ensures the given {@link Map} has keys that have {@link Label} type and values that have either
   * {@link Iterable} or {@link SkylarkNestedSet} type, and raises {@link EvalException} otherwise.
   * Returns a corresponding map where any sets are replaced by iterables.
   */
  // TODO(bazel-team): find a better way to typecheck this argument.
  @SuppressWarnings("unchecked")
  private static Map<Label, Iterable<Artifact>> checkLabelDict(
      Map<?, ?> labelDict, Location loc, Environment env) throws EvalException {
    Map<Label, Iterable<Artifact>> convertedMap = new HashMap<>();
    for (Map.Entry<?, ?> entry : labelDict.entrySet()) {
      Object key = entry.getKey();
      if (!(key instanceof Label)) {
        throw new EvalException(loc, Printer.format("invalid key %r in 'label_dict'", key));
      }
      ImmutableList.Builder<Artifact> files = ImmutableList.builder();
      Object val = entry.getValue();
      Iterable<?> valIter;
      try {
        valIter = EvalUtils.toIterableStrict(val, loc, env);
      } catch (EvalException ex) {
        // EvalException is thrown only if the type is wrong.
        throw new EvalException(
            loc, Printer.format("invalid value %r in 'label_dict': " + ex, val));
      }
      for (Object file : valIter) {
        if (!(file instanceof Artifact)) {
          throw new EvalException(loc, Printer.format("invalid value %r in 'label_dict'", val));
        }
        files.add((Artifact) file);
      }
      convertedMap.put((Label) key, files.build());
    }
    return convertedMap;
  }

  /** suffix of script to be used in case the command is too long to fit on a single line */
  private static final String SCRIPT_SUFFIX = ".script.sh";

  private static void checkDeprecated(
      String newApi, String oldApi, Location loc, SkylarkSemantics semantics) throws EvalException {
    if (semantics.incompatibleNewActionsApi()) {
      throw new EvalException(
          loc,
          "Use "
              + newApi
              + " instead of "
              + oldApi
              + ". \n"
              + "Use --incompatible_new_actions_api=false to temporarily disable this check.");
    }
  }

  /**
   * Builds a map: Label -> List of files from the given labels
   *
   * @param knownLabels List of known labels
   * @return Immutable map with immutable collections as values
   */
  private static ImmutableMap<Label, ImmutableCollection<Artifact>> makeLabelMap(
      Iterable<TransitiveInfoCollection> knownLabels) {
    ImmutableMap.Builder<Label, ImmutableCollection<Artifact>> builder = ImmutableMap.builder();

    for (TransitiveInfoCollection current : knownLabels) {
      builder.put(
          AliasProvider.getDependencyLabel(current),
          ImmutableList.copyOf(current.getProvider(FileProvider.class).getFilesToBuild()));
    }

    return builder.build();
  }
}
