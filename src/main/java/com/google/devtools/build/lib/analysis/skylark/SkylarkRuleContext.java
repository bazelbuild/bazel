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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.ActionsProvider;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.LabelExpander;
import com.google.devtools.build.lib.analysis.LabelExpander.NotUniqueExpansionException;
import com.google.devtools.build.lib.analysis.MakeVariableExpander.ExpansionException;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.FragmentCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression.FuncallException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkIndexable;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.syntax.Type.LabelClass;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
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
          + "marked as <code>executable=True</code>, no corresponding struct field is generated.";
  public static final String FILES_DOC =
      "A <code>struct</code> containing files defined in label or label list "
          + "type attributes. The struct fields correspond to the attribute names. The struct "
          + "values are <code>list</code> of <code>file</code>s. If an optional attribute is "
          + "not specified in the rule, an empty list is generated. "
          + "It is a shortcut for:"
          + "<pre class=language-python>[f for t in ctx.attr.&lt;ATTR&gt; for f in t.files]</pre>";
  public static final String FILE_DOC =
      "A <code>struct</code> containing files defined in label type "
          + "attributes marked as <code>single_file=True</code>. The struct fields correspond "
          + "to the attribute names. The struct value is always a <code>file</code> or "
          + "<code>None</code>. If an optional attribute is not specified in the rule "
          + "then the corresponding struct value is <code>None</code>. If a label type is not "
          + "marked as <code>single_file=True</code>, no corresponding struct field is generated. "
          + "It is a shortcut for:"
          + "<pre class=language-python>list(ctx.attr.&lt;ATTR&gt;.files)[0]</pre>";
  public static final String ATTR_DOC =
      "A struct to access the values of the attributes. The values are provided by "
          + "the user (if not, a default value is used).";
  public static final String SPLIT_ATTR_DOC =
      "A struct to access the values of attributes with split configurations. If the attribute is "
          + "a label list, the value of split_attr is a dict of the keys of the split (as strings) "
          + "to lists of the ConfiguredTargets in that branch of the split. If the attribute is a "
          + "label, then the value of split_attr is a dict of the keys of the split (as strings) "
          + "to single ConfiguredTargets. Attributes with split configurations still appear in the "
          + "attr struct, but their values will be single lists with all the branches of the split "
          + "merged together.";
  public static final String OUTPUTS_DOC =
      "A <code>struct</code> containing all the output files."
          + " The struct is generated the following way:<br>"
          + "<ul><li>If the rule is marked as <code>executable=True</code> the struct has an "
          + "\"executable\" field with the rules default executable <code>file</code> value."
          + "<li>For every entry in the rule's <code>outputs</code> dict an attr is generated with "
          + "the same name and the corresponding <code>file</code> value."
          + "<li>For every output type attribute a struct attribute is generated with the "
          + "same name and the corresponding <code>file</code> value or <code>None</code>, "
          + "if no value is specified in the rule."
          + "<li>For every output list type attribute a struct attribute is generated with the "
          + "same name and corresponding <code>list</code> of <code>file</code>s value "
          + "(an empty list if no value is specified in the rule).</ul>";
  public static final Function<Attribute, Object> ATTRIBUTE_VALUE_EXTRACTOR_FOR_ASPECT =
      new Function<Attribute, Object>() {
        @Nullable
        @Override
        public Object apply(Attribute attribute) {
          return attribute.getDefaultValue(null);
        }
      };

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
  private SkylarkRuleAttributesCollection attributesCollection;
  private SkylarkRuleAttributesCollection ruleAttributesCollection;
  private Info splitAttributes;

  // TODO(bazel-team): we only need this because of the css_binary rule.
  private ImmutableMap<Artifact, Label> artifactsLabelMap;
  private Info outputsObject;

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
    this.fragments = new FragmentCollection(ruleContext, ConfigurationTransition.NONE);
    this.hostFragments = new FragmentCollection(ruleContext, ConfigurationTransition.HOST);
    this.aspectDescriptor = aspectDescriptor;
    this.skylarkSemantics = skylarkSemantics;

    if (aspectDescriptor == null) {
      this.isForAspect = false;
      Collection<Attribute> attributes = ruleContext.getRule().getAttributes();
      HashMap<String, Object> outputsBuilder = new HashMap<>();
      if (ruleContext.getRule().getRuleClassObject().outputsDefaultExecutable()) {
        addOutput(outputsBuilder, "executable", ruleContext.createOutputArtifact());
      }
      ImplicitOutputsFunction implicitOutputsFunction =
          ruleContext.getRule().getImplicitOutputsFunction();

      if (implicitOutputsFunction instanceof SkylarkImplicitOutputsFunction) {
        SkylarkImplicitOutputsFunction func =
            (SkylarkImplicitOutputsFunction) implicitOutputsFunction;
        for (Map.Entry<String, String> entry :
            func.calculateOutputs(RawAttributeMapper.of(ruleContext.getRule())).entrySet()) {
          addOutput(
              outputsBuilder,
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
            addOutput(outputsBuilder, attrName, Iterables.getOnlyElement(artifacts));
          } else {
            addOutput(outputsBuilder, attrName, Runtime.NONE);
          }
        } else if (type == BuildType.OUTPUT_LIST) {
          addOutput(outputsBuilder, attrName, SkylarkList.createImmutable(artifacts));
        } else {
          throw new IllegalArgumentException(
              "Type of " + attrName + "(" + type + ") is not output type ");
        }
      }

      this.artifactsLabelMap = artifactLabelMapBuilder.build();
      this.outputsObject =
          NativeProvider.STRUCT.create(
              outputsBuilder,
              "No attribute '%s' in outputs. Make sure you declared a rule output with this name.");

      this.attributesCollection =
          buildAttributesCollection(
              this, attributes, ruleContext, attributeValueExtractorForRule(ruleContext));
      this.splitAttributes = buildSplitAttributeInfo(attributes, ruleContext);
      this.ruleAttributesCollection = null;
    } else { // ASPECT
      this.isForAspect = true;
      this.artifactsLabelMap = ImmutableMap.of();
      this.outputsObject = null;
      this.attributesCollection =
          buildAttributesCollection(
              this,
              ruleContext.getAspectAttributes().values(),
              ruleContext,
              ATTRIBUTE_VALUE_EXTRACTOR_FOR_ASPECT);
      this.splitAttributes = null;
      this.ruleAttributesCollection =
          buildAttributesCollection(
              this,
              ruleContext.getRule().getAttributes(),
              ruleContext,
              attributeValueExtractorForRule(ruleContext));
    }

    makeVariables = ruleContext.getConfigurationMakeVariableContext().collectMakeVariables();
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

  private Function<Attribute, Object> attributeValueExtractorForRule(
      final RuleContext ruleContext) {
    return new Function<Attribute, Object>() {
      @Nullable
      @Override
      public Object apply(Attribute attribute) {
        return ruleContext.attributes().get(attribute.getName(), attribute.getType());
      }
    };
  }

  private static SkylarkRuleAttributesCollection buildAttributesCollection(
      SkylarkRuleContext skylarkRuleContext,
      Collection<Attribute> attributes,
      RuleContext ruleContext,
      Function<Attribute, Object> attributeValueExtractor) {
    Builder<String, Object> attrBuilder = new Builder<>();
    Builder<String, Object> executableBuilder = new Builder<>();
    Builder<Artifact, FilesToRunProvider> executableRunfilesbuilder = new Builder<>();
    Builder<String, Object> fileBuilder = new Builder<>();
    Builder<String, Object> filesBuilder = new Builder<>();
    HashSet<Artifact> seenExecutables = new HashSet<>();
    for (Attribute a : attributes) {
      Type<?> type = a.getType();
      Object val = attributeValueExtractor.apply(a);
      // TODO(mstaib): Remove the LABEL_DICT_UNARY special case of this conditional
      // LABEL_DICT_UNARY was previously not treated as a dependency-bearing type, and was put into
      // Skylark as a Map<String, Label>; this special case preserves that behavior temporarily.
      if (type.getLabelClass() != LabelClass.DEPENDENCY || type == BuildType.LABEL_DICT_UNARY) {
        attrBuilder.put(a.getPublicName(), val == null ? Runtime.NONE
            // Attribute values should be type safe
            : SkylarkType.convertToSkylark(val, null));
        continue;
      }
      String skyname = a.getPublicName();
      if (a.isExecutable()) {
        // In Skylark only label (not label list) type attributes can have the Executable flag.
        FilesToRunProvider provider =
            ruleContext.getExecutablePrerequisite(a.getName(), Mode.DONT_CHECK);
        if (provider != null && provider.getExecutable() != null) {
          Artifact executable = provider.getExecutable();
          executableBuilder.put(skyname, executable);
          if (!seenExecutables.contains(executable)) {
            // todo(dslomov,laurentlb): In general, this is incorrect.
            // We associate the first encountered FilesToRunProvider with
            // the executable (this provider is later used to build the spawn).
            // However ideally we should associate a provider with the attribute name,
            // and pass the correct FilesToRunProvider to the spawn depending on
            // what attribute is used to access the executable.
            executableRunfilesbuilder.put(executable, provider);
            seenExecutables.add(executable);
          }
        } else {
          executableBuilder.put(skyname, Runtime.NONE);
        }
      }
      if (a.isSingleArtifact()) {
        // In Skylark only label (not label list) type attributes can have the SingleArtifact flag.
        Artifact artifact = ruleContext.getPrerequisiteArtifact(a.getName(), Mode.DONT_CHECK);
        if (artifact != null) {
          fileBuilder.put(skyname, artifact);
        } else {
          fileBuilder.put(skyname, Runtime.NONE);
        }
      }
      filesBuilder.put(
          skyname, ruleContext.getPrerequisiteArtifacts(a.getName(), Mode.DONT_CHECK).list());

      if (type == BuildType.LABEL && !a.hasSplitConfigurationTransition()) {
        Object prereq = ruleContext.getPrerequisite(a.getName(), Mode.DONT_CHECK);
        if (prereq == null) {
          prereq = Runtime.NONE;
        }
        attrBuilder.put(skyname, prereq);
      } else if (type == BuildType.LABEL_LIST
          || (type == BuildType.LABEL && a.hasSplitConfigurationTransition())) {
        List<?> allPrereq = ruleContext.getPrerequisites(a.getName(), Mode.DONT_CHECK);
        attrBuilder.put(skyname, SkylarkList.createImmutable(allPrereq));
      } else if (type == BuildType.LABEL_KEYED_STRING_DICT) {
        ImmutableMap.Builder<TransitiveInfoCollection, String> builder =
            new ImmutableMap.Builder<>();
        Map<Label, String> original = BuildType.LABEL_KEYED_STRING_DICT.cast(val);
        List<? extends TransitiveInfoCollection> allPrereq =
            ruleContext.getPrerequisites(a.getName(), Mode.DONT_CHECK);
        for (TransitiveInfoCollection prereq : allPrereq) {
          builder.put(prereq, original.get(AliasProvider.getDependencyLabel(prereq)));
        }
        attrBuilder.put(skyname, SkylarkType.convertToSkylark(builder.build(), null));
      } else if (type == BuildType.LABEL_DICT_UNARY) {
        Map<Label, TransitiveInfoCollection> prereqsByLabel = new LinkedHashMap<>();
        for (TransitiveInfoCollection target
              : ruleContext.getPrerequisites(a.getName(), Mode.DONT_CHECK)) {
          prereqsByLabel.put(target.getLabel(), target);
        }
        ImmutableMap.Builder<String, TransitiveInfoCollection> attrValue =
            new ImmutableMap.Builder<>();
        for (Map.Entry<String, Label> entry : ((Map<String, Label>) val).entrySet()) {
          attrValue.put(entry.getKey(), prereqsByLabel.get(entry.getValue()));
        }
        attrBuilder.put(skyname, attrValue.build());
      } else {
        throw new IllegalArgumentException(
            "Can't transform attribute " + a.getName() + " of type " + type
            + " to a Skylark object");
      }
    }

    return new SkylarkRuleAttributesCollection(
        skylarkRuleContext,
        ruleContext.getRule().getRuleClass(),
        attrBuilder.build(),
        executableBuilder.build(),
        fileBuilder.build(),
        filesBuilder.build(),
        executableRunfilesbuilder.build());
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

  @SkylarkModule(
    name = "rule_attributes",
    category = SkylarkModuleCategory.NONE,
    doc = "Information about attributes of a rule an aspect is applied to."
  )
  private static class SkylarkRuleAttributesCollection implements SkylarkValue {
    private final SkylarkRuleContext skylarkRuleContext;
    private final Info attrObject;
    private final Info executableObject;
    private final Info fileObject;
    private final Info filesObject;
    private final ImmutableMap<Artifact, FilesToRunProvider> executableRunfilesMap;
    private final String ruleClassName;

    private SkylarkRuleAttributesCollection(
        SkylarkRuleContext skylarkRuleContext,
        String ruleClassName, ImmutableMap<String, Object> attrs,
        ImmutableMap<String, Object> executables,
        ImmutableMap<String, Object> singleFiles,
        ImmutableMap<String, Object> files,
        ImmutableMap<Artifact, FilesToRunProvider> executableRunfilesMap) {
      this.skylarkRuleContext = skylarkRuleContext;
      this.ruleClassName = ruleClassName;
      attrObject =
          NativeProvider.STRUCT.create(
              attrs,
              "No attribute '%s' in attr. Make sure you declared a rule attribute with this name.");
      executableObject =
          NativeProvider.STRUCT.create(
              executables,
              "No attribute '%s' in executable. Make sure there is a label type attribute marked "
                  + "as 'executable' with this name");
      fileObject =
          NativeProvider.STRUCT.create(
              singleFiles,
              "No attribute '%s' in file. Make sure there is a label type attribute marked "
                  + "as 'single_file' with this name");
      filesObject =
          NativeProvider.STRUCT.create(
              files,
              "No attribute '%s' in files. Make sure there is a label or label_list type attribute "
                  + "with this name");
      this.executableRunfilesMap = executableRunfilesMap;
    }

    private void checkMutable(String attrName) throws EvalException {
      skylarkRuleContext.checkMutable("rule." + attrName);
    }

    @SkylarkCallable(name = "attr", structField = true, doc = ATTR_DOC)
    public Info getAttr() throws EvalException {
      checkMutable("attr");
      return attrObject;
    }

    @SkylarkCallable(name = "executable", structField = true, doc = EXECUTABLE_DOC)
    public Info getExecutable() throws EvalException {
      checkMutable("executable");
      return executableObject;
    }

    @SkylarkCallable(name = "file", structField = true, doc = FILE_DOC)
    public Info getFile() throws EvalException {
      checkMutable("file");
      return fileObject;
    }

    @SkylarkCallable(name = "files", structField = true, doc = FILES_DOC)
    public Info getFiles() throws EvalException {
      checkMutable("files");
      return filesObject;
    }

    @SkylarkCallable(name = "kind", structField = true, doc =
        "The kind of a rule, such as 'cc_library'")
    public String getRuleClassName() throws EvalException {
      checkMutable("kind");
      return ruleClassName;
    }

    public ImmutableMap<Artifact, FilesToRunProvider> getExecutableRunfilesMap() {
      return executableRunfilesMap;
    }

    @Override
    public boolean isImmutable() {
      return skylarkRuleContext.isImmutable();
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("<rule collection for " + skylarkRuleContext.ruleLabelCanonicalName + ">");
    }

    @Override
    public void reprLegacy(SkylarkPrinter printer) {
      printer.append("rule_collection:");
      printer.repr(skylarkRuleContext);
    }
  }

  private void addOutput(HashMap<String, Object> outputsBuilder, String key, Object value)
      throws EvalException {
    if (outputsBuilder.containsKey(key)) {
      throw new EvalException(null, "Multiple outputs with the same key: " + key);
    }
    outputsBuilder.put(key, value);
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
  @Override
  public void reprLegacy(SkylarkPrinter printer) {
    printer.append(ruleLabelCanonicalName);
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

  @SkylarkCallable(name = "bin_dir", structField = true,
      doc = "The root corresponding to bin directory.")
  public Root getBinDirectory() throws EvalException {
    checkMutable("bin_dir");
    return getConfiguration().getBinDirectory(ruleContext.getRule().getRepository());
  }

  @SkylarkCallable(name = "genfiles_dir", structField = true,
      doc = "The root corresponding to genfiles directory.")
  public Root getGenfilesDirectory() throws EvalException {
    checkMutable("genfiles_dir");
    return getConfiguration().getGenfilesDirectory(ruleContext.getRule().getRepository());
  }

  @SkylarkCallable(structField = true, doc = OUTPUTS_DOC)
  public Info outputs() throws EvalException {
    checkMutable("outputs");
    if (outputsObject == null) {
      throw new EvalException(Location.BUILTIN, "'outputs' is not defined");
    }
    return outputsObject;
  }

  @SkylarkCallable(structField = true,
      doc = "Returns rule attributes descriptor for the rule that aspect is applied to."
          + " Only available in aspect implementation functions.")
  public SkylarkRuleAttributesCollection rule() throws EvalException {
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
    SkylarkRuleImplementationFunctions.checkDeprecated(
        "ctx.actions.declare_file", "ctx.new_file", null, skylarkSemantics);
    checkMutable("new_file");
    return actionFactory.declareFile(filename, Runtime.NONE);
  }

  // Kept for compatibility with old code.
  @SkylarkCallable(documented = false)
  public Artifact newFile(Root root, String filename) throws EvalException {
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
    SkylarkRuleImplementationFunctions.checkDeprecated(
        "ctx.actions.declare_file", "ctx.new_file", null, skylarkSemantics);
    checkMutable("new_file");
    return actionFactory.declareFile(newBaseName, baseArtifact);
  }

  // Kept for compatibility with old code.
  @SkylarkCallable(documented = false)
  public Artifact newFile(Root root, Artifact baseArtifact, String suffix) throws EvalException {
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
    SkylarkRuleImplementationFunctions.checkDeprecated(
        "ctx.actions.declare_directory", "ctx.experimental_new_directory", null, skylarkSemantics);
    checkMutable("experimental_new_directory");
    return actionFactory.declareDirectory(name, siblingArtifactUnchecked);
  }

  @SkylarkCallable(documented = false)
  public NestedSet<Artifact> middleMan(String attribute) throws EvalException {
    checkMutable("middle_man");
    return AnalysisUtils.getMiddlemanFor(ruleContext, attribute, Mode.HOST);
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
          public String lookupMakeVariable(String variableName) throws ExpansionException {
            if (additionalSubstitutions.containsKey(variableName)) {
              return additionalSubstitutions.get(variableName);
            } else {
              return super.lookupMakeVariable(variableName);
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
    Root root = Root.asSourceRoot(pkg.getSourceRoot(),
        pkg.getPackageIdentifier().getRepository().isMain());
    return pkg.getBuildFile().getPath().relativeTo(root.getPath()).getPathString();
  }
}
