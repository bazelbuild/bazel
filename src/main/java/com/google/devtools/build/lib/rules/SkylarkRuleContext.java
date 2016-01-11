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
package com.google.devtools.build.lib.rules;

import com.google.common.base.Function;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.LabelExpander;
import com.google.devtools.build.lib.analysis.LabelExpander.NotUniqueExpansionException;
import com.google.devtools.build.lib.analysis.MakeVariableExpander.ExpansionException;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.FragmentCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression.FuncallException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A Skylark API for the ruleContext.
 */
@SkylarkModule(name = "ctx", doc = "The context of the rule containing helper functions and "
    + "information about attributes, depending targets and outputs. "
    + "You get a ctx object as an argument to the <code>implementation</code> function when "
    + "you create a rule.")
public final class SkylarkRuleContext {

  public static final String PROVIDER_CLASS_PREFIX = "com.google.devtools.build.lib.";

  private static final String DOC_NEW_FILE_TAIL = "Does not actually create a file on the file "
      + "system, just declares that some action will do so. You must create an action that "
      + "generates the file. If the file should be visible to other rules, declare a rule output "
      + "instead when possible. Doing so enables Blaze to associate a label with the file that "
      + "rules can refer to (allowing finer dependency control) instead of referencing the whole "
      + "rule.";

  static final LoadingCache<String, Class<?>> classCache = CacheBuilder.newBuilder()
      .initialCapacity(10)
      .maximumSize(100)
      .build(new CacheLoader<String, Class<?>>() {

      @Override
      public Class<?> load(String key) throws Exception {
        String classPath = SkylarkRuleContext.PROVIDER_CLASS_PREFIX + key;
        return Class.forName(classPath);
      }
    });
  public static final String EXECUTABLE_DOC =
      "A <code>struct</code> containing executable files defined in label type "
          + "attributes marked as <code>executable=True</code>. The struct fields correspond "
          + "to the attribute names. Each struct value is always a <code>file</code>s or "
          + "<code>None</code>. If an optional attribute is not specified in the rule "
          + "then the corresponding struct value is <code>None</code>. If a label type is not "
          + "marked as <code>executable=True</code>, no corresponding struct field is generated.";
  public static final String FILES_DOC =
      "A <code>struct</code> containing files defined in label or label list "
          + "type attributes. The struct fields correspond to the attribute names. The struct "
          + "values are <code>list</code> of <code>file</code>s. If an optional attribute is "
          + "not specified in the rule, an empty list is generated."
          + "It is a shortcut for:"
          + "<pre class=language-python>[f for t in ctx.attr.<ATTR> for f in t.files]</pre>";
  public static final String FILE_DOC =
      "A <code>struct</code> containing files defined in label type "
          + "attributes marked as <code>single_file=True</code>. The struct fields correspond "
          + "to the attribute names. The struct value is always a <code>file</code> or "
          + "<code>None</code>. If an optional attribute is not specified in the rule "
          + "then the corresponding struct value is <code>None</code>. If a label type is not "
          + "marked as <code>single_file=True</code>, no corresponding struct field is generated. "
          + "It is a shortcut for:"
          + "<pre class=language-python>list(ctx.attr.<ATTR>.files)[0]</pre>";
  public static final String ATTR_DOC =
      "A struct to access the values of the attributes. The values are provided by "
          + "the user (if not, a default value is used).";
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

  private final RuleContext ruleContext;

  private final FragmentCollection fragments;

  private final FragmentCollection hostFragments;

  private final ImmutableMap<String, String> makeVariables;
  private final SkylarkRuleAttributesCollection attributesCollection;
  private final SkylarkRuleAttributesCollection ruleAttributesCollection;

  // TODO(bazel-team): we only need this because of the css_binary rule.
  private final ImmutableMap<Artifact, Label> artifactsLabelMap;
  private final SkylarkClassObject outputsObject;

  /**
   * Determines whether this context is for rule implementation or for aspect implementation.
   */
  public enum Kind {
    RULE,
    ASPECT
  }

  /**
   * Creates a new SkylarkRuleContext using ruleContext.
   * @throws InterruptedException
   */
  public SkylarkRuleContext(RuleContext ruleContext, Kind kind)
      throws EvalException, InterruptedException {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    fragments = new FragmentCollection(ruleContext, ConfigurationTransition.NONE);
    hostFragments = new FragmentCollection(ruleContext, ConfigurationTransition.HOST);

    if (kind == Kind.RULE) {
      Collection<Attribute> attributes = ruleContext.getRule().getAttributes();
      HashMap<String, Object> outputsBuilder = new HashMap<>();
      if (ruleContext.getRule().getRuleClassObject().outputsDefaultExecutable()) {
        addOutput(outputsBuilder, "executable", ruleContext.createOutputArtifact());
      }
      ImplicitOutputsFunction implicitOutputsFunction =
          ruleContext.getRule().getRuleClassObject().getImplicitOutputsFunction();

      if (implicitOutputsFunction instanceof SkylarkImplicitOutputsFunction) {
        SkylarkImplicitOutputsFunction func =
            (SkylarkImplicitOutputsFunction)
                ruleContext.getRule().getRuleClassObject().getImplicitOutputsFunction();
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
        if (type != BuildType.OUTPUT && type != BuildType.OUTPUT_LIST) {
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
          addOutput(outputsBuilder, attrName, new MutableList(artifacts));
        } else {
          throw new IllegalArgumentException(
              "Type of " + attrName + "(" + type + ") is not output type ");
        }
      }

      this.artifactsLabelMap = artifactLabelMapBuilder.build();
      this.outputsObject =
          new SkylarkClassObject(
              outputsBuilder,
              "No attribute '%s' in outputs. Make sure you declared a rule output with this name.");

      this.attributesCollection =
          buildAttributesCollection(
              attributes, ruleContext, attributeValueExtractorForRule(ruleContext));
      this.ruleAttributesCollection = null;
    } else { // ASPECT
      this.artifactsLabelMap = ImmutableMap.of();
      this.outputsObject = null;
      this.attributesCollection =
          buildAttributesCollection(
              ruleContext.getAspectAttributes().values(),
              ruleContext,
              ATTRIBUTE_VALUE_EXTRACTOR_FOR_ASPECT);
      this.ruleAttributesCollection =
          buildAttributesCollection(
              ruleContext.getRule().getAttributes(),
              ruleContext,
              attributeValueExtractorForRule(ruleContext));
    }

    makeVariables = ruleContext.getConfigurationMakeVariableContext().collectMakeVariables();
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
      Collection<Attribute> attributes,
      RuleContext ruleContext,
      Function<Attribute, Object> attributeValueExtractor) {
    Builder<String, Object> attrBuilder = new Builder<>();
    Builder<String, Object> executableBuilder = new Builder<>();
    Builder<Artifact, FilesToRunProvider> executableRunfilesbuilder = new Builder<>();
    Builder<String, Object> fileBuilder = new Builder<>();
    Builder<String, Object> filesBuilder = new Builder<>();
    for (Attribute a : attributes) {
      Type<?> type = a.getType();
      Object val = attributeValueExtractor.apply(a);
      if (type != BuildType.LABEL && type != BuildType.LABEL_LIST) {
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
          executableRunfilesbuilder.put(executable, provider);
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
      List<?> allPrereq = ruleContext.getPrerequisites(a.getName(), Mode.DONT_CHECK);
      if (type == BuildType.LABEL) {
        Object prereq = ruleContext.getPrerequisite(a.getName(), Mode.DONT_CHECK);
        if (prereq == null) {
          prereq = Runtime.NONE;
        }
        attrBuilder.put(skyname, prereq);
      } else {
        // Type.LABEL_LIST
        attrBuilder.put(skyname, new MutableList(allPrereq));
      }
    }

    return new SkylarkRuleAttributesCollection(
        ruleContext.getRule().getRuleClass(),
        attrBuilder.build(),
        executableBuilder.build(),
        fileBuilder.build(),
        filesBuilder.build(),
        executableRunfilesbuilder.build());
  }

  @SkylarkModule(
    name = "rule_attributes",
    doc = "Information about attributes of a rule an aspect is applied to."
  )
  private static class SkylarkRuleAttributesCollection {
    private final SkylarkClassObject attrObject;
    private final SkylarkClassObject executableObject;
    private final SkylarkClassObject fileObject;
    private final SkylarkClassObject filesObject;
    private final ImmutableMap<Artifact, FilesToRunProvider> executableRunfilesMap;
    private final String ruleClassName;

    private SkylarkRuleAttributesCollection(
        String ruleClassName, ImmutableMap<String, Object> attrs,
        ImmutableMap<String, Object> executables,
        ImmutableMap<String, Object> singleFiles,
        ImmutableMap<String, Object> files,
        ImmutableMap<Artifact, FilesToRunProvider> executableRunfilesMap) {
      this.ruleClassName = ruleClassName;
      attrObject =
          new SkylarkClassObject(
              attrs,
              "No attribute '%s' in attr. Make sure you declared a rule attribute with this name.");
      executableObject =
          new SkylarkClassObject(
              executables,
              "No attribute '%s' in executable. Make sure there is a label type attribute marked "
                  + "as 'executable' with this name");
      fileObject =
          new SkylarkClassObject(
              singleFiles,
              "No attribute '%s' in file. Make sure there is a label type attribute marked "
                  + "as 'single_file' with this name");
      filesObject =
          new SkylarkClassObject(
              files,
              "No attribute '%s' in files. Make sure there is a label or label_list type attribute "
                  + "with this name");
      this.executableRunfilesMap = executableRunfilesMap;
    }

    @SkylarkCallable(name = "attr", structField = true, doc = ATTR_DOC)
    public SkylarkClassObject getAttr() {
      return attrObject;
    }

    @SkylarkCallable(name = "executable", structField = true, doc = EXECUTABLE_DOC)
    public SkylarkClassObject getExecutable() {
      return executableObject;
    }

    @SkylarkCallable(name = "file", structField = true, doc = FILE_DOC)
    public SkylarkClassObject getFile() {
      return fileObject;
    }

    @SkylarkCallable(name = "files", structField = true, doc = FILES_DOC)
    public SkylarkClassObject getFiles() {
      return filesObject;
    }

    @SkylarkCallable(name = "kind", structField = true, doc =
        "The kind of a rule, such as 'cc_library'")
    public String getRuleClassName() {
      return ruleClassName;
    }


    public ImmutableMap<Artifact, FilesToRunProvider> getExecutableRunfilesMap() {
      return executableRunfilesMap;
    }
  }

  private void addOutput(HashMap<String, Object> outputsBuilder, String key, Object value)
      throws EvalException {
    if (outputsBuilder.containsKey(key)) {
      throw new EvalException(null, "Multiple outputs with the same key: " + key);
    }
    outputsBuilder.put(key, value);
  }

  /**
   * Returns the original ruleContext.
   */
  public RuleContext getRuleContext() {
    return ruleContext;
  }

  @SkylarkCallable(name = "attr", structField = true, doc = ATTR_DOC)
  public SkylarkClassObject getAttr() {
    return attributesCollection.getAttr();
  }

  /**
   * <p>See {@link RuleContext#getExecutablePrerequisite(String, Mode)}.
   */
  @SkylarkCallable(name = "executable", structField = true, doc = EXECUTABLE_DOC)
  public SkylarkClassObject getExecutable() {
    return attributesCollection.getExecutable();
  }

  /**
   * See {@link RuleContext#getPrerequisiteArtifact(String, Mode)}.
   */
  @SkylarkCallable(name = "file", structField = true, doc = FILE_DOC)
  public SkylarkClassObject getFile() {
    return attributesCollection.getFile();
  }

  /**
   * See {@link RuleContext#getPrerequisiteArtifacts(String, Mode)}.
   */
  @SkylarkCallable(name = "files", structField = true, doc = FILES_DOC)
  public SkylarkClassObject getFiles() {
    return attributesCollection.getFiles();
  }

  @SkylarkCallable(name = "workspace_name", structField = true,
      doc = "Returns the workspace name as defined in the WORKSPACE file.")
  public String getWorkspaceName() {
    return ruleContext.getWorkspaceName();
  }

  @SkylarkCallable(name = "label", structField = true, doc = "The label of this rule.")
  public Label getLabel() {
    return ruleContext.getLabel();
  }

  @SkylarkCallable(name = "fragments", structField = true,
      doc =
      "Allows access to configuration fragments in target configuration. "
      + "Possible fields are <code>cpp</code>, "
      + "<code>java</code> and <code>jvm</code>. "
      + "However, rules have to declare their required fragments in order to access them "
      + "(see <a href=\"../rules.html#fragments\">here</a>).")
  public FragmentCollection getFragments() {
    return fragments;
  }

  @SkylarkCallable(name = "host_fragments", structField = true,
      doc =
      "Allows access to configuration fragments in host configuration. "
      + "Possible fields are <code>cpp</code>, "
      + "<code>java</code> and <code>jvm</code>. "
      + "However, rules have to declare their required fragments in order to access them "
      + "(see <a href=\"../rules.html#fragments\">here</a>).")
  public FragmentCollection getHostFragments() {
    return hostFragments;
  }

  @SkylarkCallable(name = "configuration", structField = true,
      doc = "Returns the default configuration. See the <a href=\"configuration.html\">"
          + "configuration</a> type for more details.")
  public BuildConfiguration getConfiguration() {
    return ruleContext.getConfiguration();
  }

  @SkylarkCallable(name = "host_configuration", structField = true,
      doc = "Returns the host configuration. See the <a href=\"configuration.html\">"
          + "configuration</a> type for more details.")
  public BuildConfiguration getHostConfiguration() {
    return ruleContext.getHostConfiguration();
  }

  @SkylarkCallable(structField = true, doc = OUTPUTS_DOC)
  public SkylarkClassObject outputs() throws EvalException {
    if (outputsObject == null) {
      throw new EvalException(Location.BUILTIN, "'outputs' is not defined");
    }
    return outputsObject;
  }

  @SkylarkCallable(structField = true,
      doc = "Returns rule attributes descriptor for the rule that aspect is applied to."
          + " Only avaliable in aspect implementation functions.")
  public SkylarkRuleAttributesCollection rule() throws EvalException {
    if (ruleAttributesCollection == null) {
      throw new EvalException(
          Location.BUILTIN, "'rule' is only available in aspect implementations");
    }
    return ruleAttributesCollection;
  }

  @SkylarkCallable(structField = true,
      doc = "Dictionary (String to String) of configuration variables")
  public ImmutableMap<String, String> var() {
    return makeVariables;
  }

  @Override
  public String toString() {
    return ruleContext.getLabel().toString();
  }

  @SkylarkCallable(doc = "Splits a shell command to a list of tokens.", documented = false)
  public MutableList tokenize(String optionString) throws FuncallException {
    List<String> options = new ArrayList<>();
    try {
      ShellUtils.tokenize(options, optionString);
    } catch (TokenizationException e) {
      throw new FuncallException(e.getMessage() + " while tokenizing '" + optionString + "'");
    }
    return new MutableList(options); // no env is provided, so it's effectively immutable
  }

  @SkylarkCallable(
    doc =
        "Expands all references to labels embedded within a string for all files using a mapping "
          + "from definition labels (i.e. the label in the output type attribute) to files. "
          + "Deprecated.",
    documented = false
  )
  public String expand(@Nullable String expression, SkylarkList artifacts, Label labelResolver)
      throws EvalException, FuncallException {
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

  @SkylarkCallable(
    doc =
        "Creates a file object with the given filename, in the current package. "
            + DOC_NEW_FILE_TAIL
  )
  public Artifact newFile(String filename) {
    return newFile(ruleContext.getBinOrGenfilesDirectory(), filename);
  }

  // Kept for compatibility with old code.
  @SkylarkCallable(documented = false)
  public Artifact newFile(Root root, String filename) {
    return ruleContext.getPackageRelativeArtifact(filename, root);
  }

  @SkylarkCallable(doc =
      "Creates a new file object in the same directory as the original file. "
      + DOC_NEW_FILE_TAIL)
  public Artifact newFile(Artifact baseArtifact, String newBaseName) {
    PathFragment original = baseArtifact.getRootRelativePath();
    PathFragment fragment = original.replaceName(newBaseName);
    Root root = ruleContext.getBinOrGenfilesDirectory();
    return ruleContext.getDerivedArtifact(fragment, root);
  }

  // Kept for compatibility with old code.
  @SkylarkCallable(documented = false)
  public Artifact newFile(Root root, Artifact baseArtifact, String suffix) {
    PathFragment original = baseArtifact.getRootRelativePath();
    PathFragment fragment = original.replaceName(original.getBaseName() + suffix);
    return ruleContext.getDerivedArtifact(fragment, root);
  }

  @SkylarkCallable(documented = false)
  public NestedSet<Artifact> middleMan(String attribute) {
    return AnalysisUtils.getMiddlemanFor(ruleContext, attribute);
  }

  @SkylarkCallable(documented = false)
  public boolean checkPlaceholders(String template, SkylarkList allowedPlaceholders)
      throws EvalException {
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
        "Returns a string after expanding all references to \"Make variables\". The variables "
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
      final Map<String, String> additionalSubstitutions) {
    return ruleContext.expandMakeVariables(attributeName,
        command, new ConfigurationMakeVariableContext(ruleContext.getRule().getPackage(),
            ruleContext.getConfiguration()) {
          @Override
          public String lookupMakeVariable(String name) throws ExpansionException {
            if (additionalSubstitutions.containsKey(name)) {
              return additionalSubstitutions.get(name);
            } else {
              return super.lookupMakeVariable(name);
            }
          }
        });
  }


  FilesToRunProvider getExecutableRunfiles(Artifact executable) {
    return attributesCollection.getExecutableRunfilesMap().get(executable);
  }

  @SkylarkCallable(name = "info_file", structField = true, documented = false,
      doc = "Returns the file that is used to hold the non-volatile workspace status for the " 
          + "current build request.")
  public Artifact getStableWorkspaceStatus() {
    return ruleContext.getAnalysisEnvironment().getStableWorkspaceStatusArtifact();
  }

  @SkylarkCallable(name = "version_file", structField = true, documented = false,
      doc = "Returns the file that is used to hold the volatile workspace status for the "
          + "current build request.")
  public Artifact getVolatileWorkspaceStatus() {
    return ruleContext.getAnalysisEnvironment().getVolatileWorkspaceStatusArtifact();
  }
}
