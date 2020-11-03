// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAttributesCollectionApi;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;

/** Information about attributes of a rule an aspect is applied to. */
class StarlarkAttributesCollection implements StarlarkAttributesCollectionApi {
  private final StarlarkRuleContext starlarkRuleContext;
  private final StructImpl attrObject;
  private final StructImpl executableObject;
  private final StructImpl fileObject;
  private final StructImpl filesObject;
  private final ImmutableMap<Artifact, FilesToRunProvider> executableRunfilesMap;
  private final String ruleClassName;

  static final String ERROR_MESSAGE_FOR_NO_ATTR =
      "No attribute '%s' in attr. Make sure you declared a rule attribute with this name.";

  private StarlarkAttributesCollection(
      StarlarkRuleContext starlarkRuleContext,
      String ruleClassName,
      Map<String, Object> attrs,
      Map<String, Object> executables,
      Map<String, Object> singleFiles,
      Map<String, Object> files,
      ImmutableMap<Artifact, FilesToRunProvider> executableRunfilesMap) {
    this.starlarkRuleContext = starlarkRuleContext;
    this.ruleClassName = ruleClassName;
    attrObject = StructProvider.STRUCT.create(attrs, ERROR_MESSAGE_FOR_NO_ATTR);
    executableObject =
        StructProvider.STRUCT.create(
            executables,
            "No attribute '%s' in executable. Make sure there is a label type attribute marked "
                + "as 'executable' with this name");
    fileObject =
        StructProvider.STRUCT.create(
            singleFiles,
            "No attribute '%s' in file. Make sure there is a label type attribute marked "
                + "as 'allow_single_file' with this name");
    filesObject =
        StructProvider.STRUCT.create(
            files,
            "No attribute '%s' in files. Make sure there is a label or label_list type attribute "
                + "with this name");
    this.executableRunfilesMap = executableRunfilesMap;
  }

  private void checkMutable(String attrName) throws EvalException {
    starlarkRuleContext.checkMutable("rule." + attrName);
  }

  @Override
  public StructImpl getAttr() throws EvalException {
    checkMutable("attr");
    return attrObject;
  }

  @Override
  public StructImpl getExecutable() throws EvalException {
    checkMutable("executable");
    return executableObject;
  }

  @Override
  public StructImpl getFile() throws EvalException {
    checkMutable("file");
    return fileObject;
  }

  @Override
  public StructImpl getFiles() throws EvalException {
    checkMutable("files");
    return filesObject;
  }

  @Override
  public String getRuleClassName() throws EvalException {
    checkMutable("kind");
    return ruleClassName;
  }

  public ImmutableMap<Artifact, FilesToRunProvider> getExecutableRunfilesMap() {
    return executableRunfilesMap;
  }

  @Override
  public boolean isImmutable() {
    return starlarkRuleContext.isImmutable();
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<rule collection for " + starlarkRuleContext.getRuleLabelCanonicalName() + ">");
  }

  public static Builder builder(StarlarkRuleContext ruleContext) {
    return new Builder(ruleContext);
  }

  public static class Builder {
    private final StarlarkRuleContext context;
    private final LinkedHashMap<String, Object> attrBuilder = new LinkedHashMap<>();
    private final LinkedHashMap<String, Object> executableBuilder = new LinkedHashMap<>();
    private final ImmutableMap.Builder<Artifact, FilesToRunProvider> executableRunfilesbuilder =
        ImmutableMap.builder();
    private final LinkedHashMap<String, Object> fileBuilder = new LinkedHashMap<>();
    private final LinkedHashMap<String, Object> filesBuilder = new LinkedHashMap<>();
    private final HashSet<Artifact> seenExecutables = new HashSet<>();

    private Builder(StarlarkRuleContext ruleContext) {
      this.context = ruleContext;
    }

    @SuppressWarnings("unchecked")
    public void addAttribute(Attribute a, Object val) {
      Type<?> type = a.getType();
      String skyname = a.getPublicName();

      // The first attribute with the same name wins.
      if (attrBuilder.containsKey(skyname)) {
        return;
      }

      // Some legacy native attribute types do not have a valid Starlark type. Avoid exposing
      // these to Starlark.
      if (type == BuildType.DISTRIBUTIONS || type == BuildType.TRISTATE) {
        return;
      }

      // TODO(b/140636597): Remove the LABEL_DICT_UNARY special case of this conditional
      // LABEL_DICT_UNARY was previously not treated as a dependency-bearing type, and was put into
      // Starlark as a Map<String, Label>; this special case preserves that behavior temporarily.
      if (type.getLabelClass() != LabelClass.DEPENDENCY || type == BuildType.LABEL_DICT_UNARY) {
        // Attribute values should be type safe
        attrBuilder.put(skyname, Attribute.valueToStarlark(val));
        return;
      }
      if (a.isExecutable()) {
        // In Starlark only label (not label list) type attributes can have the Executable flag.
        FilesToRunProvider provider =
            context.getRuleContext().getExecutablePrerequisite(a.getName());
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
          executableBuilder.put(skyname, Starlark.NONE);
        }
      }
      if (a.isSingleArtifact()) {
        // In Starlark only label (not label list) type attributes can have the SingleArtifact flag.
        Artifact artifact = context.getRuleContext().getPrerequisiteArtifact(a.getName());
        if (artifact != null) {
          fileBuilder.put(skyname, artifact);
        } else {
          fileBuilder.put(skyname, Starlark.NONE);
        }
      }
      filesBuilder.put(
          skyname,
          StarlarkList.immutableCopyOf(
              context.getRuleContext().getPrerequisiteArtifacts(a.getName()).list()));

      if (type == BuildType.LABEL && !a.getTransitionFactory().isSplit()) {
        Object prereq = context.getRuleContext().getPrerequisite(a.getName());
        if (prereq == null) {
          prereq = Starlark.NONE;
        }
        attrBuilder.put(skyname, prereq);
      } else if (type == BuildType.LABEL_LIST
          || (type == BuildType.LABEL && a.getTransitionFactory().isSplit())) {
        List<?> allPrereq = context.getRuleContext().getPrerequisites(a.getName());
        attrBuilder.put(skyname, StarlarkList.immutableCopyOf(allPrereq));
      } else if (type == BuildType.LABEL_KEYED_STRING_DICT) {
        Dict.Builder<TransitiveInfoCollection, String> builder = Dict.builder();
        Map<Label, String> original = BuildType.LABEL_KEYED_STRING_DICT.cast(val);
        List<? extends TransitiveInfoCollection> allPrereq =
            context.getRuleContext().getPrerequisites(a.getName());
        for (TransitiveInfoCollection prereq : allPrereq) {
          builder.put(prereq, original.get(AliasProvider.getDependencyLabel(prereq)));
        }
        attrBuilder.put(skyname, builder.buildImmutable());
      } else if (type == BuildType.LABEL_DICT_UNARY) {
        Map<Label, TransitiveInfoCollection> prereqsByLabel = new LinkedHashMap<>();
        for (TransitiveInfoCollection target :
            context.getRuleContext().getPrerequisites(a.getName())) {
          prereqsByLabel.put(target.getLabel(), target);
        }
        ImmutableMap.Builder<String, TransitiveInfoCollection> attrValue = ImmutableMap.builder();
        for (Map.Entry<String, Label> entry : ((Map<String, Label>) val).entrySet()) {
          attrValue.put(entry.getKey(), prereqsByLabel.get(entry.getValue()));
        }
        attrBuilder.put(skyname, attrValue.build());
      } else {
        throw new IllegalArgumentException(
            "Can't transform attribute "
                + a.getName()
                + " of type "
                + type
                + " to a Starlark object");
      }
    }

    public StarlarkAttributesCollection build() {
      return new StarlarkAttributesCollection(
          context,
          context.getRuleContext().getRule().getRuleClass(),
          attrBuilder,
          executableBuilder,
          fileBuilder,
          filesBuilder,
          executableRunfilesbuilder.build());
    }
  }
}
