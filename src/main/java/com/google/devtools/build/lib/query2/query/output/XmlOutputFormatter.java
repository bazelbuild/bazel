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
package com.google.devtools.build.lib.query2.query.output;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.FilesetEntry;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.build.lib.query2.compat.FakeLoadTarget;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.SynchronizedDelegatingOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.TransformerFactoryConfigurationError;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.w3c.dom.DOMException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

/**
 * An output formatter that prints the result as XML.
 */
class XmlOutputFormatter extends AbstractUnorderedFormatter {

  private AspectResolver aspectResolver;
  private DependencyFilter dependencyFilter;
  private boolean relativeLocations;
  private QueryOptions queryOptions;

  @Override
  public String getName() {
    return "xml";
  }

  @Override
  public ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
      OutputStream out, QueryOptions options, QueryEnvironment<?> env) {
    return new SynchronizedDelegatingOutputFormatterCallback<>(
        createPostFactoStreamCallback(out, options));
  }

  @Override
  public void setOptions(CommonQueryOptions options, AspectResolver aspectResolver) {
    super.setOptions(options, aspectResolver);
    this.aspectResolver = aspectResolver;
    this.dependencyFilter = FormatUtils.getDependencyFilter(options);
    this.relativeLocations = options.relativeLocations;

    Preconditions.checkArgument(options instanceof QueryOptions);
    this.queryOptions = (QueryOptions) options;
  }

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      final OutputStream out, final QueryOptions options) {
    return new OutputFormatterCallback<Target>() {

      private Document doc;
      private Element queryElem;

      @Override
      public void start() {
        try {
          DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
          doc = factory.newDocumentBuilder().newDocument();
        } catch (ParserConfigurationException e) {
          // This shouldn't be possible: all the configuration is hard-coded.
          throw new IllegalStateException("XML output failed", e);
        }
        doc.setXmlVersion("1.1");
        queryElem = doc.createElement("query");
        queryElem.setAttribute("version", "2");
        doc.appendChild(queryElem);
      }

      @Override
      public void processOutput(Iterable<Target> partialResult)
          throws IOException, InterruptedException {
        for (Target target : partialResult) {
          queryElem.appendChild(createTargetElement(doc, target));
        }
      }

      @Override
      public void close(boolean failFast) throws IOException {
        if (!failFast) {
          try {
            Transformer transformer = TransformerFactory.newInstance().newTransformer();
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            transformer.transform(new DOMSource(doc), new StreamResult(out));
          } catch (TransformerFactoryConfigurationError | TransformerException e) {
            // This shouldn't be possible: all the configuration is hard-coded.
            throw new IllegalStateException("XML output failed", e);
          }
        }
      }
    };
  }

  /**
   * Creates and returns a new DOM tree for the specified build target.
   *
   * XML structure:
   * - element tag is &lt;source-file>, &lt;generated-file> or &lt;rule
   *   class="cc_library">, following the terminology of
   *   {@link Target#getTargetKind()}.
   * - 'name' attribute is target's label.
   * - 'location' attribute is consistent with output of --output location.
   * - rule attributes are represented in the DOM structure.
   * @throws InterruptedException
   */
  private Element createTargetElement(Document doc, Target target)
      throws InterruptedException {
    Element elem;
    if (target instanceof Rule) {
      Rule rule = (Rule) target;
      elem = doc.createElement("rule");
      elem.setAttribute("class", rule.getRuleClass());
      for (Attribute attr : rule.getAttributes()) {
        PossibleAttributeValues values = PossibleAttributeValues.forRuleAndAttribute(rule, attr);
        if (values.getSource() == AttributeValueSource.RULE || queryOptions.xmlShowDefaultValues) {
          Element attrElem = createValueElement(doc, attr.getType(), values);
          attrElem.setAttribute("name", attr.getName());
          elem.appendChild(attrElem);
        }
      }

      // Include explicit elements for all direct inputs and outputs of a rule;
      // this goes beyond what is available from the attributes above, since it
      // may also (depending on options) include implicit outputs,
      // host-configuration outputs, and default values.
      for (Label label : rule.getLabels(dependencyFilter)) {
        Element inputElem = doc.createElement("rule-input");
        inputElem.setAttribute("name", label.toString());
        elem.appendChild(inputElem);
      }
      for (Label label :
          aspectResolver.computeAspectDependencies(target, dependencyFilter).values()) {
        Element inputElem = doc.createElement("rule-input");
        inputElem.setAttribute("name", label.toString());
        elem.appendChild(inputElem);
      }
      for (OutputFile outputFile: rule.getOutputFiles()) {
        Element outputElem = doc.createElement("rule-output");
        outputElem.setAttribute("name", outputFile.getLabel().toString());
        elem.appendChild(outputElem);
      }
      for (String feature : rule.getPackage().getFeatures()) {
        Element outputElem = doc.createElement("rule-default-setting");
        outputElem.setAttribute("name", feature);
        elem.appendChild(outputElem);
      }
    } else if (target instanceof PackageGroup) {
      PackageGroup packageGroup = (PackageGroup) target;
      elem = doc.createElement("package-group");
      elem.setAttribute("name", packageGroup.getName());
      Element includes = createValueElement(doc,
          BuildType.LABEL_LIST,
          packageGroup.getIncludes());
      includes.setAttribute("name", "includes");
      elem.appendChild(includes);
      Element packages =
          createValueElement(doc, Type.STRING_LIST, packageGroup.getContainedPackages());
      packages.setAttribute("name", "packages");
      elem.appendChild(packages);
    } else if (target instanceof OutputFile) {
      OutputFile outputFile = (OutputFile) target;
      elem = doc.createElement("generated-file");
      elem.setAttribute("generating-rule",
                        outputFile.getGeneratingRule().getLabel().toString());
    } else if (target instanceof InputFile) {
      elem = doc.createElement("source-file");
      InputFile inputFile = (InputFile) target;
      if (inputFile.getName().equals("BUILD")) {
        addStarlarkFilesToElement(doc, elem, inputFile);
        addFeaturesToElement(doc, elem, inputFile);
        elem.setAttribute("package_contains_errors",
            String.valueOf(inputFile.getPackage().containsErrors()));
      }

      addPackageGroupsToElement(doc, elem, inputFile);
    } else if (target instanceof EnvironmentGroup) {
      EnvironmentGroup envGroup = (EnvironmentGroup) target;
      elem = doc.createElement("environment-group");
      elem.setAttribute("name", envGroup.getName());
      Element environments = createValueElement(doc,
          BuildType.LABEL_LIST,
          envGroup.getEnvironments());
      environments.setAttribute("name", "environments");
      elem.appendChild(environments);
      Element defaults = createValueElement(doc,
          BuildType.LABEL_LIST,
          envGroup.getDefaults());
      defaults.setAttribute("name", "defaults");
      elem.appendChild(defaults);
    } else if (target instanceof FakeLoadTarget) {
      elem = doc.createElement("source-file");
    } else {
      throw new IllegalArgumentException(target.toString());
    }

    elem.setAttribute("name", target.getLabel().toString());
    String location = FormatUtils.getLocation(target, relativeLocations);
    if (!queryOptions.xmlLineNumbers) {
      int firstColon = location.indexOf(':');
      if (firstColon != -1) {
        location = location.substring(0, firstColon);
      }
    }

    elem.setAttribute("location", location);
    return elem;
  }

  private void addPackageGroupsToElement(Document doc, Element parent, Target target) {
    for (Label visibilityDependency : target.getVisibility().getDependencyLabels()) {
      Element elem = doc.createElement("package-group");
      elem.setAttribute("name", visibilityDependency.toString());
      parent.appendChild(elem);
    }

    for (Label visibilityDeclaration : target.getVisibility().getDeclaredLabels()) {
      Element elem = doc.createElement("visibility-label");
      elem.setAttribute("name", visibilityDeclaration.toString());
      parent.appendChild(elem);
    }
  }

  private void addFeaturesToElement(Document doc, Element parent, InputFile inputFile) {
    for (String feature : inputFile.getPackage().getFeatures()) {
      Element elem = doc.createElement("feature");
      elem.setAttribute("name", feature);
      parent.appendChild(elem);
    }
  }

  private void addStarlarkFilesToElement(Document doc, Element parent, InputFile inputFile)
      throws InterruptedException {
    Iterable<Label> dependencies =
        aspectResolver.computeBuildFileDependencies(inputFile.getPackage());

    for (Label starlarkFileDep : dependencies) {
      Element elem = doc.createElement("load");
      elem.setAttribute("name", starlarkFileDep.toString());
      parent.appendChild(elem);
    }
  }

  /**
   * Creates and returns a new DOM tree for the specified attribute values.
   * For non-configurable attributes, this is a single value. For configurable
   * attributes, this contains one value for each configuration.
   * (Only toplevel values are named attributes; list elements are unnamed.)
   *
   * <p>In the case of configurable attributes, multi-value attributes (e.g. lists)
   * merge all configured lists into an aggregate flattened list. Single-value attributes
   * simply refrain to set a value and annotate the DOM element as configurable.
   *
   * <P>(The ungainly qualified class name is required to avoid ambiguity with
   * OutputFormatter.OutputType.)
   */
  private static Element createValueElement(Document doc, Type<?> type, Iterable<Object> values) {
    // "Import static" with method scope:
    Type<?> FILESET_ENTRY = BuildType.FILESET_ENTRY;
    Type<?> LABEL_LIST = BuildType.LABEL_LIST;
    Type<?> LICENSE = BuildType.LICENSE;
    Type<?> STRING_LIST = Type.STRING_LIST;

    final Element elem;
    final boolean hasMultipleValues = Iterables.size(values) > 1;
    Type<?> elemType = type.getListElementType();
    if (elemType != null) { // it's a list (includes "distribs")
      elem = doc.createElement("list");
      for (Object value : values) {
        for (Object elemValue : (Collection<?>) value) {
          elem.appendChild(createValueElement(doc, elemType, elemValue));
        }
      }
    } else if (type instanceof Type.DictType) {
      Set<Object> visitedValues = new HashSet<>();
      elem = doc.createElement("dict");
      Type.DictType<?, ?> dictType = (Type.DictType<?, ?>) type;
      for (Object value : values) {
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
          if (visitedValues.add(entry.getKey())) {
            Element pairElem = doc.createElement("pair");
            elem.appendChild(pairElem);
            pairElem.appendChild(createValueElement(doc,
                    dictType.getKeyType(), entry.getKey()));
            pairElem.appendChild(createValueElement(doc,
                    dictType.getValueType(), entry.getValue()));
          }
        }
      }
    } else if (type == LICENSE) {
      elem = createSingleValueElement(doc, "license", hasMultipleValues);
      if (!hasMultipleValues) {
        License license = (License) Iterables.getOnlyElement(values);

        Element exceptions = createValueElement(doc, LABEL_LIST, license.getExceptions());
        exceptions.setAttribute("name", "exceptions");
        elem.appendChild(exceptions);

        Element licenseTypes = createValueElement(doc, STRING_LIST, license.getLicenseTypes());
        licenseTypes.setAttribute("name", "license-types");
        elem.appendChild(licenseTypes);
      }
    } else if (type == FILESET_ENTRY) {
      // Fileset entries: not configurable.
      FilesetEntry filesetEntry = (FilesetEntry) Iterables.getOnlyElement(values);
      elem = doc.createElement("fileset-entry");
      elem.setAttribute("srcdir", filesetEntry.getSrcLabel().toString());
      elem.setAttribute("destdir", filesetEntry.getDestDir().toString());
      elem.setAttribute("symlinks", filesetEntry.getSymlinkBehavior().toString());
      elem.setAttribute("strip_prefix", filesetEntry.getStripPrefix());

      if (filesetEntry.getExcludes() != null) {
        Element excludes =
            createValueElement(doc, LABEL_LIST, filesetEntry.getExcludes());
        excludes.setAttribute("name", "excludes");
        elem.appendChild(excludes);
      }
      if (filesetEntry.getFiles() != null) {
        Element files = createValueElement(doc, LABEL_LIST, filesetEntry.getFiles());
        files.setAttribute("name", "files");
        elem.appendChild(files);
      }
    } else { // INTEGER STRING LABEL DISTRIBUTION OUTPUT
      elem = createSingleValueElement(doc, type.toString(), hasMultipleValues);
      if (!hasMultipleValues && !Iterables.isEmpty(values)) {
        Object value = Iterables.getOnlyElement(values);
        // Values such as those of attribute "linkstamp" may be null.
        if (value != null) {
          try {
            elem.setAttribute("value", value.toString());
          } catch (DOMException e) {
            elem.setAttribute("value", "[[[ERROR: could not be encoded as XML]]]");
          }
        }
      }
    }
    return elem;
  }

  private static Element createValueElement(Document doc, Type<?> type, Object value) {
    return createValueElement(doc, type, ImmutableList.of(value));
  }

  /**
   * Creates the given DOM element, adding <code>configurable="yes"</code> if it represents
   * a configurable single-value attribute (configurable list attributes simply have their
   * lists merged into an aggregate flat list).
   */
  private static Element createSingleValueElement(Document doc, String name,
      boolean configurable) {
    Element elem = doc.createElement(name);
    if (configurable) {
      elem.setAttribute("configurable", "yes");
    }
    return elem;
  }
}
