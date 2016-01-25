// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android;

import static com.android.SdkConstants.ANDROID_STYLE_RESOURCE_PREFIX;
import static com.android.SdkConstants.ANDROID_URI;
import static com.android.SdkConstants.ATTR_NAME;
import static com.android.SdkConstants.ATTR_PARENT;
import static com.android.SdkConstants.ATTR_TYPE;
import static com.android.SdkConstants.DOT_CLASS;
import static com.android.SdkConstants.DOT_GIF;
import static com.android.SdkConstants.DOT_JPEG;
import static com.android.SdkConstants.DOT_JPG;
import static com.android.SdkConstants.DOT_PNG;
import static com.android.SdkConstants.DOT_XML;
import static com.android.SdkConstants.FD_RES_VALUES;
import static com.android.SdkConstants.PREFIX_ANDROID;
import static com.android.SdkConstants.STYLE_RESOURCE_PREFIX;
import static com.android.SdkConstants.TAG_ITEM;
import static com.android.SdkConstants.TAG_RESOURCES;
import static com.android.SdkConstants.TAG_STYLE;
import static com.android.utils.SdkUtils.endsWith;
import static com.android.utils.SdkUtils.endsWithIgnoreCase;
import static com.google.common.base.Charsets.UTF_8;

import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.io.ByteStreams;
import com.google.common.io.Closeables;
import com.google.common.io.Files;

import com.android.annotations.NonNull;
import com.android.annotations.Nullable;
import com.android.annotations.VisibleForTesting;
import com.android.ide.common.resources.ResourceUrl;
import com.android.ide.common.resources.configuration.DensityQualifier;
import com.android.ide.common.resources.configuration.FolderConfiguration;
import com.android.ide.common.resources.configuration.ResourceQualifier;
import com.android.ide.common.xml.XmlPrettyPrinter;
import com.android.resources.FolderTypeRelationship;
import com.android.resources.ResourceFolderType;
import com.android.resources.ResourceType;
import com.android.utils.XmlUtils;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import javax.xml.parsers.ParserConfigurationException;

/**
 * Class responsible for searching through a Gradle built tree (after resource merging, compilation
 * and ProGuarding has been completed, but before final .apk assembly), which figures out which
 * resources if any are unused, and removes them. <p> It does this by examining <ul> <li>The merged
 * manifest, to find root resource references (such as drawables used for activity icons)</li>
 * <li>The merged R class (to find the actual integer constants assigned to resources)</li> <li>The
 * ProGuard log files (to find the mapping from original symbol names to short names)</li>* <li>The
 * merged resources (to find which resources reference other resources, e.g. drawable state lists
 * including other drawables, or layouts including other layouts, or styles referencing other
 * drawables, or menus items including action layouts, etc.)</li> <li>The ProGuard output classes
 * (to find resource references in code that are actually reachable)</li> </ul> From all this, it
 * builds up a reference graph, and based on the root references (e.g. from the manifest and from
 * the remaining code) it computes which resources are actually reachable in the app, and anything
 * that is not reachable is then marked for deletion. <p> A resource is referenced in code if either
 * the field R.type.name is referenced (which is the case for non-final resource references, e.g. in
 * libraries), or if the corresponding int value is referenced (for final resource values). We check
 * this by looking at the ProGuard output classes with an ASM visitor. One complication is that code
 * can also call {@code Resources#getIdentifier(String,String,String)} where they can pass in the
 * names of resources to look up. To handle this scenario, we use the ClassVisitor to see if there
 * are any calls to the specific {@code Resources#getIdentifier} method. If not, great, the usage
 * analysis is completely accurate. If we <b>do</b> find one, we check <b>all</b> the string
 * constants found anywhere in the app, and look to see if any look relevant. For example, if we
 * find the string "string/foo" or "my.pkg:string/foo", we will then mark the string resource named
 * foo (if any) as potentially used. Similarly, if we find just "foo" or "/foo", we will mark
 * <b>all</b> resources named "foo" as potentially used. However, if the string is "bar/foo" or "
 * foo " these strings are ignored. This means we can potentially miss resources usages where the
 * resource name is completed computed (e.g. by concatenating individual characters or taking
 * substrings of strings that do not look like resource names), but that seems extremely unlikely to
 * be a real-world scenario. <p> For now, for reasons detailed in the code, this only applies to
 * file-based resources like layouts, menus and drawables, not value-based resources like strings
 * and dimensions.
 */
public class ResourceShrinker {

  private static final Logger logger = Logger.getLogger(ResourceShrinker.class.getName());

  public static final int TYPICAL_RESOURCE_COUNT = 200;
  private final List<String> resourcePackages;
  private final Path rTxt;
  private final Path classesJar;
  private final Path mergedManifest;
  private final Path mergedResourceDir;

  /**
   * The computed set of unused resources
   */
  private List<Resource> unused;
  /**
   * List of all known resources (parsed from R.java)
   */
  private List<Resource> resources = Lists.newArrayListWithExpectedSize(TYPICAL_RESOURCE_COUNT);
  /**
   * Map from R field value to corresponding resource
   */
  private Map<Integer, Resource> valueToResource =
      Maps.newHashMapWithExpectedSize(TYPICAL_RESOURCE_COUNT);
  /**
   * Map from resource type to map from resource name to resource object
   */
  private Map<ResourceType, Map<String, Resource>> typeToName =
      Maps.newEnumMap(ResourceType.class);
  /**
   * Map from resource class owners (VM format class) to corresponding resource types. This will
   * typically be the fully qualified names of the R classes, as well as any renamed versions of
   * those discovered in the mapping.txt file from ProGuard
   */
  private Map<String, ResourceType> resourceClassOwners = Maps.newHashMapWithExpectedSize(20);

  public ResourceShrinker(
      List<String> resourcePackages,
      @NonNull Path rTxt,
      @NonNull Path classesJar,
      @NonNull Path manifest,
      @NonNull Path resources) {
    this.resourcePackages = resourcePackages;
    this.rTxt = rTxt;
    this.classesJar = classesJar;
    this.mergedManifest = manifest;
    this.mergedResourceDir = resources;
  }

  public void shrink(Path destinationDir) throws IOException,
      ParserConfigurationException, SAXException {
    parseResourceTxtFile(rTxt, resourcePackages);
    recordUsages(classesJar);
    recordManifestUsages(mergedManifest);
    recordResources(mergedResourceDir);
    keepPossiblyReferencedResources();
    dumpReferences();
    findUnused();
    removeUnused(destinationDir);
  }

  /**
   * Remove resources (already identified by {@link #shrink(Path)}).
   *
   * <p>This task will copy all remaining used resources over from the full resource directory to a
   * new reduced resource directory and removes unused values from all value xml files.
   *
   * @param destination directory to copy resources into; if null, delete resources in place
   */
  private void removeUnused(Path destination) throws IOException,
      ParserConfigurationException, SAXException {
    assert unused != null; // should always call analyze() first
    int resourceCount = unused.size() * 4; // *4: account for some resource folder repetition
    Set<File> skip = Sets.newHashSetWithExpectedSize(resourceCount);
    Set<File> rewrite = Sets.newHashSetWithExpectedSize(resourceCount);
    for (Resource resource : unused) {
      if (resource.declarations != null) {
        for (File file : resource.declarations) {
          String folder = file.getParentFile().getName();
          ResourceFolderType folderType = ResourceFolderType.getFolderType(folder);
          if (folderType != null && folderType != ResourceFolderType.VALUES) {
            logger.info("Deleted unused resource " + file);
            assert skip != null;
            skip.add(file);
          } else {
            // Can't delete values immediately; there can be many resources
            // in this file, so we have to process them all
            rewrite.add(file);
          }
        }
      }
    }
    // Special case the base values.xml folder
    File values = new File(mergedResourceDir.toFile(),
        FD_RES_VALUES + File.separatorChar + "values.xml");
    boolean valuesExists = values.exists();
    if (valuesExists) {
      rewrite.add(values);
    }
    Map<File, String> rewritten = Maps.newHashMapWithExpectedSize(rewrite.size());
    // Delete value resources: Must rewrite the XML files
    for (File file : rewrite) {
      String xml = Files.toString(file, UTF_8);
      Document document = XmlUtils.parseDocument(xml, true);
      Element root = document.getDocumentElement();
      if (root != null && TAG_RESOURCES.equals(root.getTagName())) {
        List<String> removed = Lists.newArrayList();
        stripUnused(root, removed);
        logger.info("Removed " + removed.size() + " unused resources from " + file + ":\n  "
            + Joiner.on(", ").join(removed));
        String formatted = XmlPrettyPrinter.prettyPrint(document, xml.endsWith("\n"));
        rewritten.put(file, formatted);
      }
    }
    filteredCopy(mergedResourceDir.toFile(), destination, skip, rewritten);
  }

  /**
   * Copies one resource directory tree into another; skipping some files, replacing the contents of
   * some, and passing everything else through unmodified
   */
  private static void filteredCopy(File source, Path destination, Set<File> skip,
      Map<File, String> replace) throws IOException {

    File destinationFile = destination.toFile();
    if (source.isDirectory()) {
      File[] children = source.listFiles();
      if (children != null) {
        if (!destinationFile.exists()) {
          boolean success = destinationFile.mkdirs();
          if (!success) {
            throw new IOException("Could not create " + destination);
          }
        }
        for (File child : children) {
          filteredCopy(child, destination.resolve(child.getName()), skip, replace);
        }
      }
    } else if (!skip.contains(source) && source.isFile()) {
      String contents = replace.get(source);
      if (contents != null) {
        Files.write(contents, destinationFile, Charsets.UTF_8);
      } else {
        Files.copy(source, destinationFile);
      }
    }
  }

  private void stripUnused(Element element, List<String> removed) {
    ResourceType type = getResourceType(element);
    if (type == ResourceType.ATTR) {
      // Not yet properly handled
      return;
    }
    Resource resource = getResource(element);
    if (resource != null) {
      if (resource.type == ResourceType.DECLARE_STYLEABLE
          || resource.type == ResourceType.ATTR) {
        // Don't strip children of declare-styleable; we're not correctly
        // tracking field references of the R_styleable_attr fields yet
        return;
      }
      if (!resource.reachable
          && (resource.type == ResourceType.STYLE
              || resource.type == ResourceType.PLURALS
              || resource.type == ResourceType.ARRAY)) {
        NodeList children = element.getChildNodes();
        for (int i = children.getLength() - 1; i >= 0; i--) {
          Node child = children.item(i);
          element.removeChild(child);
        }
        return;
      }
    }
    NodeList children = element.getChildNodes();
    for (int i = children.getLength() - 1; i >= 0; i--) {
      Node child = children.item(i);
      if (child.getNodeType() == Node.ELEMENT_NODE) {
        stripUnused((Element) child, removed);
      }
    }
    if (resource != null && !resource.reachable) {
      removed.add(resource.getUrl());
      // for themes etc where .'s have been replaced by _'s
      String name = element.getAttribute(ATTR_NAME);
      if (name.isEmpty()) {
        name = resource.name;
      }
      Node nextSibling = element.getNextSibling();
      Node parent = element.getParentNode();
      NodeList oldChildren = element.getChildNodes();
      parent.removeChild(element);
      Document document = element.getOwnerDocument();
      element = document.createElement("item");
      for (int i = 0; i < oldChildren.getLength(); i++) {
        element.appendChild(oldChildren.item(i));
      }
      element.setAttribute(ATTR_NAME, name);
      element.setAttribute(ATTR_TYPE, resource.type.getName());
      String text = null;
      switch (resource.type) {
        case BOOL:
          text = "true";
          break;
        case DIMEN:
          text = "0dp";
          break;
        case INTEGER:
          text = "0";
          break;
      }
      element.setTextContent(text);
      parent.insertBefore(element, nextSibling);
    }
  }

  private static String getFieldName(Element element) {
    return getFieldName(element.getAttribute(ATTR_NAME));
  }

  @Nullable
  private Resource getResource(Element element) {
    ResourceType type = getResourceType(element);
    if (type != null) {
      String name = getFieldName(element);
      return getResource(type, name);
    }
    return null;
  }

  private static ResourceType getResourceType(Element element) {
    String tagName = element.getTagName();
    switch (tagName) {
      case TAG_ITEM:
        String typeName = element.getAttribute(ATTR_TYPE);
        if (!typeName.isEmpty()) {
          return ResourceType.getEnum(typeName);
        }
        break;
      case "string-array":
      case "integer-array":
        return ResourceType.ARRAY;
      default:
        return ResourceType.getEnum(tagName);
    }
    return null;
  }

  private void findUnused() {
    List<Resource> roots = Lists.newArrayList();
    for (Resource resource : resources) {
      if (resource.reachable && resource.type != ResourceType.ID
          && resource.type != ResourceType.ATTR) {
        roots.add(resource);
      }
    }
    logger.fine(String.format("The root reachable resources are: %s",
        Joiner.on(",\n   ").join(roots)));
    Map<Resource, Boolean> seen = new IdentityHashMap<>(resources.size());
    for (Resource root : roots) {
      visit(root, seen);
    }
    List<Resource> unused = Lists.newArrayListWithExpectedSize(resources.size());
    for (Resource resource : resources) {
      if (!resource.reachable && resource.isRelevantType()) {
        unused.add(resource);
      }
    }
    this.unused = unused;
  }

  private static void visit(Resource root, Map<Resource, Boolean> seen) {
    if (seen.containsKey(root)) {
      return;
    }
    seen.put(root, Boolean.TRUE);
    root.reachable = true;
    if (root.references != null) {
      for (Resource referenced : root.references) {
        visit(referenced, seen);
      }
    }
  }

  private void dumpReferences() {
    for (Resource resource : resources) {
      if (resource.references != null) {
        logger.info(resource + " => " + resource.references);
      }
    }
  }

  private void keepPossiblyReferencedResources() {
    if (!mFoundGetIdentifier || mStrings == null) {
      // No calls to android.content.res.Resources#getIdentifier; no need
      // to worry about string references to resources
      return;
    }
    List<String> strings = new ArrayList<String>(mStrings);
    Collections.sort(strings);
    logger.fine(String.format("android.content.res.Resources#getIdentifier present: %s",
        mFoundGetIdentifier));
    logger.fine("Referenced Strings:");
    for (String s : strings) {
      s = s.trim().replace("\n", "\\n");
      if (s.length() > 40) {
        s = s.substring(0, 37) + "...";
      } else if (s.isEmpty()) {
        continue;
      }
      logger.fine("  " + s);
    }

    Set<String> names = Sets.newHashSetWithExpectedSize(50);
    for (Map<String, Resource> map : typeToName.values()) {
      names.addAll(map.keySet());
    }
    for (String string : mStrings) {
      // Check whether the string looks relevant
      // We consider three types of strings:
      //  (1) simple resource names, e.g. "foo" from @layout/foo
      //      These might be the parameter to a getIdentifier() call, or could
      //      be composed into a fully qualified resource name for the getIdentifier()
      //      method. We match these for *all* resource types.
      //  (2) Relative source names, e.g. layout/foo, from @layout/foo
      //      These might be composed into a fully qualified resource name for
      //      getIdentifier().
      //  (3) Fully qualified resource names of the form package:type/name.
      int n = string.length();
      boolean justName = true;
      boolean haveSlash = false;
      for (int i = 0; i < n; i++) {
        char c = string.charAt(i);
        if (c == '/') {
          haveSlash = true;
          justName = false;
        } else if (c == '.' || c == ':') {
          justName = false;
        } else if (!Character.isJavaIdentifierPart(c)) {
          // This shouldn't happen; we've filtered out these strings in
          // the {@link #referencedString} method
          assert false : string;
          break;
        }
      }
      String name;
      if (justName) {
        // Check name (below)
        name = string;
      } else if (!haveSlash) {
        // If we have more than just a symbol name, we expect to also see a slash
        //noinspection UnnecessaryContinue
        continue;
      } else {
        // Try to pick out the resource name pieces; if we can find the
        // resource type unambiguously; if not, just match on names
        int slash = string.indexOf('/');
        assert slash != -1; // checked with haveSlash above
        name = string.substring(slash + 1);
        if (name.isEmpty() || !names.contains(name)) {
          continue;
        }
        // See if have a known specific resource type
        if (slash > 0) {
          int colon = string.indexOf(':');
          String typeName = string.substring(colon != -1 ? colon + 1 : 0, slash);
          ResourceType type = ResourceType.getEnum(typeName);
          if (type == null) {
            continue;
          }
          Resource resource = getResource(type, name);
          if (resource != null) {
            logger.fine("Marking " + resource + " used because it "
                + "matches string pool constant " + string);
          }
          markReachable(resource);
          continue;
        }
        // fall through and check the name
      }
      if (names.contains(name)) {
        for (Map<String, Resource> map : typeToName.values()) {
          Resource resource = map.get(string);
          if (resource != null) {
            logger.fine("Marking " + resource + " used because it "
                + "matches string pool constant " + string);
          }
          markReachable(resource);
        }
      } else if (Character.isDigit(name.charAt(0))) {
        // Just a number? There are cases where it calls getIdentifier by
        // a String number; see for example SuggestionsAdapter in the support
        // library which reports supporting a string like "2130837524" and
        // "android.resource://com.android.alarmclock/2130837524".
        try {
          int id = Integer.parseInt(name);
          if (id != 0) {
            markReachable(valueToResource.get(id));
          }
        } catch (NumberFormatException e) {
          // pass
        }
      }
    }
  }

  private void recordResources(Path resDir)
      throws IOException, SAXException, ParserConfigurationException {

    File[] resourceFolders = resDir.toFile().listFiles();
    if (resourceFolders != null) {
      for (File folder : resourceFolders) {
        ResourceFolderType folderType = ResourceFolderType.getFolderType(folder.getName());
        if (folderType != null) {
          recordResources(folderType, folder);
        }
      }
    }
  }

  private void recordResources(@NonNull ResourceFolderType folderType, File folder)
      throws ParserConfigurationException, SAXException, IOException {
    File[] files = folder.listFiles();
    FolderConfiguration config = FolderConfiguration.getConfigForFolder(folder.getName());
    boolean isDefaultFolder = false;
    if (config != null) {
      isDefaultFolder = true;
      for (int i = 0, n = FolderConfiguration.getQualifierCount(); i < n; i++) {
        ResourceQualifier qualifier = config.getQualifier(i);
        // Densities are special: even if they're present in just (say) drawable-hdpi
        // we'll match it on any other density
        if (qualifier != null && !(qualifier instanceof DensityQualifier)) {
          isDefaultFolder = false;
          break;
        }
      }
    }
    if (files != null) {
      for (File file : files) {
        String path = file.getPath();
        boolean isXml = endsWithIgnoreCase(path, DOT_XML);
        Resource from = null;
        // Record resource for the whole file
        if (folderType != ResourceFolderType.VALUES
            && (isXml
            || endsWith(path, DOT_PNG) //also true for endsWith(name, DOT_9PNG)
            || endsWith(path, DOT_JPG)
            || endsWith(path, DOT_GIF)
            || endsWith(path, DOT_JPEG))) {
          List<ResourceType> types = FolderTypeRelationship.getRelatedResourceTypes(
              folderType);
          ResourceType type = types.get(0);
          assert type != ResourceType.ID : folderType;
          String name = file.getName();
          name = name.substring(0, name.indexOf('.'));
          Resource resource = getResource(type, name);
          if (resource != null) {
            resource.addLocation(file);
            if (isDefaultFolder) {
              resource.hasDefault = true;
            }
            from = resource;
          }
        }
        if (isXml) {
          // For value files, and drawables and colors etc also pull in resource
          // references inside the file
          recordResourcesUsages(file, isDefaultFolder, from);
        }
      }
    }
  }

  private void recordManifestUsages(Path manifest)
      throws IOException, ParserConfigurationException, SAXException {
    String xml = Files.toString(manifest.toFile(), UTF_8);
    Document document = XmlUtils.parseDocument(xml, true);
    recordManifestUsages(document.getDocumentElement());
  }

  private void recordResourcesUsages(@NonNull File file, boolean isDefaultFolder,
      @Nullable Resource from)
      throws IOException, ParserConfigurationException, SAXException {
    String xml = Files.toString(file, UTF_8);
    Document document = XmlUtils.parseDocument(xml, true);
    recordResourceReferences(file, isDefaultFolder, document.getDocumentElement(), from);
  }

  @Nullable
  private Resource getResource(@NonNull ResourceType type, @NonNull String name) {
    Map<String, Resource> nameMap = typeToName.get(type);
    if (nameMap != null) {
      return nameMap.get(getFieldName(name));
    }
    return null;
  }

  @Nullable
  private Resource getResource(@NonNull String possibleUrlReference) {
    ResourceUrl url = ResourceUrl.parse(possibleUrlReference);
    if (url != null && !url.framework) {
      return getResource(url.type, url.name);
    }
    return null;
  }

  private void recordManifestUsages(Node node) {
    short nodeType = node.getNodeType();
    if (nodeType == Node.ELEMENT_NODE) {
      Element element = (Element) node;
      NamedNodeMap attributes = element.getAttributes();
      for (int i = 0, n = attributes.getLength(); i < n; i++) {
        Attr attr = (Attr) attributes.item(i);
        markReachable(getResource(attr.getValue()));
      }
    } else if (nodeType == Node.TEXT_NODE) {
      // Does this apply to any manifests??
      String text = node.getNodeValue().trim();
      markReachable(getResource(text));
    }
    NodeList children = node.getChildNodes();
    for (int i = 0, n = children.getLength(); i < n; i++) {
      Node child = children.item(i);
      recordManifestUsages(child);
    }
  }

  private void recordResourceReferences(@NonNull File file, boolean isDefaultFolder,
      @NonNull Node node, @Nullable Resource from) {
    short nodeType = node.getNodeType();
    if (nodeType == Node.ELEMENT_NODE) {
      Element element = (Element) node;
      if (from != null) {
        NamedNodeMap attributes = element.getAttributes();
        for (int i = 0, n = attributes.getLength(); i < n; i++) {
          Attr attr = (Attr) attributes.item(i);
          Resource resource = getResource(attr.getValue());
          if (resource != null) {
            from.addReference(resource);
          }
        }
        // Android Wear. We *could* limit ourselves to only doing this in files
        // referenced from a manifest meta-data element, e.g.
        // <meta-data android:name="com.google.android.wearable.beta.app"
        //    android:resource="@xml/wearable_app_desc"/>
        // but given that that property has "beta" in the name, it seems likely
        // to change and therefore hardcoding it for that key risks breakage
        // in the future.
        if ("rawPathResId".equals(element.getTagName())) {
          StringBuilder sb = new StringBuilder();
          NodeList children = node.getChildNodes();
          for (int i = 0, n = children.getLength(); i < n; i++) {
            Node child = children.item(i);
            if (child.getNodeType() == Element.TEXT_NODE
                || child.getNodeType() == Element.CDATA_SECTION_NODE) {
              sb.append(child.getNodeValue());
            }
          }
          if (sb.length() > 0) {
            Resource resource = getResource(ResourceType.RAW, sb.toString().trim());
            from.addReference(resource);
          }
        }
      }
      Resource definition = getResource(element);
      if (definition != null) {
        from = definition;
        definition.addLocation(file);
        if (isDefaultFolder) {
          definition.hasDefault = true;
        }
      }
      String tagName = element.getTagName();
      if (TAG_STYLE.equals(tagName)) {
        if (element.hasAttribute(ATTR_PARENT)) {
          String parent = element.getAttribute(ATTR_PARENT);
          if (!parent.isEmpty() && !parent.startsWith(ANDROID_STYLE_RESOURCE_PREFIX)
              && !parent.startsWith(PREFIX_ANDROID)) {
            String parentStyle = parent;
            if (!parentStyle.startsWith(STYLE_RESOURCE_PREFIX)) {
              parentStyle = STYLE_RESOURCE_PREFIX + parentStyle;
            }
            Resource ps = getResource(getFieldName(parentStyle));
            if (ps != null && definition != null) {
              definition.addReference(ps);
            }
          }
        } else {
          // Implicit parent styles by name
          String name = getFieldName(element);
          while (true) {
            int index = name.lastIndexOf('_');
            if (index != -1) {
              name = name.substring(0, index);
              Resource ps = getResource(STYLE_RESOURCE_PREFIX + getFieldName(name));
              if (ps != null && definition != null) {
                definition.addReference(ps);
              }
            } else {
              break;
            }
          }
        }
      }
      if (TAG_ITEM.equals(tagName)) {
        // In style? If so the name: attribute can be a reference
        if (element.getParentNode() != null
            && element.getParentNode().getNodeName().equals(TAG_STYLE)) {
          String name = element.getAttributeNS(ANDROID_URI, ATTR_NAME);
          if (!name.isEmpty() && !name.startsWith("android:")) {
            Resource resource = getResource(ResourceType.ATTR, name);
            if (definition == null) {
              Element style = (Element) element.getParentNode();
              definition = getResource(style);
              if (definition != null) {
                from = definition;
                definition.addReference(resource);
              }
            }
          }
        }
      }
    } else if (nodeType == Node.TEXT_NODE || nodeType == Node.CDATA_SECTION_NODE) {
      String text = node.getNodeValue().trim();
      Resource textResource = getResource(getFieldName(text));
      if (textResource != null && from != null) {
        from.addReference(textResource);
      }
    }
    NodeList children = node.getChildNodes();
    for (int i = 0, n = children.getLength(); i < n; i++) {
      Node child = children.item(i);
      recordResourceReferences(file, isDefaultFolder, child, from);
    }
  }

  public static String getFieldName(@NonNull String styleName) {
    return styleName.replace('.', '_').replace('-', '_').replace(':', '_');
  }

  private static void markReachable(@Nullable Resource resource) {
    if (resource != null) {
      resource.reachable = true;
    }
  }

  private Set<String> mStrings;
  private boolean mFoundGetIdentifier;

  private void referencedString(@NonNull String string) {
    // See if the string is at all eligible; ignore strings that aren't
    // identifiers (has java identifier chars and nothing but .:/), or are empty or too long
    if (string.isEmpty() || string.length() > 80) {
      return;
    }
    boolean haveIdentifierChar = false;
    for (int i = 0, n = string.length(); i < n; i++) {
      char c = string.charAt(i);
      boolean identifierChar = Character.isJavaIdentifierPart(c);
      if (!identifierChar && c != '.' && c != ':' && c != '/') {
        // .:/ are for the fully qualified resuorce names
        return;
      } else if (identifierChar) {
        haveIdentifierChar = true;
      }
    }
    if (!haveIdentifierChar) {
      return;
    }
    if (mStrings == null) {
      mStrings = Sets.newHashSetWithExpectedSize(300);
    }
    mStrings.add(string);
  }

  private void recordUsages(Path jarFile) throws IOException {
    if (!jarFile.toFile().exists()) {
      return;
    }
    ZipInputStream zis = null;
    try {
      FileInputStream fis = new FileInputStream(jarFile.toFile());
      try {
        zis = new ZipInputStream(fis);
        ZipEntry entry = zis.getNextEntry();
        while (entry != null) {
          String name = entry.getName();
          if (name.endsWith(DOT_CLASS)) {
            byte[] bytes = ByteStreams.toByteArray(zis);
            if (bytes != null) {
              ClassReader classReader = new ClassReader(bytes);
              classReader.accept(new UsageVisitor(), 0);
            }
          }
          entry = zis.getNextEntry();
        }
      } finally {
        Closeables.close(fis, true);
      }
    } finally {
      Closeables.close(zis, true);
    }
  }

  private void parseResourceTxtFile(Path rTxt, List<String> resourcePackages) throws IOException {
    BufferedReader reader = java.nio.file.Files.newBufferedReader(rTxt, Charset.defaultCharset());
    String line;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.split(" ");
      ResourceType type = ResourceType.getEnum(tokens[1]);
      for (String resourcePackage : resourcePackages) {
        resourceClassOwners.put(resourcePackage.replace('.', '/') + "/R$" + type.getName(), type);
      }
      if (type == ResourceType.STYLEABLE) {
        if (tokens[0].equals("int[]")) {
          addResource(ResourceType.DECLARE_STYLEABLE, tokens[2], null);
        } else {
          // TODO(jongerrish): Implement stripping of styleables.
        }
      } else {
        addResource(type, tokens[2], tokens[3]);
      }
    }
  }

  private void addResource(@NonNull ResourceType type, @NonNull String name,
      @Nullable String value) {
    int realValue = value != null ? Integer.decode(value) : -1;
    Resource resource = getResource(type, name);
    if (resource != null) {
      //noinspection VariableNotUsedInsideIf
      if (value != null) {
        if (resource.value == -1) {
          resource.value = realValue;
        } else {
          assert realValue == resource.value;
        }
      }
      return;
    }
    resource = new Resource(type, name, realValue);
    resources.add(resource);
    if (realValue != -1) {
      valueToResource.put(realValue, resource);
    }
    Map<String, Resource> nameMap = typeToName.get(type);
    if (nameMap == null) {
      nameMap = Maps.newHashMapWithExpectedSize(30);
      typeToName.put(type, nameMap);
    }
    nameMap.put(name, resource);
    // TODO: Assert that we don't set the same resource multiple times to different values.
    // Could happen if you pass in stale data!
  }

  @VisibleForTesting
  List<Resource> getAllResources() {
    return resources;
  }

  /**
   * Metadata about an Android resource
   */
  public static class Resource {

    /**
     * Type of resource
     */
    public ResourceType type;
    /**
     * Name of resource
     */
    public String name;
    /**
     * Integer id location
     */
    public int value;
    /**
     * Whether this resource can be reached from one of the roots (manifest, code)
     */
    public boolean reachable;
    /**
     * Whether this resource has a default definition (e.g. present in a resource folder with no
     * qualifiers). For id references, an inline definition (@+id) does not count as a default
     * definition.
     */
    public boolean hasDefault;
    /**
     * Resources this resource references. For example, a layout can reference another via an
     * include; a style reference in a layout references that layout style, and so on.
     */
    public List<Resource> references;
    public final List<File> declarations = Lists.newArrayList();

    private Resource(ResourceType type, String name, int value) {
      this.type = type;
      this.name = name;
      this.value = value;
    }

    @Override
    public String toString() {
      return type + ":" + name + ":" + value;
    }

    @SuppressWarnings("RedundantIfStatement") // Generated by IDE
    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      Resource resource = (Resource) o;
      if (name != null ? !name.equals(resource.name) : resource.name != null) {
        return false;
      }
      if (type != resource.type) {
        return false;
      }
      return true;
    }

    @Override
    public int hashCode() {
      int result = type != null ? type.hashCode() : 0;
      result = 31 * result + (name != null ? name.hashCode() : 0);
      return result;
    }

    public void addLocation(@NonNull File file) {
      declarations.add(file);
    }

    public void addReference(@Nullable Resource resource) {
      if (resource != null) {
        if (references == null) {
          references = Lists.newArrayList();
        } else if (references.contains(resource)) {
          return;
        }
        references.add(resource);
      }
    }

    public String getUrl() {
      return '@' + type.getName() + '/' + name;
    }

    public boolean isRelevantType() {
      return type != ResourceType.ID; // && getFolderType() != ResourceFolderType.VALUES;
    }
  }

  private class UsageVisitor extends ClassVisitor {

    public UsageVisitor() {
      super(Opcodes.ASM4);
    }

    @Override
    public MethodVisitor visitMethod(int access, final String name,
        String desc, String signature, String[] exceptions) {
      return new MethodVisitor(Opcodes.ASM4) {
        @Override
        public void visitLdcInsn(Object cst) {
          if (cst instanceof Integer) {
            Integer value = (Integer) cst;
            markReachable(valueToResource.get(value));
          } else if (cst instanceof String) {
            String string = (String) cst;
            referencedString(string);
          }
        }

        @Override
        public void visitFieldInsn(int opcode, String owner, String name, String desc) {
          if (opcode == Opcodes.GETSTATIC) {
            ResourceType type = resourceClassOwners.get(owner);
            if (type != null) {
              Resource resource = getResource(type, name);
              if (resource != null) {
                markReachable(resource);
              }
            }
          }
        }

        @Override
        public void visitMethodInsn(int opcode, String owner, String name, String desc) {
          super.visitMethodInsn(opcode, owner, name, desc);
          if (owner.equals("android/content/res/Resources")
              && name.equals("getIdentifier")
              && desc.equals(
              "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)I")) {
            mFoundGetIdentifier = true;
            // TODO: Check previous instruction and see if we can find a literal
            // String; if so, we can more accurately dispatch the resource here
            // rather than having to check the whole string pool!
          }
        }
      };
    }
  }
}
