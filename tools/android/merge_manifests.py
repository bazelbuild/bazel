# Copyright 2015 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merges two android manifest xml files."""

import re
import sys
import xml.dom.minidom

from tools.android import android_permissions
from third_party.py import gflags

FLAGS = gflags.FLAGS
EXCLUDE_ALL_ARG = 'all'

gflags.DEFINE_multistring(
    'exclude_permission', None,
    'Permissions to be excluded, e.g.: "android.permission.READ_LOGS".'
    'This is a multistring, so multiple of those flags can be provided.'
    'Pass "%s" to exclude all permissions contributed by mergees.'
    % EXCLUDE_ALL_ARG)
gflags.DEFINE_multistring(
    'mergee', None,
    'Mergee manifest that will be merged to merger manifest.'
    'This is a multistring, so multiple of those flags can be provided.')
gflags.DEFINE_string('merger', None,
                     'Merger AndroidManifest file to be merged.')
gflags.DEFINE_string('output', None, 'Output file with merged manifests.')

USAGE = """Error, invalid arguments.
Usage: merge_manifests.py --merger=<merger> --mergee=<mergee1> --mergee=<merge2>
    --exclude_permission=[Exclude permissions from mergee] --output=<output>
Examples:
  merge_manifests.py --merger=manifest.xml --mergee=manifest2.xml
    --mergee=manifest3.xml --exclude_permission=android.permission.READ_LOGS
    --output=AndroidManifest.xml

    merge_manifests.py --merger=manifest.xml --mergee=manifest2.xml
    --mergee=manifest3.xml --exclude_permission=%s
    --output=AndroidManifest.xml
""" % EXCLUDE_ALL_ARG


class UndefinedPlaceholderException(Exception):
  """Exception thrown when encountering a placeholder without a replacement.
  """
  pass


class MalformedManifestException(Exception):
  """Exception thrown when encountering a fatally malformed manifest.
  """
  pass


class MergeManifests(object):
  """A utility class for merging two android manifest.xml files.

  This is useful when including another app as android library.
  """
  _ACTIVITY = 'activity'
  _ANDROID_NAME = 'android:name'
  _ANDROID_LABEL = 'android:label'
  _INTENT_FILTER = 'intent-filter'
  _MANIFEST = 'manifest'
  _USES_PERMISSION = 'uses-permission'
  _USES_PERMISSION_SDK_23 = 'uses-permission-sdk-23'
  _NODES_TO_COPY_FROM_MERGEE = {
      _MANIFEST: [
          'instrumentation',
          'permission',
          _USES_PERMISSION,
          _USES_PERMISSION_SDK_23,
          'uses-feature',
          'permission-group',
          ],
      'application': [
          'activity',
          'activity-alias',
          'provider',
          'receiver',
          'service',
          'uses-library',
          'meta-data',
          ],
  }
  _NODES_TO_REMOVE_FROM_MERGER = []
  _PACKAGE = 'package'

  def __init__(self, merger, mergees, exclude_permissions=None):
    """Constructs and initializes the MergeManifests object.

    Args:
      merger: First (merger) AndroidManifest.xml string.
      mergees: mergee AndroidManifest.xml strings, a list.
      exclude_permissions: Permissions to be excludeed from merging,
        string list. "all" means don't include any permissions.
    """
    self._merger = merger
    self._mergees = mergees
    self._exclude_permissions = exclude_permissions
    self._merger_dom = xml.dom.minidom.parseString(self._merger[0])

  def _ApplyExcludePermissions(self, dom):
    """Apply exclude filters.

    Args:
      dom: Document dom object from which to exclude permissions.
    """
    if self._exclude_permissions:
      exclude_all_permissions = EXCLUDE_ALL_ARG in self._exclude_permissions
      for element in (dom.getElementsByTagName(self._USES_PERMISSION) +
       dom.getElementsByTagName(self._USES_PERMISSION_SDK_23)):
        if element.hasAttribute(self._ANDROID_NAME):
          attrib = element.getAttribute(self._ANDROID_NAME)
          if exclude_all_permissions or attrib in self._exclude_permissions:
            element.parentNode.removeChild(element)

  def _ExpandPackageName(self, node):
    """Set the package name if it is in a short form.

    Filtering logic for what elements have package expansion:
    If the name starts with a dot, always prefix it with the package.
    If the name has a dot anywhere else, do not prefix it.
    If the name has no dot at all, also prefix it with the package.

    The massageManifest function shows where this rule is applied:

    In the application element, on the name and backupAgent attributes.
    In the activity, service, receiver, provider, and activity-alias elements,
    on the name attribute.
    In the activity-alias element, on the targetActivity attribute.

    Args:
      node: Xml Node for which to expand package name.
    """
    package_name = node.getElementsByTagName(self._MANIFEST).item(
        0).getAttribute(self._PACKAGE)

    if not package_name:
      return

    for element in node.getElementsByTagName('*'):
      if element.nodeName not in [
          'activity',
          'activity-alias',
          'application',
          'service',
          'receiver',
          'provider',
          ]:
        continue

      self._ExpandPackageNameHelper(package_name, element, self._ANDROID_NAME)

      if element.nodeName == 'activity':
        self._ExpandPackageNameHelper(package_name, element,
                                      'android:parentActivityName')

      if element.nodeName == 'activity-alias':
        self._ExpandPackageNameHelper(package_name, element,
                                      'android:targetActivity')
        continue

      if element.nodeName == 'application':
        self._ExpandPackageNameHelper(package_name, element,
                                      'android:backupAgent')

  def _ExpandPackageNameHelper(self, package_name, element, attribute_name):
    if element.hasAttribute(attribute_name):
      class_name = element.getAttribute(attribute_name)

      if class_name.startswith('.'):
        pass
      elif '.' not in class_name:
        class_name = '.' + class_name
      else:
        return

      element.setAttribute(attribute_name, package_name + class_name)

  def _RemoveFromMerger(self):
    """Remove from merger."""
    for tag_name in self._NODES_TO_REMOVE_FROM_MERGER:
      elements = self._merger_dom.getElementsByTagName(tag_name)
      for element in elements:
        element.parentNode.removeChild(element)

  def _RemoveAndroidLabel(self, node):
    """Remove android:label.

    We do this because it is not required by merger manifest,
    and it might contain @string references that will not allow compilation.

    Args:
      node: Node for which to remove Android labels.
    """
    if node.hasAttribute(self._ANDROID_LABEL):
      node.removeAttribute(self._ANDROID_LABEL)

  def _IsDuplicate(self, node_to_copy, node):
    """Is element a duplicate?"""
    for merger_node in self._merger_dom.getElementsByTagName(node_to_copy):
      if (merger_node.getAttribute(self._ANDROID_NAME) ==
          node.getAttribute(self._ANDROID_NAME)):
        return True
    return False

  def _RemoveIntentFilters(self, node):
    """Remove intent-filter in activity element.

    So there are no duplicate apps.

    Args:
      node: Node for which to remove intent filters.
    """
    intent_filters = node.getElementsByTagName(self._INTENT_FILTER)
    if intent_filters.length > 0:
      for sub_node in intent_filters:
        node.removeChild(sub_node)

  def _FindElementComment(self, node):
    """Find element's comment.

    Assumes that element's comment can be just above the element.
    Searches previous siblings and looks for the first non text element
    that is of a nodeType of comment node.

    Args:
      node: Node for which to find a comment.
    Returns:
      Elements's comment node, None if not found.
    """
    while node.previousSibling:
      node = node.previousSibling
      if node.nodeType is node.COMMENT_NODE:
        return node
      if node.nodeType is not node.TEXT_NODE:
        return None
    return None

  def _ReplaceArgumentPlaceholders(self, dom):
    """Replaces argument placeholders with their values.

    Modifies the attribute values of the input node.

    Args:
      dom: Xml node that should get placeholders replaced.
    """

    placeholders = {
        'packageName': self._merger_dom.getElementsByTagName(
            self._MANIFEST).item(0).getAttribute(self._PACKAGE),
    }

    for element in dom.getElementsByTagName('*'):
      for i in range(element.attributes.length):
        attr = element.attributes.item(i)
        attr.value = self._ReplaceArgumentHelper(placeholders, attr.value)

  def _ReplaceArgumentHelper(self, placeholders, attr):
    """Replaces argument placeholders within a single string.

    Args:
      placeholders: A dict mapping between placeholder names and their
                    replacement values.
      attr: A string in which to replace argument placeholders.

    Returns:
      A string with placeholders replaced, or the same string if no placeholders
      were found.
    """
    match_placeholder = '\\${([a-zA-Z]*)}'

    # Returns the replacement string for found matches.
    def PlaceholderReplacer(matchobj):
      found_placeholder = matchobj.group(1)
      if found_placeholder not in placeholders:
        raise UndefinedPlaceholderException(
            'Undefined placeholder when substituting arguments: '
            + found_placeholder)
      return placeholders[found_placeholder]

    attr = re.sub(match_placeholder, PlaceholderReplacer, attr)

    return attr

  def _SortAliases(self):
    applications = self._merger_dom.getElementsByTagName('application')
    if not applications:
      return
    for alias in applications[0].getElementsByTagName('activity-alias'):
      comment_node = self._FindElementComment(alias)
      while comment_node is not None:
        applications[0].appendChild(comment_node)
        comment_node = self._FindElementComment(alias)
      applications[0].appendChild(alias)

  def _FindMergerParent(self, tag_to_copy, destination_tag_name, mergee_dom):
    """Finds merger parent node, or appends mergee equivalent node if none."""
    # Merger parent element to which to add merged elements.
    if self._merger_dom.getElementsByTagName(destination_tag_name):
      return self._merger_dom.getElementsByTagName(destination_tag_name)[0]
    else:
      mergee_element = mergee_dom.getElementsByTagName(destination_tag_name)[0]
      # find the parent
      parents = self._merger_dom.getElementsByTagName(
          mergee_element.parentNode.tagName)
      if not parents:
        raise MalformedManifestException(
            'Malformed manifest has tag %s but no parent tag %s',
            (tag_to_copy, destination_tag_name))
      # append the mergee child as the first child.
      return parents[0].insertBefore(mergee_element, parents[0].firstChild)

  def _OrderManifestChildren(self):
    """Moves elements of the manifest tag into the correct order."""
    manifest = self._merger_dom.getElementsByTagName('manifest')[0]
    # The application element must be the last element in the manifest tag.
    applications = self._merger_dom.getElementsByTagName('application')
    if applications:
      manifest.appendChild(applications[0])

  def Merge(self):
    """Takes two manifests, and merges them together to produce a third."""
    self._RemoveFromMerger()
    self._ExpandPackageName(self._merger_dom)

    for dom, filename in self._mergees:
      mergee_dom = xml.dom.minidom.parseString(dom)
      self._ReplaceArgumentPlaceholders(mergee_dom)
      self._ExpandPackageName(mergee_dom)
      self._ApplyExcludePermissions(mergee_dom)

      for destination, values in sorted(
          self._NODES_TO_COPY_FROM_MERGEE.iteritems()):
        for node_to_copy in values:
          for node in mergee_dom.getElementsByTagName(node_to_copy):
            if self._IsDuplicate(node_to_copy, node):
              continue

            merger_parent = self._FindMergerParent(node_to_copy,
                                                   destination,
                                                   mergee_dom)

            # Append the merge comment.
            merger_parent.appendChild(
                self._merger_dom.createComment(' Merged from file: %s ' %
                                               filename))

            # Append mergee's comment, if present.
            comment_node = self._FindElementComment(node)
            if comment_node:
              merger_parent.appendChild(comment_node)

            # Append element from mergee to merger.
            merger_parent.appendChild(node)

    # Insert top level comment about the merge.
    top_comment = (
        ' *** WARNING *** DO NOT EDIT! THIS IS GENERATED MANIFEST BY '
        'MERGE_MANIFEST TOOL.\n'
        '  Merger manifest:\n    %s\n' % self._merger[1] +
        '  Mergee manifests:\n%s' % '\n'.join(
            ['    %s' % mergee[1] for mergee in self._mergees]) +
        '\n  ')
    manifest_element = self._merger_dom.getElementsByTagName('manifest')[0]
    manifest_element.insertBefore(self._merger_dom.createComment(top_comment),
                                  manifest_element.firstChild)

    self._SortAliases()
    self._OrderManifestChildren()
    return self._merger_dom.toprettyxml(indent='  ')


def _ReadFiles(files):
  results = []
  for file_name in files:
    results.append(_ReadFile(file_name))
  return results


def _ReadFile(file_name):
  with open(file_name, 'r') as my_file:
    return (my_file.read(), file_name,)


def _ValidateAndWarnPermissions(exclude_permissions):
  unknown_permissions = (
      set(exclude_permissions)
      - set([EXCLUDE_ALL_ARG])
      - android_permissions.PERMISSIONS)
  return '\n'.join([
      'WARNING:\n\t Specified permission "%s" is not a standard permission. '
      'Is it a typo?' % perm for perm in unknown_permissions])


def main():
  if not FLAGS.merger:
    raise RuntimeError('Missing merger value.\n' + USAGE)
  if len(FLAGS.mergee) < 1:
    raise RuntimeError('Missing mergee value.\n' + USAGE)
  if not FLAGS.output:
    raise RuntimeError('Missing output value.\n' + USAGE)
  if FLAGS.exclude_permission:
    warning = _ValidateAndWarnPermissions(FLAGS.exclude_permission)
    if warning:
      print warning

  merged_manifests = MergeManifests(_ReadFile(FLAGS.merger),
                                    _ReadFiles(FLAGS.mergee),
                                    FLAGS.exclude_permission
                                   ).Merge()

  with open(FLAGS.output, 'w') as out_file:
    for line in merged_manifests.split('\n'):
      if not line.strip():
        continue
      out_file.write(line.encode('utf8') + '\n')

if __name__ == '__main__':
  FLAGS(sys.argv)
  main()
