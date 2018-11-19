#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, json, urllib2


release_number = 20

def search_all_with_label(label):
    return "https://github.com/bazelbuild/bazel/issues?q=is%%3Aissue+label%%3A%s" % label


issue_report_format = "   - [--%s](https://github.com/bazelbuild/bazel/issues/%s)"

url_breaking = "https://api.github.com/repos/bazelbuild/bazel/issues?state=all&labels=breaking-change-0.%d" % (release_number)
search_url_breaking = search_all_with_label("breaking-change-0.%d" % release_number)

url_breaking_next = "https://api.github.com/repos/bazelbuild/bazel/issues?state=all&labels=breaking-change-0.%d" % (release_number + 1)
search_url_breaking_next = search_all_with_label("breaking-change-0.%d" % (release_number + 1))

url_migration = "https://api.github.com/repos/bazelbuild/bazel/issues?state=all&labels=migration-0.%d" % (release_number)
search_url_migration = search_all_with_label("migration-0.%d" % release_number)


def load_json(url):
  response = urllib2.urlopen(url).read()
  return json.loads(response)


print("[Breaking changes in 0.%d](%s)" % (release_number,search_url_breaking))
for issue in load_json(url_breaking):
  flag = issue["title"].split(":")[0]
  print(issue_report_format % (flag, issue["number"]))

print("[0.%d is a migration window for the following changes](%s)" % (release_number, search_url_migration))
for issue in load_json(url_migration):
  flag = issue["title"].split(":")[0]
  print(issue_report_format % (flag, issue["number"]))

print("[Breaking changes in the next release (0.%d)](%s)" % (release_number + 1, search_url_breaking_next))
for issue in load_json(url_breaking_next):
  flag = issue["title"].split(":")[0]
  print(issue_report_format % (flag, issue["number"]))
