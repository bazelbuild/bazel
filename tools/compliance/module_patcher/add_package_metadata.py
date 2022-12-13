#!/usr/bin/env python3
# vim: set sw=4 ts=4 :
"""Adds license() and package_info() rules to a repository.

This tool modifies the BUILD files in a source tree to add license targets.
The resulting structure will normally be

- A license target in the workspace root BUILD file
- All other build files point to the root license target via
  package.default_package_metadata.

The intended use is to modify a repository after downloading, but before
returning from the repository rule defining it.

This reference version implements the command line API for BazelRepoPatcher

- Program starts with the working directory as the repository to be patched.
- Command line args
  --top: may specify a differnt root for the repo
  --verbose: turns on verbose output
  name=value ... A set of name/value pairs.

"""

import argparse
# from collections.abc import Sequence
import os
import re
import sys


PACKAGE_RE = re.compile(r'^package\((?P<decls>[^)]*)\)', flags=re.MULTILINE)

def trim_extension(name: str) -> str:
    """Trim the well known package extenstions."""
    for ext in ('.deb', '.jar', '.tar', '.tar.bz2', '.tar.gz', '.gz', '.tgz', '.zip'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def guess_package_name_and_version(url: str) -> str:
    """Guess a package name from a URL."""
    # .../rules_x-2.3.5.tgz => rules_x
    basename = trim_extension(url.split('/')[-1])
    name = basename
    version = None
    parts = basename.split('-')
    if parts[-1][0].isdigit():
      name = '-'.join(parts[0:-1])
      version = parts[-1]
    # print(url, '=>', name, version)
    return name, version


class BuildRewriter(object):

    def __init__(self,
                 top: str,
                 copyright_notice: str = None,
                 license_file: str = None,
                 license_kinds: str = None,
                 package_name: str = None,
                 package_url: str = None,
                 package_version: str = None,
                 verbose: bool = False):
        if package_url:
            p_name, p_version = guess_package_name_and_version(package_url)
        else:
            p_name = os.path.basename(os.getcwd())
            p_version = None
        self.top = top
        self.verbose = verbose
        self.copyright_notice = copyright_notice
        self.license_file = license_file
        if license_kinds:
          self.license_kinds = license_kinds.split(',')
        else:
          self.license_kinds = None
        self.package_name = package_name or p_name
        self.package_url = package_url
        self.package_version = package_version or p_version

        # Collected by scanning the tree
        self.top_build = None
        self.other_builds = []
        self.license_files = []

    def print(self):
        print('  top:', self.top)
        print('  license_file:', self.license_file)
        print('  license_kinds:', self.license_kinds)
        print('  package_name:', self.package_name)
        print('  package_url:', self.package_url)
        print('  package_version:', self.package_version)
        print('  top_build:', self.top_build)
        print('  other_builds:', self.other_builds)

    def read_source_tree(self):
        # Gather BUILD and license files
        for root, dirs, files in os.walk(self.top):
            # Do not traverse into .git tree
            if '.git' in dirs:
                dirs.remove('.git')
            found_license = False
            notices = []
            for f in files:
                 # Normalized path (no leading ./) from top
                rel_path = os.path.join(root, f)[len(self.top)+1:]
                if f in ('BUILD', 'BUILD.bazel'):
                    build_path = os.path.join(root, f)
                    if root == self.top: 
                        self.top_build = build_path
                    else:
                        self.other_builds.append(build_path)
                if f.upper().startswith('LICENSE'):
                    found_license = True
                    self.license_files.append(rel_path)
                if f.upper().startswith('NOTICE'):
                    notices.append(rel_path)

                # FUTURE: Make this extensible so that users can add their own
                # scanning code. To modif

            if not found_license and notices:
                self.license_files.extend(notices)

    def adjust_source_tree(self):
        # If we did not find a root BUILD file, it should always be safe to
        # create one and put the license there.
        if not self.top_build:
            self.top_build = os.path.join(self.top, 'BUILD')
            with open(self.top_build, 'w', encoding='utf-8') as tmp:
                tmp.write('# This file was generated at WORKSPACE setup.\n')
                print("Created", self.top_build)

    def point_to_top_level_license(self, build_file: str):
        # Points the BUILD file at a path to the top level liceense declaration
        with open(build_file, 'r', encoding='utf-8') as inp:
            content = inp.read()
        new_content = add_default_package_metadata(content, "//:license", "//:package_info")
        if content != new_content:
            os.remove(build_file)
            with open(build_file, 'w', encoding='utf-8') as out:
               out.write(new_content)

    def select_license_file(self):
        if self.license_file:
            return
        if len(self.license_files) == 0:
            print('Warning: package %s at %s contains no potential license files.' %
                  (self.package_name, self.top))
            return
        self.license_file = self.license_files[0]
        if len(self.license_files) > 1:
            print('Warning: package', self.package_name, 'at', self.top,
                  'contains multiple potential license files:',
                  self.license_files)

    def create_license_target(self) -> str:
        """Creates the text of a license() target for this package."""

        target = [
            'license(',
            '    name = "license",',
        ]
        if self.copyright_notice:
            target.append('    copyright_notice = "%s",' % self.copyright_notice)
        if self.license_file:
            target.append('    license_text = "%s",' % self.license_file)
        if self.license_kinds:
            target.append('    license_kinds = [')
            for kind in self.license_kinds:
                target.append('        "%s",' % kind)
            target.append('    ],')
        else:
            target.append('    license_kinds = [],')
        target.append(')')
        return '\n'.join(target)

    def create_package_info_target(self) -> str:
        """Creates the text of a package_info() target for this package."""
        target = [
            'package_info(',
            '    name = "package_info",',
        ]
        if self.package_name:
            target.append('    package_name = "%s",' % self.package_name)
        if self.package_version:
            target.append('    package_version = "%s",' % self.package_version)
        if self.package_url:
            target.append('    package_url = "%s",' % self.package_url)
        target.append(')')
        return '\n'.join(target)

    def ensure_top_level_license(self, fallback_license, fallback_info):
        add_license(self.top_build, fallback_license, fallback_info)


def add_default_package_metadata(
    content: str,
    license_label: str,
    package_info_label: str) -> str:
    """Add a default_package_metadata clause to the package().

    Do not add if there already is one.
    Move package() to the first non-load statement.
    """
    # Build what the package statement should contain
    dal = 'default_package_metadata=["%s", "%s"],' % (
        license_label, package_info_label)
    m = PACKAGE_RE.search(content)
    if m:
        decls = m.group('decls')
        if decls.find('default_applicable_licenses') >= 0:
            # Do nothing
            return content
        if decls.find('default_package_metadata') >= 0:
            # Do nothing
            return content

        package_decl = '\n'.join([
            'package(',
            '    ' + decls.strip().rstrip(',') + ',',
            '    ' + dal,
            ')'])
        span = m.span(0)
        content = content[0:span[0]] + content[span[1]:]
    else:
        package_decl = 'package(%s)' % dal

    # Now splice it into the correct place. That is always before
    # any existing rules.
    ret = []
    package_added = False
    for line in content.split('\n'):
        if not package_added:
            t = line.strip()
            if (t
                and not line.startswith(' ')
                and not t.startswith('#')
                and not t.startswith(')')
                and not t.startswith('load')):
                  ret.append('')
                  ret.append(package_decl)
                  package_added = True
        ret.append(line)
    return '\n'.join(ret)


def add_license(build_file: str, license_target: str, info_target: str):
    # Points the BUILD file at a path to the top level liceense declaration
    with open(build_file, 'r', encoding='utf-8') as inp:
        content = inp.read()

    # Do not overwrite an existing one
    # TBD: We obviously have to be able to.
    if '\nlicense(' in content:  # )
        license_target = None
    if '\npackage_info(' in content:  # )
        info_target = None

    license_load = """load("@rules_license//rules:license.bzl", "license")"""
    must_add_load = not license_load in content

    info_load = """load("@rules_license//rules:package_info.bzl", "package_info")"""
    must_add_info_load = not info_load in content

    new_content = add_default_package_metadata(content, "//:license", "//:package_info")
    # Now splice it into the correct place. That is always before
    # any existing rules.
    ret = []
    license_added = False
    for line in new_content.split('\n'):
        if not license_added:
            t = line.strip()
            if t and not t.startswith('#'):
                if must_add_load:
                   ret.append(license_load)
                   must_add_load = False
                if must_add_info_load:
                   ret.append(info_load)
                   must_add_info_load = False
            if (t
                and not line.startswith(' ')
                and not t.startswith('#')
                and not t.startswith(')')
                and not t.startswith('load')
                and not t.startswith('package')):
                  ret.append('')
                  if license_target:
                      ret.append(license_target)
                  if info_target:
                      ret.append(info_target)
                  license_added = True
        ret.append(line)
    if not license_added:
        if must_add_load:
            ret.append(license_load)
        if must_add_info_load:
            ret.append(info_load)
        ret.append('')
        if license_target:
           ret.append(license_target)
        if info_target:
            ret.append(info_target)

    new_content = '\n'.join(ret)

    if content != new_content:
        os.remove(build_file)
        with open(build_file, 'w', encoding='utf-8') as out:
           out.write(new_content)

# def args_to_dict(args: Sequence[str]) -> dict:
def args_to_dict(args) -> dict:
    ret = {}
    for arg in args:
        tmp = arg.split('=')
        if len(tmp) < 2:
            print('Unparsable arg. Must be in the form name=value:',
                  arg,
                  file=sys.stderr)
            sys.exit(1)
        ret[tmp[0]] = '='.join(tmp[1:])
    return ret


# def main(argv: Sequence[str]) -> None:
def main(argv) -> None:
    parser = argparse.ArgumentParser(description='Add license targets')
    parser.add_argument('--top', type=str, help='Top of source tree')
    parser.add_argument('--verbose', action='store_true', help='Be verbose')
    parser.add_argument('info', nargs='+', help='name=value pairs')

    args = parser.parse_args()
    # XXX: Remove before submit.
    args.verbose = True
    if args.verbose:
        if args.top:
            print(' '.join(argv))
        else:
            print(argv[0], '--top=%s' % os.getcwd(), ' '.join(argv[1:]))

    try:
        if not isinstance(args.info, list):
            print("WTF?  Info not list", args.info)
        info = args_to_dict(args.info)
        if args.verbose:
            print(info)
    except Exception as e:
        print("WHAT?", args.info)

    rewriter = BuildRewriter(
        top=args.top or '.',
        copyright_notice = info.get('copyright_notice'),
        license_file = info.get('license_file'),
        license_kinds = info.get('license_kinds'),
        package_url = info.get('package_url'),
        package_name = info.get('package_name'),
        package_version = info.get('package_version'),
        verbose = args.verbose,
    )
    rewriter.read_source_tree()
    rewriter.adjust_source_tree()
    rewriter.select_license_file()
    if not rewriter.license_file:
        return
    if os.path.sep in rewriter.license_file:
        print('Did not find license file at top level.')
        return
    license = rewriter.create_license_target()
    package_info = rewriter.create_package_info_target()
    if args.verbose:
        rewriter.print()
        print('Synthesized license:', license)
        print('Synthesized info:', package_info)
    rewriter.ensure_top_level_license(
        fallback_license=license,
        fallback_info=package_info)
    for build_file in rewriter.other_builds:
        rewriter.point_to_top_level_license(build_file)


if __name__ == '__main__':
    main(sys.argv)
