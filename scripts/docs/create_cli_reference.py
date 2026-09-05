# Copyright 2026 The Bazel Authors. All rights reserved.
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
"""Converts command-line-reference proto to MDX with ParamField components."""

import argparse
import base64
import itertools
import textwrap

from src.main.protobuf import bazel_flags_pb2

_MDX_SPECIAL_CHARS = {
    '{': '&lcub;',
    '}': '&rcub;',
    '<': '&lt;',
    '>': '&gt;',
    '$': '&#36;',
}


def _sanitize_special_mdx_chars(text):
  for char, entity in _MDX_SPECIAL_CHARS.items():
    text = text.replace(char, entity)
  return text


_CATEGORY_DESCRIPTIONS = {
    'BAZEL_CLIENT_OPTIONS': 'Options that appear before the command and are parsed by the client',
    'EXECUTION_STRATEGY': 'Options that control build execution',
    'TOOLCHAIN': 'Options that configure the toolchain used for action execution',
    'OUTPUT_SELECTION': 'Options that control the output of the command',
    'OUTPUT_PARAMETERS': 'Options that let the user configure the intended output, affecting its value, as opposed to its existence',
    'INPUT_STRICTNESS': 'Options that affect how strictly Bazel enforces valid build inputs (rule definitions,  flag combinations, etc.)',
    'SIGNING': 'Options that affect the signing outputs of a build',
    'STARLARK_SEMANTICS': 'This option affects semantics of the Starlark language or the build API accessible to BUILD files, .bzl files, or WORKSPACE files.',
    'TESTING': 'Options that govern the behavior of the test environment or test runner',
    'QUERY': 'Options relating to query output and semantics',
    'MOD_COMMAND': 'Options relating to mod command output and semantics',
    'BZLMOD': 'Options relating to Bzlmod output and semantics',
    'BUILD_TIME_OPTIMIZATION': 'Options that trigger optimizations of the build time',
    'LOGGING': 'Options that affect the verbosity, format or location of logging',
    'GENERIC_INPUTS': 'Options specifying or altering a generic input to a Bazel command that does not fall into other categories.',
    'REMOTE': 'Remote caching and execution options',
    'UNCATEGORIZED': 'Miscellaneous options, not otherwise categorized.',
}

_FLAG_TEMPLATE = """<ParamField path="--{has_negative_flag}{name}" type="{type_converter}"{default_value}{deprecation}>
{abbreviation}{documentation}{expansions}{allows_multiple}{tags}
</ParamField>
"""

_MAIN_TEMPLATE = """## Commands

| | |
| --- | --- |
{commands}

{all_flags}

### Option Effect Tags

| | |
| --- | --- |
{effect_tags}

### Option Metadata Tags

| | |
| --- | --- |
{metadata_tags}
"""

_SECTION_TEMPLATE = """## {heading}{anchor}
{inherits}{flags}
"""

_TAG_TABLE_TEMPLATE = (
    '| <span id="{type}_tag_{name_upper}">`{name}`</span> | {description} |'
)

_COMMAND_TEMPLATE = '| [`{name}`](#{name}) | {description} |'


_HIDDEN_FLAG_TAGS = set({'HIDDEN', 'INTERNAL'})

class Tag:

  def __init__(self, type, name, description):
    self.type = type
    self.name = name
    self.description = description

  def to_link(self):
    return f'[`{self.name}`](#{self.type}_tag_{self.name.upper()})'

  def to_table_entry(self):
    return _TAG_TABLE_TEMPLATE.format(
        type=self.type,
        name=self.name,
        name_upper=self.name.upper(),
        description=self.description,
    )


class Flag:

  def __init__(self, flag_info, known_tags):
    self.flag_info = flag_info
    self._tags = tuple(
        known_tags[tag.lower()]
        for tag in itertools.chain(
            flag_info.effect_tags, flag_info.metadata_tags
        )
        if tag != 'UNKNOWN'
    )

  @staticmethod
  def should_document(flag_info) -> bool:
      if not _HIDDEN_FLAG_TAGS.isdisjoint(flag_info.metadata_tags):
        return False
      if flag_info.documentation_category == 'UNDOCUMENTED':
        return False

      return True

  @property
  def name(self) -> str:
    return self.flag_info.name

  @property
  def documentation(self) -> str:
    text = self.flag_info.documentation.strip()
    text = text.replace('%{product}', 'bazel')
    return textwrap.indent(text, "  ")

  @property
  def has_negative_flag(self) -> str:
    return '[no]' if self.flag_info.has_negative_flag else ''

  @property
  def abbreviation(self) -> str:
    if not self.flag_info.abbreviation:
      return ''
    return f'Short form: `{self.flag_info.abbreviation}`\n\n'

  @property
  def commands(self) -> list[str]:
    return list(self.flag_info.commands)

  @property
  def default_value(self) -> str:
    if not self.flag_info.default_value:
      return ""
    val = _sanitize_special_mdx_chars(self.flag_info.default_value)
    return f' default="{val}"'

  @property
  def type_converter(self) -> str:
    raw = self.flag_info.type_description or self.flag_info.type_converter
    return _sanitize_special_mdx_chars(raw)

  @property
  def tags(self):
    if not self._tags:
      return ''
    return '\n\n  Tags: {}'.format(', '.join(t.to_link() for t in self._tags))

  @property
  def allows_multiple(self):
    if not self.flag_info.allows_multiple:
      return ''
    return '\n\n  *May be used multiple times; values are accumulated.*'

  @property
  def expansions(self) -> str:
    exps = list(self.flag_info.option_expansions)
    if not exps:
      return ''
    items = '\n'.join(f'  - `{e}`' for e in exps)
    return f'\n\n  Expands to:\n{items}'

  @property
  def deprecation(self):
    if (self.flag_info.deprecation_warning
        or 'DEPRECATED' in self.flag_info.metadata_tags):
      return ' deprecated'
    return ''

  def __str__(self):
    return _FLAG_TEMPLATE.format(
        name=self.name,
        documentation=self.documentation,
        has_negative_flag=self.has_negative_flag,
        abbreviation=self.abbreviation,
        default_value=self.default_value,
        type_converter=self.type_converter,
        deprecation=self.deprecation,
        expansions=self.expansions,
        allows_multiple=self.allows_multiple,
        tags=self.tags,
    )


class Section:

  def __init__(self, heading, direct_flags, config_flags=(),
               anchor=None, inherits_from=None):
    self._heading = heading
    self._anchor = anchor
    self._direct_flags = list(direct_flags)
    self._config_flags = list(config_flags)
    self._inherits_from = inherits_from or []

  @property
  def anchor(self) -> str:
    if not self._anchor:
      return ''
    return f' {{#{self._anchor}}}'

  @property
  def inherits(self) -> str:
    if not self._inherits_from:
      return ''
    refs = ' and '.join(f'[{p}](#{p})' for p in self._inherits_from)
    return f'\nInherits all options from {refs}.\n'

  @staticmethod
  def _render_flags_by_category(flags):
    by_category = {}
    for f in flags:
      by_category.setdefault(f.flag_info.documentation_category, []).append(f)

    sections = []
    for cat, desc in _CATEGORY_DESCRIPTIONS.items():
      cat_flags = by_category.get(cat, [])
      if not cat_flags:
        continue
      cat_flags.sort(key=lambda f: f.name)
      sections.append(f'\n{desc}\n')
      sections.extend(str(f) for f in cat_flags)
    return '\n'.join(sections)

  @property
  def flags(self) -> str:
    return '\n'.join(
        s for s in (
            self._render_flags_by_category(self._direct_flags),
            self._render_flags_by_category(self._config_flags),
        ) if s
    )

  def render(self) -> str:
    return _SECTION_TEMPLATE.format(
        heading=self._heading,
        anchor=self.anchor,
        inherits=self.inherits,
        flags=self.flags,
    )


class CliRefBuilder:

  def __init__(self, flag_collection):
    self._commands = list(flag_collection.commands)
    self._effect_tags = tuple(
        Tag('effect', name, desc)
        for name, desc in flag_collection.effect_tags.items()
    )
    self._metadata_tags = tuple(
        Tag('metadata', name, desc)
        for name, desc in flag_collection.metadata_tags.items()
    )
    known_tags = {
        tag.name: tag
        for tag in itertools.chain(self._effect_tags, self._metadata_tags)
    }

    all_flags = [
      Flag(f, known_tags) for f in flag_collection.flag_infos
      if Flag.should_document(f)
    ]
    self._sections = self._build_sections(all_flags)

  def _build_sections(self, all_flags):
    flags_by_section = {}
    for f in all_flags:
      for section in f.flag_info.sections:
        flags_by_section.setdefault(section, []).append(f)

    sections = [
        Section('Startup Options', flags_by_section.get('startup', [])),
        Section('Options Common to all Commands',
                flags_by_section.get('common', []),
                anchor='common_options'),
    ]

    config_flags = flags_by_section.get('config', [])
    for cmd in self._commands:
      heading = cmd.name[0].upper() + cmd.name[1:]
      sections.append(Section(
          f'{heading} Options',
          flags_by_section.get(cmd.name, []),
          config_flags=config_flags if cmd.uses_configuration_options else (),
          anchor=cmd.name,
          inherits_from=list(cmd.inherits_options_from),
      ))
    return sections

  def render_commands(self) -> str:
    return '\n'.join(
        _COMMAND_TEMPLATE.format(name=cmd.name, description=cmd.description)
        for cmd in self._commands
    )

  def render_all_flags(self) -> str:
    return '\n'.join(s.render() for s in self._sections)

  def render_effect_tags(self) -> str:
    return '\n'.join(tag.to_table_entry() for tag in self._effect_tags)

  def render_metadata_tags(self) -> str:
    return '\n'.join(tag.to_table_entry() for tag in self._metadata_tags)

  def render(self) -> str:
    return _MAIN_TEMPLATE.format(
        commands=self.render_commands(),
        all_flags=self.render_all_flags(),
        effect_tags=self.render_effect_tags(),
        metadata_tags=self.render_metadata_tags(),
    )


def convert(b64_proto):
  """A base64 encoded FlagCollection into .mdx optimized for Acorn.

  Args:
    b64_proto: str; a base64 encoded FlagCollection proto.

  Returns:
    The MDX content with ParamField components.
  """
  options_msg = bazel_flags_pb2.FlagCollection()
  options_msg.ParseFromString(base64.b64decode(b64_proto))
  cli_ref = CliRefBuilder(options_msg)
  return cli_ref.render()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Convert a base64-encoded FlagCollection proto to MDX.',
  )
  parser.add_argument(
      'src',
      type=argparse.FileType('r'),
      help='Path to read base64-encoded proto from. Defaults to stdin.',
  )
  parser.add_argument(
      '-o', '--out',
      dest='dest',
      type=argparse.FileType('w'),
      default='-',
      help='Path to write MDX output to. Defaults to stdout.',
  )
  return parser.parse_args()


def generate_cli_reference(src, dest):
  b64_proto = src.read()
  mdx = convert(b64_proto)
  dest.write(mdx)


if __name__ == '__main__':
  generate_cli_reference(**vars(_parse_args()))
