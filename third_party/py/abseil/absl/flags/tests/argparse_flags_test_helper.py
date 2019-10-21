# Copyright 2018 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test helper for argparse_flags_test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

from absl import app
from absl import flags
from absl.flags import argparse_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('absl_echo', None, 'The echo message from absl.flags.')


def parse_flags_simple(argv):
  """Simple example for absl.flags + argparse."""
  parser = argparse_flags.ArgumentParser(
      description='A simple example of argparse_flags.')
  parser.add_argument(
      '--argparse_echo', help='The echo message from argparse_flags')
  return parser.parse_args(argv[1:])


def main_simple(args):
  print('--absl_echo is', FLAGS.absl_echo)
  print('--argparse_echo is', args.argparse_echo)


def roll_dice(args):
  print('Rolled a dice:', random.randint(1, args.num_faces))


def shuffle(args):
  inputs = list(args.inputs)
  random.shuffle(inputs)
  print('Shuffled:', ' '.join(inputs))


def parse_flags_subcommands(argv):
  """Subcommands example for absl.flags + argparse."""
  parser = argparse_flags.ArgumentParser(
      description='A subcommands example of argparse_flags.')
  parser.add_argument('--argparse_echo',
                      help='The echo message from argparse_flags')

  subparsers = parser.add_subparsers(help='The command to execute.')

  roll_dice_parser = subparsers.add_parser(
      'roll_dice', help='Roll a dice.')
  roll_dice_parser.add_argument('--num_faces', type=int, default=6)
  roll_dice_parser.set_defaults(command=roll_dice)

  shuffle_parser = subparsers.add_parser(
      'shuffle', help='Shuffle inputs.')
  shuffle_parser.add_argument(
      'inputs', metavar='I', nargs='+', help='Inputs to shuffle.')
  shuffle_parser.set_defaults(command=shuffle)

  return parser.parse_args(argv[1:])


def main_subcommands(args):
  main_simple(args)
  args.command(args)


if __name__ == '__main__':
  main_func_name = os.environ['MAIN_FUNC']
  flags_parser_func_name = os.environ['FLAGS_PARSER_FUNC']
  app.run(main=globals()[main_func_name],
          flags_parser=globals()[flags_parser_func_name])
