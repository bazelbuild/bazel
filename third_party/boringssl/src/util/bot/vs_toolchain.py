# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import json
import os
import pipes
import shutil
import subprocess
import sys


script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'gyp', 'pylib'))
json_data_file = os.path.join(script_dir, 'win_toolchain.json')


import gyp


def SetEnvironmentAndGetRuntimeDllDirs():
  """Sets up os.environ to use the depot_tools VS toolchain with gyp, and
  returns the location of the VS runtime DLLs so they can be copied into
  the output directory after gyp generation.
  """
  vs2013_runtime_dll_dirs = None
  depot_tools_win_toolchain = \
      bool(int(os.environ.get('DEPOT_TOOLS_WIN_TOOLCHAIN', '1')))
  if sys.platform in ('win32', 'cygwin') and depot_tools_win_toolchain:
    if not os.path.exists(json_data_file):
      Update()
    with open(json_data_file, 'r') as tempf:
      toolchain_data = json.load(tempf)

    toolchain = toolchain_data['path']
    version = toolchain_data['version']
    win_sdk = toolchain_data.get('win_sdk')
    if not win_sdk:
      win_sdk = toolchain_data['win8sdk']
    wdk = toolchain_data['wdk']
    # TODO(scottmg): The order unfortunately matters in these. They should be
    # split into separate keys for x86 and x64. (See CopyVsRuntimeDlls call
    # below). http://crbug.com/345992
    vs2013_runtime_dll_dirs = toolchain_data['runtime_dirs']

    os.environ['GYP_MSVS_OVERRIDE_PATH'] = toolchain
    os.environ['GYP_MSVS_VERSION'] = version
    # We need to make sure windows_sdk_path is set to the automated
    # toolchain values in GYP_DEFINES, but don't want to override any
    # otheroptions.express
    # values there.
    gyp_defines_dict = gyp.NameValueListToDict(gyp.ShlexEnv('GYP_DEFINES'))
    gyp_defines_dict['windows_sdk_path'] = win_sdk
    os.environ['GYP_DEFINES'] = ' '.join('%s=%s' % (k, pipes.quote(str(v)))
        for k, v in gyp_defines_dict.iteritems())
    os.environ['WINDOWSSDKDIR'] = win_sdk
    os.environ['WDK_DIR'] = wdk
    # Include the VS runtime in the PATH in case it's not machine-installed.
    runtime_path = ';'.join(vs2013_runtime_dll_dirs)
    os.environ['PATH'] = runtime_path + ';' + os.environ['PATH']
  return vs2013_runtime_dll_dirs


def _GetDesiredVsToolchainHashes():
  """Load a list of SHA1s corresponding to the toolchains that we want installed
  to build with."""
  # Use Chromium's VS2013.
  return ['ee7d718ec60c2dc5d255bbe325909c2021a7efef']


def FindDepotTools():
  """Returns the path to depot_tools in $PATH."""
  for path in os.environ['PATH'].split(os.pathsep):
    if os.path.isfile(os.path.join(path, 'gclient.py')):
      return path
  raise Exception("depot_tools not found!")


def Update():
  """Requests an update of the toolchain to the specific hashes we have at
  this revision. The update outputs a .json of the various configuration
  information required to pass to gyp which we use in |GetToolchainDir()|.
  """
  depot_tools_win_toolchain = \
      bool(int(os.environ.get('DEPOT_TOOLS_WIN_TOOLCHAIN', '1')))
  if sys.platform in ('win32', 'cygwin') and depot_tools_win_toolchain:
    depot_tools_path = FindDepotTools()
    get_toolchain_args = [
        sys.executable,
        os.path.join(depot_tools_path,
                    'win_toolchain',
                    'get_toolchain_if_necessary.py'),
        '--output-json', json_data_file,
      ] + _GetDesiredVsToolchainHashes()
    subprocess.check_call(get_toolchain_args)

  return 0


def main():
  if not sys.platform.startswith(('win32', 'cygwin')):
    return 0
  commands = {
      'update': Update,
  }
  if len(sys.argv) < 2 or sys.argv[1] not in commands:
    print >>sys.stderr, 'Expected one of: %s' % ', '.join(commands)
    return 1
  return commands[sys.argv[1]](*sys.argv[2:])


if __name__ == '__main__':
  sys.exit(main())
