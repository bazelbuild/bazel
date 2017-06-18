This is jarjar. They don't seem to tag their releases, so I just used whatever was at HEAD. Reproduction:

1. `git clone https://github.com/shevek/jarjar`
2. `git checkout 69d2972ea10eefa66cdc2c1c283cb6ef79b3c5ba`
3. Copy the git tree to `third_party/java/jarjar`
4. Keep the existing `BUILD` file
