Project: /_project.yaml
Book: /_book.yaml

<style>
  .padbottom { padding-bottom: 10px; }
  .etabox {
    background: #EFEFEF;
    color: #38761D;
    font-size: 15px;
    font-weight: bold;
    display: inline;
    padding: 6px;
    margin-right: 10px;
  }
  .donestatus {
    color: #00D000;
    font-weight: bold;
    padding-left: 10px;
  }
  .inprogressstatus {
    color: #D0D000;
    font-weight: bold;
    padding-left: 10px;
  }
  .notstartedstatus {
    color: #D00000;
    font-weight: bold;
    padding-left: 10px;
  }
</style>

# Bazel Configurability 2021 Roadmap

{% include "_buttons.html" %}

*Last verified: 2021-01-25* ([update history](https://github.com/bazelbuild/bazel-website/commits/master/roadmaps/configuration.md))

*Point of contact:* [gregestren](https://github.com/gregestren)

*Discuss:*  [Configurability roadmap: discussion](https://github.com/bazelbuild/bazel/issues/6431)

## Previous Roadmaps

* [2020](https://github.com/bazelbuild/bazel-website/blob/master/roadmaps/2020/configuration.md) (w/ EOY review)
* [2019](https://github.com/bazelbuild/bazel-website/blob/master/roadmaps/2019/configuration.md) (w/ EOY review)
* [2018](https://github.com/bazelbuild/bazel-website/blob/master/roadmaps/2018/configuration.md)

## Goal

`$ bazel build //:all` *just works*, for any project and any platforms.

* Builds don't require command-line flags.
* Each target automatically uses correct settings (such as `android_binary` uses the right NDK).
* It's easy to build for multiple platforms.
* Builds scale well, particularly w.r.t graph size and action caching.

We also support
[`cquery`](https://bazel.build/query/cquery), [`Starlark
configuration`](https://bazel.build/extending/config),
and
[`select()`](https://bazel.build/docs/configurable-attributes).

## Roadmap

Dates are approximate based on our best understanding of problem complexity
and developer availability. In 2021 we intend to focus more effort on fewer
projects at once. We'll only set ETAs for actively prioritized work in the
interest of accurate expectations.

### Platforms

<div class="padbottom"></div>
<span class="etabox">Q3 2021</span>**Android rules use the new [platforms
API](https://bazel.build/concepts/platforms)**
<span class="inprogressstatus">IN PROGRESS</span> ([#11749](https://github.com/bazelbuild/bazel/issues/11749))

* This is our main priority for the beginning of 2021.

<div class="padbottom"></div>
<span class="etabox">Q3 2021</span>**Builds support [multiple execution
platforms](https://docs.google.com/document/d/1U9HzdDmtRnm244CaRM6JV-q2408mbNODAMewcGjnnbM/)**
<span class="inprogressstatus">IN PROGRESS</span> ([#11748](https://github.com/bazelbuild/bazel/issues/11748))

<div class="padbottom"></div>
<span class="etabox">paused</span>**C++ rules use the new [platformsfall API](https://bazel.build/concepts/platforms)**
<span class="inprogressstatus">IN PROGRESS</span> ([#6516](https://github.com/bazelbuild/bazel/issues/6516))

* This is blocked on Android platforms. We can turn this on with a simple flag flip.

<div class="padbottom"></div>
<span class="etabox">paused</span>**Multi-platform targets**
<span class="notstartedstatus">NOT STARTED</span>

* Let targets declare that they should build for multiple platforms
* Listed here because of user request

<div class="padbottom"></div>
<span class="etabox">paused</span>**Deprecate and remove `--cpu` and related flags**
<span class="notstartedstatus">NOT STARTED</span>

* This is an aspirational goal that falls out of migrating all rules to platforms.

### Efficiency

<div class="padbottom"></div>
<span class="etabox">2021</span>**An experimental Bazel mode caches
cross-platform Java compilation**
<span class="inprogressstatus">IN PROGRESS</span> ([#6526](https://github.com/bazelbuild/bazel/issues/6526))

* Improves multi-platform build speed
* Underallocated, so progress is slow
