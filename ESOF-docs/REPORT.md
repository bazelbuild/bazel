# ESOF #


## Project description ##

[Bazel] (http://bazel.io/) is an open source project from Google aiming to provide faster _builds_ of software. It is still in _beta_, but the goal is to achieve a stable version in May of 2016. The full detailed roadmap can be checked [here] (http://bazel.io/roadmap.html).
+Everyone is welcome to support the project, but it is required to fulfill all the [requirements] (http://bazel.io/contributing.html). The programming language used to program the _BUILD files_ is [Skylark] (http://bazel.io/docs/skylark/concepts.html) which "is a superset of the core build language and its syntax is a subset of Python". When the project finishes, it will support [multiple programming languages] (http://bazel.io/docs/build-encyclopedia.html#Rules) (such as Java or C++). To get started, please check [here] (http://bazel.io/docs/getting-started.html) how to do it.

## Software Process ##

The software process used in Bazel project is "Incremental development and delivery". After a brief conversation with the contributors of this project, they stated that the [Bazaar model] (https://en.wikipedia.org/wiki/The_Cathedral_and_the_Bazaar) is a good description of the software process used, and that usually falls under the Agile methodology category. They also claimed that Google uses OKRs for planning and expressing what the company, a team and each individual engineer wants to focus on in the next quarter. We can find a good presentation about this process on the [Google Ventures] (http://www.gv.com/lib/how-google-sets-goals-objectives-and-key-results-okrs) website.
Incremental development and delevery process constructs the system in increments and evaluates each increment before proceeding to the development of the next increment.

## Comment about Software Process used ##

Benefits:
* The cost of accommodating changing customer
requirements is reduced.
* Less documentation to change
* Unstable requirements can be left for later stages of development
* More frequent and early customer feedback
* Customer value can be delivered with each increment so
system functionality is available earlier.
* Early increments act as a prototype to help elicit
requirements for later increments.
* Lower risk of overall project failure.
* The highest priority system services tend to receive the
most testing.

Problems:
* System structure tends to degrade as new increments are
added.
* Unless time and money is spent on refactoring to improve the
software, regular change tends to corrupt its structure. Incorporating
further software changes becomes increasingly difficult and costly.
* It can be hard to identify upfront common facilities that are
needed by all increments, so level of reuse may be suboptimal.
* Incremental delivery may not be possible for replacement
systems as increments have less functionality than the system
being replaced.
* The nature of incremental development of the specification
together with the software may be not be adequate for
establishing a development contract at the begin.
