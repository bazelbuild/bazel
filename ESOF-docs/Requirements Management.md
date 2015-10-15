# Relatório 2 - Gestão de Requesitos #
## TASK LIST ##
- [x] send email to bazel developpers
- [ ] writing the report
- [ ] use case diagram (see the notes at the end of this document)

## Elicitation ##

## Analysis and negotiation ##

## Specification ##

## Validation ##

## Informação ##
Why would I want to use Bazel?

Bazel may give you faster build times because it can recompile only the files that need to be recompiled. Similarly, it can skip re-running tests that it knows haven't changed.

Bazel produces deterministic results. This eliminates skew between incremental and clean builds, laptop and CI system, etc.

Bazel can build different client and server apps with the same tool from the same workspace. For example, you can change a client/server protocol in a single commit, and test that the updated mobile app works with the updated server, building both with the same tool, reaping all the aforementioned benefits of Bazel.

Can I see examples?

Yes. For a simple example, see:

https://github.com/bazelbuild/bazel/blob/master/examples/cpp/BUILD

The Bazel source code itself provides more complex examples:

https://github.com/bazelbuild/bazel/blob/master/src/main/java/BUILD\ https://github.com/bazelbuild/bazel/blob/master/src/test/java/BUILD

What is Bazel best at?

Bazel shines at building and testing projects with the following properties:

Projects with a large codebase
Projects written in (multiple) compiled languages
Projects that deploy on multiple platforms
Projects that have extensive tests

I think you assume that our process is a lot more formal than it actually is :) Feel free to get back to us if you have any other questions.

We don't really use any heavyweight methodology -- we have a roadmap that is essentially a set of promises about what feature will be delivered when, we set quarterly goals that are a bit too ambitious for comfort, and try to stick to the plan by doing as good coding as we possibly can. Both the roadmap and the quarterly goals are determined by a somewhat chaotic process involving a few meetings and usually a document editable by all of us.

The reason why don't really have any requirements process to speak of is twofold: 
Even if we mess up very badly, no one is going to die (unlike e.g. in the automotive or medical industries) and Bazel doesn't run on space probes where you can't just send a repairman, so we can't make uncorrectable mistakes. In areas where fixing things is hard (for example, with the syntax/semantics of Skylark), we actually have a bit of a problem.
We are software developers writing software for other software developers, so we understand our domain pretty well (unlike if I had to write e.g. software in finance, about which I don't know a lot)
With this in mind, see my answers inline.


- There were any other type of requirements besides the software ones? (Like quality and user requirements.)

As ashamed as I am to say, I don't really grasp what the difference is between a software, a quality and a user requirement. Okay, I'm lying, I guess a quality requirement is how good the software should be, and it was pretty much "it should be as good as possible".
 
- How did you proceed in the requirements elicitation? There were place for meetings? Questionnaires (surveys)? 

We do a lot of informal in-person conversations all the time and there are a few more formal meetings for setting our quarterly goals. The latter are somewhat painful, but in general, the process is pretty reasonable.
 
- Did you find any problems in the requirements? How did you surpass them? Which techniques did you use to analyze and negotiate them?
We talked to each other, and if we agreed that a particular idea is bad, we threw it out :)

- After this did you produce a SRS document? May I see it?
What is an SRS document? I guess this answers your question :) Software... Requirement... Something?
 
- How were the requirements validated? Where they inspected and reviewed?
Well, our quarterly goals go both up and down the management chain: we try to fit in with what the rest of Google does and if we decide to do something obviously bad, we are called out on that.

## Referências ##

Página oficial: http://bazel.io/ 

## Contribuições ##
* António Ramadas:
* João Guarda:
* Rui Vilares:
* Trabalhando em grupo:

## Autores ##

### Turma 1 - Grupo 4 ###

* António Manuel Vieira Ramadas
* João Diogo Trindade Guarda
* Rui Miguel Teixeira Vilares

Engenharia de Software (ESOF)

Faculdade de Engenharia da Universidade do Porto
18 de outubro de 2015
